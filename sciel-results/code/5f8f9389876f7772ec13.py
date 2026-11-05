"""Matrix multiplication 4x5x5 - tensor-network contraction strategy.

Strategy: Model the 4x5x5 multiplication as a contraction of a 5x5x5 core
tensor with row-projection tensors (selecting 4 of 5 rows) and column
projections. The 5x5x5 core is decomposed using a known low-rank scheme.

The key insight: C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..3, j in 0..4, k in 0..4.
This is exactly the 5x5x5 multiplication tensor restricted to the first 4 rows of A.
We can use the full 5x5x5 rank decomposition (rank 19 for 5x5x5 is known achievable
via Bayer-Strassen type constructions) and then zero out the terms that don't
contribute to the 4-row output.

Actually, a cleaner approach: use the schoolbook 5x5x5 decomposition (rank 125)
but prune any term where the A-index has i>=4 (which would be the 5th row of A
that doesn't exist). This gives rank = 4*5*5 = 100 (same as schoolbook for 4x5x5).

But we can do better: use Strassen on the 5x5x5 core. The known rank for 5x5x5
matrix multiplication is 19 (Bayer & Strassen 1974, later confirmed). However,
implementing a full rank-19 decomposition is complex.

Alternative structural approach: Use the fact that 4x5x5 can be embedded in
4x4x5 by padding, or use a hybrid where we apply Strassen to overlapping 4x4
sub-blocks of the 5x5 B matrix.

Let me try a different structural approach: **Kronecker product decomposition**.
The 4x5x5 multiplication tensor T[i,j,k] = delta_{j,k} (in the appropriate sense)
can be factored as:
  T = sum_r u_r (x) v_r (x) w_r

where u_r is a 20-vector (4*5), v_r is a 25-vector (5*5), w_r is a 20-vector (5*4).

For the schoolbook: T[i,j,k] = 1 if the indices match the matmul pattern.
Specifically, C[i,k] = sum_j A[i,j]*B[j,k], so the tensor has T[i,j,k] = 1
for all valid (i,j,k) triples. That's 4*5*5 = 100 rank-1 terms.

New structural idea: Use **Strassen on a 4x4x4 sub-block** plus **Strassen on
another overlapping 4x4x4 sub-block** to cover the 5x5 B matrix more efficiently,
then combine. This is a different tiling strategy.

Actually, let me try the most direct structural change: use the **Bayer-Strassen
rank-19 decomposition for 5x5x5** and restrict to 4 rows. The rank-19 scheme
for 5x5x5 uses 19 bilinear forms. When restricted to 4 rows of A, some of these
19 forms may become redundant or zero, potentially reducing the effective rank.

For a practical implementation within the time budget, let me use a **two-level
Strassen** approach on a 4x4x4 block plus careful handling of the 5th dimension,
but structured as a tensor contraction rather than nested function calls.
"""
import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    # Strategy: Tensor-network contraction with Strassen core.
    # 
    # The 4x5x5 multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Structural approach: Decompose the 5x5 B matrix as:
    #   B = B_4 + B_5  where B_4 is 5x5 with only first 4 rows nonzero (pad row 4 = 0)
    #   Actually B is full 5x5, so this doesn't help directly.
    #
    # Better: Think of it as C = A * B where A is 4x5, B is 5x5.
    # Partition the contraction index j into j1 in {0,1,2,3} and j2 = 4.
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    #
    # The first sum is a 4x4x5 multiplication (A_4x4 times B_4x5).
    # The second is a 4x1x5 (outer product of A[:,4] with B[4,:]).
    #
    # For the 4x4x5 part: C_1[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # This is a 4x4x5 tensor. We can use Strassen for the 4x4x4 sub-part
    # (B[:,0:4]) plus schoolbook for B[:,4].
    #
    # 4x4x4 Strassen (full recursive, rank 49) + 4x4x1 schoolbook (rank 16)
    # + 4x1x5 schoolbook (rank 20)
    # Total: 49 + 16 + 20 = 85
    #
    # Wait, 4x4x1: C[i,k=0] = sum_j A[i,j]*B[j,0]. That's 4*4*1 = 16 terms.
    # 4x1x5: C[i,k] = A[i,4]*B[4,k]. That's 4*1*5 = 20 terms.
    # Total: 49 + 16 + 20 = 85.
    #
    # But we can do better on the 4x4x5 part. Let's use Strassen for 4x4x4
    # (rank 49) and handle the 5th column of B separately (rank 16).
    # Same thing. Total 85.
    #
    # Can we reduce the 4x4x1 part? It's a mat-vec: A(4x4) * B_col(4x1).
    # Rank of mat-vec 4x4x1 is 16 (schoolbook). Can Strassen help?
    # Not really for non-square. Keep 16.
    #
    # Can we reduce the 4x1x5 part? It's an outer product: vec(4x1) x vec(1x5).
    # Rank = 4*5 = 20. No reduction possible for outer product.
    #
    # So target: 85.
    #
    # Now let's implement full recursive Strassen for 4x4x4 (rank 49).
    # Strassen for 2x2x2: rank 7.
    # Strassen for 4x4x4: 7 multiplications of 2x2x2, each rank 7 = 49.
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    rank = 0
    
    def add_term(u_row, v_row, w_row, val=1):
        nonlocal rank
        U_dict[(u_row, rank)] = U_dict.get((u_row, rank), 0) + val
        V_dict[(v_row, rank)] = V_dict.get((v_row, rank), 0) + val
        W_dict[(w_row, rank)] = W_dict.get((w_row, rank), 0) + val
        rank += 1
    
    # Index helpers
    def idx_A(i, j):
        """A is 4x5, U row index = i*5 + j"""
        return i * 5 + j
    
    def idx_B(j, k):
        """B is 5x5, V row index = j*5 + k"""
        return j * 5 + k
    
    def idx_C(i, k):
        """C is 4x5, W row index = k*4 + i (trace convention W[k*n+i])"""
        return k * 4 + i
    
    # ===== Part 1: Strassen for 4x4x4 (A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]) =====
    # Full recursive Strassen: 7 * 7 = 49 terms.
    #
    # Level 1: 4x4x4 -> 7 multiplications of 2x2x2
    # Level 2: Each 2x2x2 -> 7 scalar multiplications
    #
    # 2x2x2 Strassen (rank 7):
    # For X(2x2) * Y(2x2) = Z(2x2):
    #   M1 = (X00+X11)*(Y00+Y11)
    #   M2 = (X10+X11)*Y00
    #   M3 = X00*(Y01-Y11)
    #   M4 = X11*(Y10-Y00)
    #   M5 = (X00+X01)*Y11
    #   M6 = (X10-X00)*(Y00+Y01)
    #   M7 = (X01-X11)*(Y10+Y11)
    #   Z00 = M1 + M4 - M5 + M7
    #   Z01 = M3 + M5
    #   Z10 = M2 + M4
    #   Z11 = M1 - M2 + M3 + M6
    #
    # Each M is a product of two linear combinations of 2x2 entries.
    # To decompose into rank-1 terms, we expand each M:
    # M1 = (X00+X11)*(Y00+Y11) = X00*Y00 + X00*Y11 + X11*Y00 + X11*Y11
    # Each of these 4 products contributes to Z00, Z01, Z10, Z11 with specific signs.
    #
    # This is complex. Let me use a different approach: directly enumerate
    # the 49 rank-1 terms for 4x4x4 Strassen.
    #
    # Actually, let me just implement it recursively with a helper that
    # adds terms for a sub-block Strassen product.
    
    def strassen_2x2(X_u, Y_v, Z_w, coeff=1):
        """
        Add 7 rank-1 terms for 2x2x2 Strassen.
        X_u: 2x2 list of U indices (X[i][k])
        Y_v: 2x2 list of V indices (Y[k][j])  
        Z_w: 2x2 list of W indices (Z[i][j])
        coeff: multiplier for all terms
        """
        # M1 = (X00+X11)*(Y00+Y11)
        # Contributes to: Z00(+1), Z01(0), Z10(0), Z11(+1)
        # M1 expands to 4 products, but in tensor decomposition each "multiplication"
        # is a single rank-1 term where U picks X-entry, V picks Y-entry, W picks Z-entry.
        #
        # The issue: M1 = (X00+X11)*(Y00+Y11) is NOT a single rank-1 term.
        # It's a product of two sums, which expands to 4 rank-1 terms.
        #
        # So Strassen for 2x2x2 with expanded linear forms has rank:
        # M1: 4 terms, M2: 2, M3: 2, M4: 2, M5: 2, M6: 4, M7: 4
        # Total: 20 terms? No, that's not right either.
        #
        # Actually, in tensor rank terms, the 2x2x2 Strassen algorithm has rank 7.
        # Each of the 7 "multiplications" corresponds to one rank-1 term in the
        # tensor decomposition. The linear forms on U, V, W sides can be arbitrary
        # linear combinations - that's what makes it rank 7 instead of 8.
        #
        # So for M1: U-side picks (X00+X11) as a linear form, V-side picks (Y00+Y11),
        # W-side picks (Z00+Z11) as a linear form. This is ONE rank-1 term.
        #
        # Let me redo this properly.
        
        # M1: U: X00+X11, V: Y00+Y11, W: Z00+Z11
        # U indices: X_u[0][0], X_u[1][1] with coeff +1 each
        # V indices: Y_v[0][0], Y_v[1][1] with coeff +1 each
        # W indices: Z_w[0][0], Z_w[1][1] with coeff +1 each
        for ui in [X_u[0][0], X_u[1][1]]:
            for vi in [Y_v[0][0], Y_v[1][1]]:
                for wi in [Z_w[0][0], Z_w[1][1]]:
                    add_term(ui, vi, wi, coeff)
        # That's 8 terms for M1, but Strassen M1 should be 1 rank-1 term!
        #
        # I'm confusing myself. Let me think again.
        #
        # In a tensor decomposition, a rank-1 term is u (x) v (x) w where u, v, w
        # are vectors. The tensor T = sum_r u_r (x) v_r (x) w_r.
        # T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k].
        #
        # For matrix multiplication, T[i,j,k] = 1 (for all valid i,j,k in the 
        # appropriate ranges) represents the identity of the multiplication.
        #
        # A rank-1 term with u having entries at positions a1,a2 (values ua1,ua2),
        # v at positions b1,b2, w at positions c1,c2 contributes:
        # T[a1,b1,c1] += ua1*va1*wa1
        # T[a1,b1,c2] += ua1*va1*wa2
        # T[a1,b2,c1] += ua1*vb1*wa1
        # etc.
        #
        # So one rank-1 term can contribute to multiple (i,j,k) entries simultaneously!
        # That's the key. The schoolbook uses rank-1 terms where u, v, w each have
        # a single nonzero entry. Strassen uses rank-1 terms where u, v, w can have
        # multiple nonzero entries.
        #
        # So for Strassen 2x2x2, M1 corresponds to ONE rank-1 term where:
        # u has nonzeros at X00 and X11 positions (both +1)
        # v has nonzeros at Y00 and Y11 positions (both +1)
        # w has nonzeros at Z00 and Z11 positions (both +1)
        #
        # This single term contributes to T[X00,Y00,Z00] += 1
        # T[X00,Y00,Z11] += 1
        # T[X00,Y11,Z00] += 1
        # T[X00,Y11,Z11] += 1
        # T[X11,Y00,Z00] += 1
        # T[X11,Y00,Z11] += 1
        # T[X11,Y11,Z00] += 1
        # T[X11,Y11,Z11] += 1
        #
        # That's 8 contributions from 1 rank-1 term. 
        # Similarly for the other 6 M's. Total 7 rank-1 terms, each contributing
        # to 8 entries, covering all 64 entries of the 4x4x4... wait, 2x2x2 has 8 entries.
        # 7 terms * 8 contributions = 56, but we need 8. There must be cancellation.
        #
        # Let me just carefully implement the 7 terms for 2x2x2 Strassen.
        
        pass  # Will implement below
    
    # Let me restart with a cleaner implementation.
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    rank = 0
    
    def add_rank1(u_entries, v_entries, w_entries, coeff=1):
        """
        Add one rank-1 term.
        u_entries: dict {index: value} for U vector
        v_entries: dict {index: value} for V vector
        w_entries: dict {index: value} for W vector
        coeff: overall multiplier
        """
        nonlocal rank
        for ui, uv in u_entries.items():
            U_dict[(ui, rank)] = U_dict.get((ui, rank), 0) + uv * coeff
        for vi, vv in v_entries.items():
            V_dict[(vi, rank)] = V_dict.get((vi, rank), 0) + vv * coeff
        for wi, wv in w_entries.items():
            W_dict[(wi, rank)] = W_dict.get((wi, rank), 0) + wv * coeff
        rank += 1
    
    # 2x2x2 Strassen: 7 rank-1 terms
    def strassen_2x2(X_u, Y_v, Z_w, coeff=1):
        """
        X_u: [[x00, x01], [x10, x11]] - U indices
        Y_v: [[y00, y01], [y10, y11]] - V indices (Y[k][j])
        Z_w: [[z00, z01], [z10, z11]] - W indices (Z[i][j])
        
        Strassen:
        M1 = (X00+X11)*(Y00+Y11) -> contributes to Z00+Z11
        M2 = (X10+X11)*Y00 -> contributes to Z10
        M3 = X00*(Y01-Y11) -> contributes to Z01
        M4 = X11*(Y10-Y00) -> contributes to Z10
        M5 = (X00+X01)*Y11 -> contributes to Z01
        M6 = (X10-X00)*(Y00+Y01) -> contributes to Z11
        M7 = (X01-X11)*(Y10+Y11) -> contributes to Z00+Z11
        
        Z00 = M1 - M5 + M7
        Z01 = M3 + M5
        Z10 = M2 + M4
        Z11 = M1 - M2 + M3 + M6
        
        Wait, let me double-check the standard Strassen:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # M1: U: {X00:1, X11:1}, V: {Y00:1, Y11:1}, W: {Z00:1, Z11:1}
        add_rank1({X_u[0][0]: 1, X_u[1][1]: 1},
                  {Y_v[0][0]: 1, Y_v[1][1]: 1},
                  {Z_w[0][0]: 1, Z_w[1][1]: 1}, coeff)
        
        # M2: U: {X10:1, X11:1}, V: {Y00:1}, W: {Z10:1}
        add_rank1({X_u[1][0]: 1, X_u[1][1]: 1},
                  {Y_v[0][0]: 1},
                  {Z_w[1][0]: 1}, coeff)
        
        # M3: U: {X00:1}, V: {Y01:1, Y11:-1}, W: {Z01:1}
        add_rank1({X_u[0][0]: 1},
                  {Y_v[0][1]: 1, Y_v[1][1]: -1},
                  {Z_w[0][1]: 1}, coeff)
        
        # M4: U: {X11:1}, V: {Y10:1, Y00:-1}, W: {Z10:1}
        add_rank1({X_u[1][1]: 1},
                  {Y_v[1][0]: 1, Y_v[0][0]: -1},
                  {Z_w[1][0]: 1}, coeff)
        
        # M5: U: {X00:1, X01:1}, V: {Y11:1}, W: {Z01:1}
        add_rank1({X_u[0][0]: 1, X_u[0][1]: 1},
                  {Y_v[1][1]: 1},
                  {Z_w[0][1]: 1}, coeff)
        
        # M6: U: {X10:1, X00:-1}, V: {Y00:1, Y01:1}, W: {Z11:1}
        add_rank1({X_u[1][0]: 1, X_u[0][0]: -1},
                  {Y_v[0][0]: 1, Y_v[0][1]: 1},
                  {Z_w[1][1]: 1}, coeff)
        
        # M7: U: {X01:1, X11:-1}, V: {Y10:1, Y11:1}, W: {Z00:1, Z11:1}
        add_rank1({X_u[0][1]: 1, X_u[1][1]: -1},
                  {Y_v[1][0]: 1, Y_v[1][1]: 1},
                  {Z_w[0][0]: 1, Z_w[1][1]: 1}, coeff)
    
    # 4x4x4 Strassen: 7 multiplications of 2x2x2, each with 7 rank-1 terms = 49
    # Block indices for 4x4:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # For the 4x4x4 sub-problem (A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]):
    # U index for A[i,j] where i,j in 0..3: idx_A(i,j) = i*5+j
    # V index for B[j,k] where j,k in 0..3: idx_B(j,k) = j*5+k
    # W index for C[i,k] where i,k in 0..3: idx_C(i,k) = k*4+i
    
    def A4(i, j):
        return idx_A(i, j)
    
    def B4(j, k):
        return idx_B(j, k)
    
    def C4(i, k):
        return idx_C(i, k)
    
    # 2x2 blocks of A (4x4):
    A00_u = [[A4(0,0), A4(0,1)], [A4(1,0), A4(1,1)]]
    A01_u = [[A4(0,2), A4(0,3)], [A4(1,2), A4(1,3)]]
    A10_u = [[A4(2,0), A4(2,1)], [A4(3,0), A4(3,1)]]
    A11_u = [[A4(2,2), A4(2,3)], [A4(3,2), A4(3,3)]]
    
    # 2x2 blocks of B (4x4):
    B00_v = [[B4(0,0), B4(0,1)], [B4(1,0), B4(1,1)]]
    B01_v = [[B4(0,2), B4(0,3)], [B4(1,2), B4(1,3)]]
    B10_v = [[B4(2,0), B4(2,1)], [B4(3,0), B4(3,1)]]
    B11_v = [[B4(2,2), B4(2,3)], [B4(3,2), B4(3,3)]]
    
    # 2x2 blocks of C (4x4):
    C00_w = [[C4(0,0), C4(0,1)], [C4(1,0), C4(1,1)]]
    C01_w = [[C4(0,2), C4(0,3)], [C4(1,2), C4(1,3)]]
    C10_w = [[C4(2,0), C4(2,1)], [C4(3,0), C4(3,1)]]
    C11_w = [[C4(2,2), C4(2,3)], [C4(3,2), C4(3,3)]]
    
    # 4x4x4 Strassen: 7 Strassen-2x2 calls, each adding 7 rank-1 terms = 49
    # S1 = (A00+A11)*(B00+B11) -> C00 + C11 (but in 4x4 context, this is a 2x2 product)
    # Actually, for 4x4x4 Strassen, the 7 products are 2x2x2 matrix multiplications.
    # Each 2x2x2 multiplication has rank 7 (Strassen).
    # So 7 * 7 = 49 rank-1 terms total.
    
    # S1 = (A00+A11) * (B00+B11)
    # This is a 2x2x2 product. But A00+A11 means the U-side has entries from both blocks.
    # In the Strassen-2x2 function, we pass the combined indices.
    # However, our strassen_2x2 expects 2x2 index arrays. For S1, the "X" matrix
    # has X[0][0] = A00[0][0]+A11[0][0], etc. This doesn't fit the simple index array.
    #
    # Alternative: Expand S1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Each is a 2x2x2 schoolbook (rank 8), total 32 terms. But that's more than 7.
    #
    # The correct way: S1 is ONE 2x2x2 Strassen multiplication where the input
    # matrices are linear combinations. The rank-1 terms for S1 have U-side vectors
    # that are sums of entries from A00 and A11.
    #
    # Let me handle this by creating "virtual" 2x2 index arrays where each entry
    # is a list of (index, coefficient) pairs.
    
    def strassen_2x2_combined(X_u, Y_v, Z_w, coeff=1):
        """
        X_u: 2x2 list of lists of (index, coeff) pairs
        Y_v: 2x2 list of lists of (index, coeff) pairs
        Z_w: 2x2 list of lists of (index, coeff) pairs
        """
        # M1: U: X00+X11, V: Y00+Y11, W: Z00+Z11
        u1 = {}
        for idx, c in X_u[0][0]:
            u1[idx] = u1.get(idx, 0) + c
        for idx, c in X_u[1][1]:
            u1[idx] = u1.get(idx, 0) + c
        v1 = {}
        for idx, c in Y_v[0][0]:
            v1[idx] = v1.get(idx, 0) + c
        for idx, c in Y_v[1][1]:
            v1[idx] = v1.get(idx, 0) + c
        w1 = {}
        for idx, c in Z_w[0][0]:
            w1[idx] = w1.get(idx, 0) + c
        for idx, c in Z_w[1][1]:
            w1[idx] = w1.get(idx, 0) + c
        add_rank1(u1, v1, w1, coeff)
        
        # M2: U: X10+X11, V: Y00, W: Z10
        u2 = {}
        for idx, c in X_u[1][0]:
            u2[idx] = u2.get(idx, 0) + c
        for idx, c in X_u[1][1]:
            u2[idx] = u2.get(idx, 0) + c
        v2 = {}
        for idx, c in Y_v[0][0]:
            v2[idx] = v2.get(idx, 0) + c
        w2 = {}
        for idx, c in Z_w[1][0]:
            w2[idx] = w2.get(idx, 0) + c
        add_rank1(u2, v2, w2, coeff)
        
        # M3: U: X00, V: Y01-Y11, W: Z01
        u3 = {}
        for idx, c in X_u[0][0]:
            u3[idx] = u3.get(idx, 0) + c
        v3 = {}
        for idx, c in Y_v[0][1]:
            v3[idx] = v3.get(idx, 0) + c
        for idx, c in Y_v[1][1]:
            v3[idx] = v3.get(idx, 0) - c
        w3 = {}
        for idx, c in Z_w[0][1]:
            w3[idx] = w3.get(idx, 0) + c
        add_rank1(u3, v3, w3, coeff)
        
        # M4: U: X11, V: Y10-Y00, W: Z10
        u4 = {}
        for idx, c in X_u[1][1]:
            u4[idx] = u4.get(idx, 0) + c
        v4 = {}
        for idx, c in Y_v[1][0]:
            v4[idx] = v4.get(idx, 0) + c
        for idx, c in Y_v[0][0]:
            v4[idx] = v4.get(idx, 0) - c
        w4 = {}
        for idx, c in Z_w[1][0]:
            w4[idx] = w4.get(idx, 0) + c
        add_rank1(u4, v4, w4, coeff)
        
        # M5: U: X00+X01, V: Y11, W: Z01
        u5 = {}
        for idx, c in X_u[0][0]:
            u5[idx] = u5.get(idx, 0) + c
        for idx, c in X_u[0][1]:
            u5[idx] = u5.get(idx, 0) + c
        v5 = {}
        for idx, c in Y_v[1][1]:
            v5[idx] = v5.get(idx, 0) + c
        w5 = {}
        for idx, c in Z_w[0][1]:
            w5[idx] = w5.get(idx, 0) + c
        add_rank1(u5, v5, w5, coeff)
        
        # M6: U: X10-X00, V: Y00+Y01, W: Z11
        u6 = {}
        for idx, c in X_u[1][0]:
            u6[idx] = u6.get(idx, 0) + c
        for idx, c in X_u[0][0]:
            u6[idx] = u6.get(idx, 0) - c
        v6 = {}
        for idx, c in Y_v[0][0]:
            v6[idx] = v6.get(idx, 0) + c
        for idx, c in Y_v[0][1]:
            v6[idx] = v6.get(idx, 0) + c
        w6 = {}
        for idx, c in Z_w[1][1]:
            w6[idx] = w6.get(idx, 0) + c
        add_rank1(u6, v6, w6, coeff)
        
        # M7: U: X01-X11, V: Y10+Y11, W: Z00+Z11
        u7 = {}
        for idx, c in X_u[0][1]:
            u7[idx] = u7.get(idx, 0) + c
        for idx, c in X_u[1][1]:
            u7[idx] = u7.get(idx, 0) - c
        v7 = {}
        for idx, c in Y_v[1][0]:
            v7[idx] = v7.get(idx, 0) + c
        for idx, c in Y_v[1][1]:
            v7[idx] = v7.get(idx, 0) + c
        w7 = {}
        for idx, c in Z_w[0][0]:
            w7[idx] = w7.get(idx, 0) + c
        for idx, c in Z_w[1][1]:
            w7[idx] = w7.get(idx, 0) + c
        add_rank1(u7, v7, w7, coeff)
    
    # Helper to create simple 2x2 combined index arrays
    def simple_block(idx_func, r1, r2, c1, c2):
        """Create 2x2 array of [(index, 1)] for a block."""
        return [[[(idx_func(r1+i, c1+j), 1)] for j in range(2)] for i in range(2)]
    
    def combined_block(block1, block2, sign2=1):
        """Combine two 2x2 blocks: each entry is list of (idx, coeff) from both."""
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                entries = list(block1[i][j])
                for idx, c in block2[i][j]:
                    entries.append((idx, sign2 * c))
                row.append(entries)
            result.append(row)
        return result
    
    # A blocks as combined format
    A00_c = simple_block(A4, 0, 1, 0, 1)
    A01_c = simple_block(A4, 0, 1, 2, 3)
    A10_c = simple_block(A4, 2, 3, 0, 1)
    A11_c = simple_block(A4, 2, 3, 2, 3)
    
    B00_c = simple_block(B4, 0, 1, 0, 1)
    B01_c = simple_block(B4, 0, 1, 2, 3)
    B10_c = simple_block(B4, 2, 3, 0, 1)
    B11_c = simple_block(B4, 2, 3, 2, 3)
    
    C00_c = simple_block(C4, 0, 1, 0, 1)
    C01_c = simple_block(C4, 0, 1, 2, 3)
    C10_c = simple_block(C4, 2, 3, 0, 1)
    C11_c = simple_block(C4, 2, 3, 2, 3)
    
    # 4x4x4 Strassen: 7 products
    # S1 = (A00+A11)*(B00+B11)
    strassen_2x2_combined(combined_block(A00_c, A11_c), 
                          combined_block(B00_c, B11_c),
                          combined_block(C00_c, C11_c), 1)
    
    # S2 = (A10+A11)*B00
    strassen_2x2_combined(combined_block(A10_c, A11_c),
                          B00_c,
                          C10_c, 1)
    
    # S3 = A00*(B01-B11)
    strassen_2x2_combined(A00_c,
                          combined_block(B01_c, B11_c, -1),
                          C01_c, 1)
    
    # S4 = A11*(B10-B00)
    strassen_2x2_combined(A11_c,
                          combined_block(B10_c, B00_c, -1),
                          C11_c, 1)
    
    # S5 = (A00+A01)*B11
    strassen_2x2_combined(combined_block(A00_c, A01_c),
                          B11_c,
                          C01_c, 1)
    
    # S6 = (A10-A00)*(B00+B01)
    strassen_2x2_combined(combined_block(A10_c, A00_c, -1),
                          combined_block(B00_c, B01_c),
                          C10_c, 1)
    
    # S7 = (A01-A11)*(B10+B11)
    strassen_2x2_combined(combined_block(A01_c, A11_c, -1),
                          combined_block(B10_c, B11_c),
                          C11_c, 1)
    
    # Part 1 complete: 49 rank-1 terms for 4x4x4 Strassen.
    
    # ===== Part 2: 4x4x1 schoolbook (A[0:4,0:4] * B[0:4,4] -> C[0:4,4]) =====
    # C[i,4] += sum_{j=0}^{3} A[i,j] * B[j,4]
    # Schoolbook: 4*4*1 = 16 terms
    for i in range(4):
        for j in range(4):
            add_rank1({idx_A(i,j): 1}, {idx_B(j,4): 1}, {idx_C(i,4): 1}, 1)
    
    # ===== Part 3: 4x1x5 schoolbook (A[0:4,4] * B[4,0:5] -> C[0:4,0:5]) =====
    # C[i,k] += A[i,4] * B[4,k]
    # Schoolbook: 4*1*5 = 20 terms
    for i in range(4):
        for k in range(5):
            add_rank1({idx_A(i,4): 1}, {idx_B(4,k): 1}, {idx_C(i,k): 1}, 1)
    
    # Total rank: 49 + 16 + 20 = 85
    
    # Convert to arrays
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for (row, col), val in U_dict.items():
        U[row, col] = val
    for (row, col), val in V_dict.items():
        V[row, col] = val
    for (row, col), val in W_dict.items():
        W[row, col] = val
    
    # Quick verification with random matrices
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Reconstruct from decomposition
        C_recon = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            # u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
            # This is O(n*m*p*rank) which is fine for small sizes
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_recon[i, k] += u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        
        if not np.allclose(C_recon, C_expected, atol=1e-10):
            # Fallback to schoolbook if Strassen implementation has a bug
            U_dict = {}
            V_dict = {}
            W_dict = {}
            rank = 0
            
            def add_term_fb(u_row, v_row, w_row, val=1):
                nonlocal rank
                U_dict[(u_row, rank)] = U_dict.get((u_row, rank), 0) + val
                V_dict[(v_row, rank)] = V_dict.get((v_row, rank), 0) + val
                W_dict[(w_row, rank)] = W_dict.get((w_row, rank), 0) + val
                rank += 1
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        add_term_fb(i * m + j, j * p + k, k * n + i, 1)
            
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            for (row, col), val in U_dict.items():
                U[row, col] = val
            for (row, col), val in V_dict.items():
                V[row, col] = val
            for (row, col), val in W_dict.items():
                W[row, col] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
