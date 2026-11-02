"""Bootstrap spawn: direct structural construction with rank reduction via linear dependencies."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    
    Uses Strassen's algorithm on 2x2 blocks to reduce rank from schoolbook 96 to 88.
    For the 3x4x8 problem:
    - Split j-dimension (size 4) into two groups of 2
    - Apply Strassen to each 2x2x2 block (replaces 8 mults with 7)
    - 3 * 4 = 12 blocks of size (1x2x2) -> 12 * 7 = 84
    - Plus boundary terms for the odd dimension i (size 3 = 2+1)
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank_schoolbook = n * m * p  # 96
    
    # Schoolbook decomposition as fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Strassen-based decomposition for 3x4x8
    # Split j into j1 (j=0,1) and j2 (j=2,3)
    # Split k into k1 (k=0,1,2,3) and k2 (k=4,5,6,7) -- wait, p=8
    # Actually, let's think more carefully.
    
    # For A (3x4) * B (4x8) = C (3x8)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Split j = j_hi * 2 + j_lo where j_lo in {0,1}, j_hi in {0,1}
    # So j=0,1 are low group, j=2,3 are high group
    
    # For each i in {0,1,2} and k in {0,...,7}:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Group by j_hi:
    # C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    # For the first group (j=0,1): this is a 1x2 * 2x8 multiplication
    # We can apply Strassen-like tricks if we split k into pairs
    
    # Let's split k into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    # For each pair (k_lo, k_hi) and each i:
    # C[i, k_lo] = A[i,0]*B[0,k_lo] + A[i,1]*B[1,k_lo] + A[i,2]*B[2,k_lo] + A[i,3]*B[3,k_lo]
    # C[i, k_hi] = A[i,0]*B[0,k_hi] + A[i,1]*B[1,k_hi] + A[i,2]*B[2,k_hi] + A[i,3]*B[3,k_hi]
    
    # For fixed i and k-pair, we have 4 multiplications. Can we reduce to 3?
    # Not directly with Strassen since it's 1x2 * 2x2
    
    # Alternative: split i into pairs too
    # i = 0,1 as one group, i=2 as another
    # For i in {0,1}: 2x4 * 4x8
    # Split j into {0,1} and {2,3}
    # For each k-pair: 2x2 * 2x2 -> can use Strassen (7 mults instead of 8)
    
    # Let's do:
    # i_group1 = {0,1}, i_group2 = {2}
    # j_group1 = {0,1}, j_group2 = {2,3}
    # k_groups = {(0,1), (2,3), (4,5), (6,7)}
    
    # For i_group1 and each k_group: 2x2 * 2x2 -> Strassen gives 7 mults
    # 2 i-groups * 4 k-groups * 7 = 56 mults for i in {0,1}
    
    # For i=2 and each k_group: 1x2 * 2x2 -> 4 mults (no Strassen reduction)
    # 1 * 4 * 4 = 16 mults for i=2
    
    # Total: 56 + 16 = 72? That seems too good. Let me reconsider.
    
    # Actually for 1x2 * 2x2, we can't apply Strassen. We need 4 mults.
    # But wait, we could also group i=2 with i=0 or i=1...
    
    # Let me reconsider the structure. The key insight from the inspiration:
    # "replacing 8 scalar multiplications with 7 per 2x2 block"
    
    # For the full 3x4x8:
    # If we could decompose into 2x2x2 blocks, each needs 7 mults
    # 3x4x8 = ? 2x2x2 blocks
    # i: 3 = 1*2 + 1 (one 2-block + one 1)
    # j: 4 = 2*2 (two 2-blocks)
    # k: 8 = 4*2 (four 2-blocks)
    # So: (1*2 + 1) * 2 * 4 = 12 blocks of size 2x2x2 + 8 blocks of size 1x2x2
    # 12 * 7 + 8 * 4 = 84 + 32 = 116? That's worse than schoolbook!
    
    # I think I'm overcomplicating this. Let me look at the measured result:
    # "reduces the multiplication count from schoolbook 96 to 88"
    # So we save 8 multiplications.
    
    # One way: find 8 linear dependencies in the schoolbook decomposition.
    # The schoolbook has 96 rank-1 terms. If we can find 8 that are linearly
    # dependent, we can remove them.
    
    # Actually, let me try a different approach: use the fact that for matrix
    # multiplication, certain combinations of terms can be merged.
    
    # Let me just construct the Strassen decomposition properly.
    
    # For 2x2 * 2x2 = 2x2, Strassen uses 7 mults:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our problem, let's split:
    # i: 0,1 (pair) and 2 (single)
    # j: 0,1 (pair) and 2,3 (pair)  
    # k: 0,1 (pair), 2,3 (pair), 4,5 (pair), 6,7 (pair)
    
    # For i in {0,1}, j in {0,1}, k in each pair: 2x2x2 -> 7 mults
    # That's 1 * 1 * 4 = 4 blocks * 7 = 28 mults
    
    # For i in {0,1}, j in {2,3}, k in each pair: 2x2x2 -> 7 mults
    # That's 1 * 1 * 4 = 4 blocks * 7 = 28 mults
    
    # For i=2, j in {0,1}, k in each pair: 1x2x2 -> 4 mults
    # That's 1 * 1 * 4 = 4 blocks * 4 = 16 mults
    
    # For i=2, j in {2,3}, k in each pair: 1x2x2 -> 4 mults
    # That's 1 * 1 * 4 = 4 blocks * 4 = 16 mults
    
    # Total: 28 + 28 + 16 + 16 = 88 mults!
    
    # Great, so the rank is 88. Let me construct this explicitly.
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        """Add a rank-1 term with given indices and values."""
        if not U:
            U.append([0] * (n * m))
            V.append([0] * (m * p))
            W.append([0] * (p * n))
        
        U[-1][u_idx] += u_val
        V[-1][v_idx] += v_val
        W[-1][w_idx] += w_val
    
    # Helper to add a Strassen 2x2x2 block
    # A is 2x2, B is 2x2, C is 2x2
    # i_base: base index for i (0 or 2)
    # j_base: base index for j (0 or 2)
    # k_base: base index for k (0, 2, 4, or 6)
    
    def add_strassen_block(i_base, j_base, k_base):
        """Add Strassen decomposition for 2x2 * 2x2 = 2x2 block."""
        # A indices: (i_base, i_base+1) x (j_base, j_base+1)
        # B indices: (j_base, j_base+1) x (k_base, k_base+1)
        # C indices: (i_base, i_base+1) x (k_base, k_base+1)
        
        # M1 = (A00+A11)(B00+B11)
        add_term(i_base * m + j_base, j_base * p + k_base, k_base * n + i_base, 1, 1, 1)
        add_term((i_base + 1) * m + (j_base + 1), (j_base + 1) * p + (k_base + 1), (k_base + 1) * n + (i_base + 1), 1, 1, 1)
        
        # M2 = (A10+A11)B00
        add_term((i_base + 1) * m + j_base, j_base * p + k_base, k_base * n + (i_base + 1), 1, 1, 1)
        add_term((i_base + 1) * m + (j_base + 1), j_base * p + k_base, k_base * n + (i_base + 1), 1, 1, 1)
        
        # M3 = A00(B10-B11)
        add_term(i_base * m + j_base, (j_base + 1) * p + k_base, (k_base + 1) * n + i_base, 1, 1, -1)
        add_term(i_base * m + j_base, (j_base + 1) * p + (k_base + 1), (k_base + 1) * n + i_base, 1, -1, 1)
        
        # M4 = A11(B01-B00)
        add_term((i_base + 1) * m + (j_base + 1), j_base * p + k_base, k_base * n + (i_base + 1), 1, -1, 1)
        add_term((i_base + 1) * m + (j_base + 1), j_base * p + (k_base + 1), k_base * n + (i_base + 1), 1, 1, -1)
        
        # M5 = (A00+A01)B11
        add_term(i_base * m + j_base, (j_base + 1) * p + (k_base + 1), (k_base + 1) * n + i_base, 1, 1, 1)
        add_term(i_base * m + (j_base + 1), (j_base + 1) * p + (k_base + 1), (k_base + 1) * n + i_base, 1, 1, 1)
        
        # M6 = (A10-A00)(B00+B10)
        add_term((i_base + 1) * m + j_base, j_base * p + k_base, k_base * n + (i_base + 1), 1, 1, 1)
        add_term(i_base * m + j_base, j_base * p + k_base, k_base * n + i_base, -1, 1, 1)
        add_term((i_base + 1) * m + j_base, (j_base + 1) * p + k_base, (k_base + 1) * n + (i_base + 1), 1, 1, 1)
        add_term(i_base * m + j_base, (j_base + 1) * p + k_base, (k_base + 1) * n + i_base, -1, 1, 1)
        
        # M7 = (A01-A11)(B01+B11)
        add_term(i_base * m + (j_base + 1), j_base * p + (k_base + 1), (k_base + 1) * n + i_base, 1, 1, 1)
        add_term((i_base + 1) * m + (j_base + 1), j_base * p + (k_base + 1), (k_base + 1) * n + (i_base + 1), -1, 1, 1)
        add_term(i_base * m + (j_base + 1), (j_base + 1) * p + (k_base + 1), (k_base + 1) * n + i_base, 1, 1, 1)
        add_term((i_base + 1) * m + (j_base + 1), (j_base + 1) * p + (k_base + 1), (k_base + 1) * n + (i_base + 1), -1, 1, 1)
    
    # This approach is getting complicated. Let me use a cleaner formulation.
    
    # Actually, let me just directly construct the 88-rank decomposition.
    # The key is that for each 2x2x2 block, we use 7 rank-1 terms instead of 8.
    
    # Let me restart with a cleaner approach.
    
    U_new = []
    V_new = []
    W_new = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        """Add a rank-1 term."""
        U_new.append(u_vec)
        V_new.append(v_vec)
        W_new.append(w_vec)
    
    def make_u(i, j):
        """Create U vector for position (i,j)."""
        vec = [0] * (n * m)
        vec[i * m + j] = 1
        return vec
    
    def make_v(j, k):
        """Create V vector for position (j,k)."""
        vec = [0] * (m * p)
        vec[j * p + k] = 1
        return vec
    
    def make_w(k, i):
        """Create W vector for position (k,i)."""
        vec = [0] * (p * n)
        vec[k * n + i] = 1
        return vec
    
    def add_scaled_rank1(u, v, w, scale_u, scale_v, scale_w):
        """Add a scaled rank-1 term."""
        U_new.append([x * scale_u for x in u])
        V_new.append([x * scale_v for x in v])
        W_new.append([x * scale_w for x in w])
    
    # For Strassen on 2x2x2 block at (i_base, j_base, k_base):
    # We need 7 rank-1 terms
    
    def add_strassen(i_base, j_base, k_base):
        """Add 7 rank-1 terms for Strassen 2x2x2 block."""
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        u1 = [0] * (n * m)
        u1[i_base * m + j_base] = 1
        u1[(i_base + 1) * m + (j_base + 1)] = 1
        v1 = [0] * (m * p)
        v1[j_base * p + k_base] = 1
        v1[(j_base + 1) * p + (k_base + 1)] = 1
        w1 = [0] * (p * n)
        w1[k_base * n + i_base] = 1
        w1[(k_base + 1) * n + (i_base + 1)] = 1
        add_rank1(u1, v1, w1)
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        u2 = [0] * (n * m)
        u2[(i_base + 1) * m + j_base] = 1
        u2[(i_base + 1) * m + (j_base + 1)] = 1
        v2 = [0] * (m * p)
        v2[j_base * p + k_base] = 1
        w2 = [0] * (p * n)
        w2[k_base * n + (i_base + 1)] = 1
        add_rank1(u2, v2, w2)
        
        # M3 = A[i0,j0] * (B[j1,k0] - B[j1,k1])
        u3 = [0] * (n * m)
        u3[i_base * m + j_base] = 1
        v3 = [0] * (m * p)
        v3[(j_base + 1) * p + k_base] = 1
        v3[(j_base + 1) * p + (k_base + 1)] = -1
        w3 = [0] * (p * n)
        w3[k_base * n + i_base] = 1
        add_rank1(u3, v3, w3)
        
        # M4 = A[i1,j1] * (B[j0,k1] - B[j0,k0])
        u4 = [0] * (n * m)
        u4[(i_base + 1) * m + (j_base + 1)] = 1
        v4 = [0] * (m * p)
        v4[j_base * p + (k_base + 1)] = 1
        v4[j_base * p + k_base] = -1
        w4 = [0] * (p * n)
        w4[k_base * n + (i_base + 1)] = 1
        add_rank1(u4, v4, w4)
        
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        u5 = [0] * (n * m)
        u5[i_base * m + j_base] = 1
        u5[i_base * m + (j_base + 1)] = 1
        v5 = [0] * (m * p)
        v5[(j_base + 1) * p + (k_base + 1)] = 1
        w5 = [0] * (p * n)
        w5[(k_base + 1) * n + i_base] = 1
        add_rank1(u5, v5, w5)
        
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j1,k0])
        u6 = [0] * (n * m)
        u6[(i_base + 1) * m + j_base] = 1
        u6[i_base * m + j_base] = -1
        v6 = [0] * (m * p)
        v6[j_base * p + k_base] = 1
        v6[(j_base + 1) * p + k_base] = 1
        w6 = [0] * (p * n)
        w6[k_base * n + (i_base + 1)] = 1
        add_rank1(u6, v6, w6)
        
        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j0,k1] + B[j1,k1])
        u7 = [0] * (n * m)
        u7[i_base * m + (j_base + 1)] = 1
        u7[(i_base + 1) * m + (j_base + 1)] = -1
        v7 = [0] * (m * p)
        v7[j_base * p + (k_base + 1)] = 1
        v7[(j_base + 1) * p + (k_base + 1)] = 1
        w7 = [0] * (p * n)
        w7[(k_base + 1) * n + i_base] = 1
        add_rank1(u7, v7, w7)
    
    # For 1x2x2 block at (i, j_base, k_base): need 4 rank-1 terms
    def add_1x2x2(i, j_base, k_base):
        """Add 4 rank-1 terms for 1x2x2 block."""
        for j_offset in range(2):
            for k_offset in range(2):
                u = [0] * (n * m)
                u[i * m + j_base + j_offset] = 1
                v = [0] * (m * p)
                v[(j_base + j_offset) * p + k_base + k_offset] = 1
                w = [0] * (p * n)
                w[(k_base + k_offset) * n + i] = 1
                add_rank1(u, v, w)
    
    # Now apply:
    # i_base = 0 (covers i=0,1), i_base = 2 (covers i=2, but we only use i=2)
    # j_base = 0 (covers j=0,1), j_base = 2 (covers j=2,3)
    # k_base = 0, 2, 4, 6
    
    for j_base in [0, 2]:
        for k_base in [0, 2, 4, 6]:
            # 2x2x2 block for i=0,1
            add_strassen(0, j_base, k_base)
            # 1x2x2 block for i=2
            add_1x2x2(2, j_base, k_base)
    
    # Check if we have exactly 88 terms
    if len(U_new) == 88:
        # Verify the decomposition is correct
        # For each (i,k), check that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all j
        # Actually, the tensor T[i,j,k] should equal sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        
        # Let's verify with random matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B_test
        
        U_np = np.array(U_new).T  # shape (rank, n*m)
        V_np = np.array(V_new).T  # shape (rank, m*p)
        W_np = np.array(W_new).T  # shape (rank, p*n)
        
        # Compute C using the decomposition
        C_decomp = np.zeros((n, p))
        for r in range(len(U_new)):
            u = U_np[r].reshape(n, m)
            v = V_np[r].reshape(m, p)
            w = W_np[r].reshape(p, n)
            C_decomp += np.einsum('ij,jk,ki->ik', u, v, w)
        
        if np.allclose(C_decomp, C_test, atol=1e-10):
            best_result = {"U": U_new, "V": V_new, "W": W_new}
            best_rank = 88
        else:
            # Debug: print mismatch
            print("Decomposition verification failed!")
            print("Max diff:", np.max(np.abs(C_decomp - C_test)))
            # Return schoolbook
            pass
    else:
        print(f"Expected 88 terms, got {len(U_new)}")
    
    return best_result
# EVOLVE-BLOCK-END

