import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20
    
    # Part 1: Recursive Strassen 4x4x4 (rank 49)
    # Part 2: Schoolbook 4x4x1 (rank 16)
    # Total rank: 65
    
    rank_part1 = 49
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # --- Part 2: Schoolbook for column 4 (k=4) ---
    # A (4x4) * b (4x1)
    # V index: j * p + k. Here k=4. So V[j*5 + 4, col]
    # W index: k * n + i. Here k=4. So W[4*4 + i, col]
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0  # k=4
            W[4 * n + i, col] = 1.0  # k=4, i=row of C
    
    # --- Part 1: Recursive Strassen 4x4x4 ---
    # We decompose 4x4x4 into 7 products of 2x2x2 matrices.
    # Each 2x2x2 is decomposed into 7 scalar products (Strassen).
    # Total 49 products.
    
    # Helper functions for index mapping
    def a_idx(bi, u, v):
        """Global index in A for block bi, local (u,v)"""
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j
    
    def b_idx(bi, u, v):
        """Global index in B for block bi, local (u,v) - for 4x4 part (cols 0-3)"""
        j = (bi // 2) * 2 + u  # row of B
        k = (bi % 2) * 2 + v  # col of B (0-3)
        return j * p + k
    
    def c_idx(bi, u, v):
        """Global index in C for block bi, local (u,v)"""
        i = (bi // 2) * 2 + u  # row of C
        k = (bi % 2) * 2 + v  # col of C (0-3)
        return k * n + i
    
    # Strassen 2x2x2 definitions (for the inner level)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 7 outer Strassen products (Mk for k=0..6):
    # Each Mk = A'_k * B'_k where A'_k and B'_k are 2x2 matrices
    # A'_k = sum of coeff * A_block
    # B'_k = sum of coeff * B_block
    
    # Then each 2x2x2 multiplication uses inner Strassen (7 scalar products)
    
    # Outer Strassen definitions
    A_side = [
        [(0, 1.0), (3, 1.0)],   # M1: A00+A11
        [(2, 1.0), (3, 1.0)],   # M2: A10+A11
        [(0, 1.0)],             # M3: A00
        [(3, 1.0)],             # M4: A11
        [(0, 1.0), (1, 1.0)],   # M5: A00+A01
        [(2, 1.0), (0, -1.0)],  # M6: A10-A00
        [(1, 1.0), (3, -1.0)],  # M7: A01-A11
    ]
    
    B_side = [
        [(0, 1.0), (3, 1.0)],   # M1: B00+B11
        [(0, 1.0)],             # M2: B00
        [(1, 1.0), (3, -1.0)],  # M3: B01-B11
        [(2, 1.0), (0, -1.0)],  # M4: B10-B00
        [(3, 1.0)],             # M5: B11
        [(0, 1.0), (1, 1.0)],   # M6: B00+B01
        [(2, 1.0), (3, 1.0)],   # M7: B10+B11
    ]
    
    out_coeff = [
        {0: 1.0, 3: 1.0},       # M1 -> C00, C11
        {2: 1.0, 3: -1.0},      # M2 -> C10, -C11
        {1: 1.0, 3: 1.0},       # M3 -> C01, C11
        {0: 1.0, 2: 1.0},       # M4 -> C00, C10
        {0: -1.0, 1: 1.0},      # M5 -> -C00, C01
        {3: 1.0},               # M6 -> C11
        {0: 1.0},               # M7 -> C00
    ]
    
    # Inner Strassen 2x2x2 definitions
    # For a 2x2x2 multiplication: A (2x2) * B (2x2) = C (2x2)
    # Inner Strassen products:
    # N1 = (A00+A11)(B00+B11)
    # N2 = (A10+A11)B00
    # N3 = A00(B01-B11)
    # N4 = A11(B10-B00)
    # N5 = (A00+A01)B11
    # N6 = (A10-A00)(B00+B01)
    # N7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = N1 + N4 - N5 + N7
    # C01 = N3 + N5
    # C10 = N2 + N4
    # C11 = N1 - N2 + N3 + N6
    
    inner_A_side = [
        [(0, 1.0), (3, 1.0)],   # N1: A00+A11
        [(2, 1.0), (3, 1.0)],   # N2: A10+A11
        [(0, 1.0)],             # N3: A00
        [(3, 1.0)],             # N4: A11
        [(0, 1.0), (1, 1.0)],   # N5: A00+A01
        [(2, 1.0), (0, -1.0)],  # N6: A10-A00
        [(1, 1.0), (3, -1.0)],  # N7: A01-A11
    ]
    
    inner_B_side = [
        [(0, 1.0), (3, 1.0)],   # N1: B00+B11
        [(0, 1.0)],             # N2: B00
        [(1, 1.0), (3, -1.0)],  # N3: B01-B11
        [(2, 1.0), (0, -1.0)],  # N4: B10-B00
        [(3, 1.0)],             # N5: B11
        [(0, 1.0), (1, 1.0)],   # N6: B00+B01
        [(2, 1.0), (3, 1.0)],   # N7: B10+B11
    ]
    
    inner_out_coeff = [
        {0: 1.0, 3: 1.0},       # N1 -> C00, C11
        {2: 1.0, 3: -1.0},      # N2 -> C10, -C11
        {1: 1.0, 3: 1.0},       # N3 -> C01, C11
        {0: 1.0, 2: 1.0},       # N4 -> C00, C10
        {0: -1.0, 1: 1.0},      # N5 -> -C00, C01
        {3: 1.0},               # N6 -> C11
        {0: 1.0},               # N7 -> C00
    ]
    
    # For each outer Strassen product (mk in 0..6):
    # The 2x2 matrices A'_mk and B'_mk are formed by linear combinations
    # Then we compute A'_mk * B'_mk using inner Strassen (7 scalar products)
    
    for mk in range(7):
        # For each inner Strassen product (nk in 0..6):
        for nk in range(7):
            col = mk * 7 + nk
            
            # U: A'_mk at local position (u,v) = sum of coeff * A_block
            # A'_mk = sum_{(bi, coeff) in A_side[mk]} coeff * A_bi
            # Then inner Strassen: A''_nk = sum_{(bi2, coeff2) in inner_A_side[nk]} coeff2 * A'_mk[bi2]
            # So A''_nk[u,v] = sum_{(bi, c1) in A_side[mk]} sum_{(bi2, c2) in inner_A_side[nk]} 
            #   c1 * c2 * A_bi[local position corresponding to bi2]
            
            # Wait, I need to think about this more carefully.
            # A'_mk is a 2x2 matrix where A'_mk[i,j] = sum_{(bi, coeff) in A_side[mk]} coeff * A_bi[i,j]
            # Here i,j are local 2x2 indices.
            
            # Then inner Strassen computes A'_mk * B'_mk.
            # N_nk = A''_nk * B''_nk where A''_nk and B''_nk are linear combinations of 2x2 blocks of A'_mk and B'_mk.
            
            # But A'_mk is already a 2x2 matrix. The "blocks" of A'_mk in the inner Strassen are:
            # A'_mk[0,0], A'_mk[0,1], A'_mk[1,0], A'_mk[1,1]
            # These are scalars (elements of the 2x2 matrix).
            
            # So inner Strassen N_nk = A''_nk * B''_nk where A''_nk and B''_nk are scalars.
            # A''_nk = sum of coeff * A'_mk[bi2] for bi2 in {0,1,2,3}
            # where bi2=0 -> A'_mk[0,0], bi2=1 -> A'_mk[0,1], bi2=2 -> A'_mk[1,0], bi2=3 -> A'_mk[1,1]
            
            # A'_mk[bi2] = sum_{(bi, c1) in A_side[mk]} c1 * A_bi[local_pos(bi2)]
            # where local_pos(bi2) = (bi2//2, bi2%2)
            
            # So A''_nk = sum_{(bi, c1) in A_side[mk]} sum_{(bi2, c2) in inner_A_side[nk]} c1 * c2 * A_bi[local_pos(bi2)]
            
            # U index: global A index for block bi, local position local_pos(bi2)
            # = a_idx(bi, bi2//2, bi2%2)
            
            for (bi, c1) in A_side[mk]:
                for (bi2, c2) in inner_A_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    U[a_idx(bi, u_local, v_local), col] += c1 * c2
            
            # Similarly for V
            for (bi, c1) in B_side[mk]:
                for (bi2, c2) in inner_B_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    V[b_idx(bi, u_local, v_local), col] += c1 * c2
            
            # W: Output coefficients
            # The result of N_nk contributes to C blocks.
            # C'_mk is a 2x2 matrix.
            # C'_mk[bi2] = sum_{(bi3, c3) in out_coeff[mk]} c3 * (result of inner Strassen for block bi3 of A'_mk * B'_mk)
            
            # Wait, the structure is:
            # C'_mk = f(A'_mk, B'_mk) using Strassen.
            # The output of inner Strassen N_nk is a scalar.
            # This scalar contributes to the elements of C'_mk.
            # Specifically, C'_mk[bi3] = sum_{nk} (coeff from inner_out_coeff[nk][bi3]) * N_nk
            
            # So for each block bi3 in the output of the outer Strassen (which is a 2x2 matrix C'_mk):
            # C'_mk[bi3] = sum_{nk} inner_out_coeff[nk].get(bi3, 0) * N_nk
            
            # Then the final C is formed by:
            # C[bi] = sum_{mk} out_coeff[mk].get(bi, 0) * C'_mk
            
            # Combining:
            # C[bi] = sum_{mk} out_coeff[mk].get(bi, 0) * (sum_{nk} inner_out_coeff[nk].get(bi3, 0) * N_nk)
            # where bi3 is the local index in C'_mk corresponding to the global block bi.
            
            # Actually, it's simpler:
            # N_nk is a scalar product.
            # It contributes to C'_mk[bi3] with weight inner_out_coeff[nk].get(bi3, 0).
            # C'_mk[bi3] contributes to C[global_block] with weight out_coeff[mk].get(global_block, 0).
            
            # So for each global block bi (0..3):
            # We need to find if bi corresponds to a local block bi3 in C'_mk.
            # C'_mk is a 2x2 matrix. Its blocks are C'_mk[0], C'_mk[1], C'_mk[2], C'_mk[3].
            # These correspond to global blocks? No.
            # A'_mk is a 2x2 matrix of scalars? No, A'_mk is a 2x2 matrix of 2x2 blocks?
            # No, in the recursive step:
            # Level 1: 4x4x4. A is 4x4. Split into 2x2 blocks of 2x2 matrices.
            # A'_mk is a 2x2 matrix where each element is a 2x2 matrix?
            # No, the standard Strassen recursion:
            # A (4x4) = [A00 A01; A10 A11] where Aij are 2x2.
            # A'_mk = sum c * Aij. So A'_mk is a 2x2 matrix.
            # Then we multiply A'_mk (2x2) by B'_mk (2x2) to get C'_mk (2x2).
            # This multiplication is done via inner Strassen.
            # The result C'_mk is a 2x2 matrix.
            # The elements of C'_mk are scalars? No, they are 2x2 matrices?
            # No, A'_mk elements are 2x2 matrices.
            # So A'_mk is a 2x2 matrix of 2x2 matrices.
            # Inner Strassen multiplies two 2x2 matrices of 2x2 matrices?
            # No, that would be 4x4x4 again.
            
            # Let's re-read the inspiration code logic.
            # It treats the 2x2 matrix A'_mk as having 4 scalar elements?
            # No, it maps indices.
            # a_idx(bi, u, v) maps block bi (0..3) and local (u,v) to global A index.
            # So A is viewed as 16 scalars.
            # A'_mk is a linear combination of these 16 scalars.
            # So A'_mk is a 2x2 matrix of scalars?
            # No, A'_mk is a 2x2 matrix where each entry is a scalar?
            # If A'_mk is 2x2 of scalars, then A'_mk * B'_mk is 2x2 of scalars.
            # But we are multiplying 4x4 matrices.
            
            # Ah, the recursion is:
            # 4x4x4 -> 7 * (2x2x2)
            # Each 2x2x2 is multiplied using Strassen (7 scalar products).
            # So total 49 scalar products.
            
            # So A'_mk is a 2x2 matrix of scalars?
            # No, A is 4x4. A'_mk = sum c * A_block.
            # A_block is 2x2. So A'_mk is 2x2.
            # Wait, if A'_mk is 2x2, and B'_mk is 2x2, their product is 2x2.
            # But the result should be 4x4.
            
            # The trick is that the "blocks" in the first level Strassen are 2x2 matrices.
            # So A'_mk is a 2x2 matrix of 2x2 matrices.
            # Then we multiply A'_mk (2x2 of 2x2) by B'_mk (2x2 of 2x2).
            # This is a 2x2x2 multiplication of 2x2 matrices.
            # This can be done with 7 multiplications of 2x2 matrices.
            # Each 2x2 matrix multiplication can be done with 7 scalar multiplications (Strassen).
            # So 7 * 7 = 49 scalar multiplications.
            
            # So in the code:
            # U, V, W are for scalar products.
            # The indices in U, V, W correspond to the scalar entries of A, B, C.
            
            # So for the inner Strassen (2x2x2 of 2x2 matrices):
            # We treat the 2x2 matrix A'_mk as having 4 entries: A'_mk[0,0], A'_mk[0,1], A'_mk[1,0], A'_mk[1,1].
            # Each of these entries is a 2x2 matrix.
            # So A'_mk[0,0] is a 2x2 matrix, which is a linear combination of the 4 scalars in A_block 0.
            
            # So the code in the inspiration is correct.
            # It iterates over the scalar components.
            
            # Let's trace the W index.
            # N_nk is a scalar product.
            # It contributes to C'_mk[bi3] where bi3 is 0..3 (local 2x2 block index).
            # C'_mk[bi3] is a 2x2 matrix.
            # So N_nk contributes to the 4 scalars in C'_mk[bi3].
            # The weights are the same for all 4 scalars?
            # No, Strassen for 2x2x2 (scalar) gives coefficients for C00, C01, C10, C11.
            # Here the "scalars" are 2x2 matrices.
            # So the coefficients apply to the whole 2x2 block.
            
            # So for each local block bi3 (0..3) in C'_mk:
            # The weight is inner_out_coeff[nk].get(bi3, 0).
            # This weight applies to the global block corresponding to C'_mk[bi3].
            
            # But C'_mk is a 2x2 matrix of 2x2 blocks.
            # So C'_mk[bi3] is one of the 4 blocks of the 4x4 result.
            # Wait, C'_mk is the result of multiplying A'_mk (2x2 of 2x2) by B'_mk (2x2 of 2x2).
            # So C'_mk is a 2x2 matrix of 2x2 matrices.
            # So C'_mk has 4 blocks, each 2x2.
            # These 4 blocks are the 4 blocks of the final 4x4 matrix C.
            
            # So bi3 in inner_out_coeff[nk] corresponds to the global block index in C.
            
            # So:
            # For each bi3 in 0..3:
            #   weight_inner = inner_out_coeff[nk].get(bi3, 0)
            #   If weight_inner != 0:
            #      For each local (u,v) in the 2x2 block bi3:
            #         c_global = c_idx(bi3, u, v)
            #         W[c_global, col] += weight_inner * (weight_outer?)
            
            # No, the outer Strassen combines the 7 results C'_mk.
            # C[bi] = sum_{mk} out_coeff[mk].get(bi, 0) * C'_mk[bi]
            
            # So:
            # C[bi] = sum_{mk} out_coeff[mk].get(bi, 0) * (sum_{nk} inner_out_coeff[nk].get(bi, 0) * N_nk)
            
            # So for a fixed mk, nk, and global block bi:
            # weight = out_coeff[mk].get(bi, 0) * inner_out_coeff[nk].get(bi, 0)
            
            # If weight != 0:
            #   For each local (u,v) in block bi:
            #      c_global = c_idx(bi, u, v)
            #      W[c_global, col] += weight
            
            for bi in range(4):
                w_outer = out_coeff[mk].get(bi, 0.0)
                w_inner = inner_out_coeff[nk].get(bi, 0.0)
                w_total = w_outer * w_inner
                if w_total != 0:
                    for u in range(2):
                        for v in range(2):
                            W[c_idx(bi, u, v), col] += w_total
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_exp = A @ B
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    ap = 0.0
                    for a in range(n):
                        for b in range(m):
                            ap += U[a*m+b, r] * A[a, b]
                    bp = 0.0
                    for b in range(m):
                        for c in range(p):
                            bp += V[b*p+c, r] * B[b, c]
                    val += ap * bp * W[k*n+i, r]
                C_comp[i, k] = val
        if not np.allclose(C_comp, C_exp, atol=1e-4):
            return False
    return True

if __name__ == "__main__":
    res = construct()
    if _verify(res['U'], res['V'], res['W']):
        print("Verification passed. Rank:", len(res['U'][0]))
    else:
        print("Verification failed.")

