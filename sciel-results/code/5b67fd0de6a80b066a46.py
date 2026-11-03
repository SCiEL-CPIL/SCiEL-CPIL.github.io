import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_decomp(N):
    """
    Returns U, V, W for N x N x N matrix multiplication using Strassen's algorithm.
    N must be a power of 2.
    U, V, W are lists of lists (or numpy arrays) of shape (N*N, Rank).
    """
    if N == 1:
        return [[1.0]], [[1.0]], [[1.0]]
    
    half = N // 2
    # Recursive call for half size
    U_rec, V_rec, W_rec = _strassen_decomp(half)
    rank_rec = len(U_rec[0])
    
    # Indices for N x N matrix
    # A[i, k], B[k, j], C[i, j]
    # We need to map the sub-problems to the global indices.
    
    # There are 7 Strassen terms.
    # Each term involves a multiplication of two half-sized matrices.
    # Let's define the 7 terms (u_coeffs, v_coeffs)
    # u_coeffs: linear combo of A blocks
    # v_coeffs: linear combo of B blocks
    
    # Blocks:
    # A: A00, A01, A10, A11 (each half x half)
    # B: B00, B01, B10, B11
    # C: C00, C01, C10, C11
    
    # Term 1: (A00 + A11) * (B00 + B11) -> contributes to C00, C11
    # Term 2: (A10 + A11) * B00 -> contributes to C10, C11
    # Term 3: A00 * (B01 - B11) -> contributes to C00, C01
    # Term 4: A11 * (B10 - B00) -> contributes to C10, C11
    # Term 5: (A00 + A01) * B11 -> contributes to C01, C11
    # Term 6: (A10 - A00) * (B00 + B01) -> contributes to C10, C11
    # Term 7: (A01 - A11) * (B10 + B11) -> contributes to C00, C01, C10, C11?
    # Actually, the output combinations are:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will construct U, V, W by iterating over the 7 terms.
    # For each term, we have a rank_rec decomposition of the half-size multiplication.
    # We need to "embed" this decomposition into the N-size space.
    
    U_full = []
    V_full = []
    W_full = []
    
    # Helper to get global index for block (bi, bj) in N x N matrix
    # Block (bi, bj) covers rows bi*half to (bi+1)*half-1 and cols bj*half to (bj+1)*half-1
    def get_block_indices(bi, bj):
        rows = range(bi * half, (bi + 1) * half)
        cols = range(bj * half, (bj + 1) * half)
        return [(r * N + c) for r in rows for c in cols]
    
    # Define the 7 terms: (A_blocks, B_blocks, C_contributions)
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_contributions: list of (block_idx, coeff) for the result of this term
    
    terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # A00+A11, B00+B11 -> C00+1, C11+1? No, M1 is added to C00 and C11.
        # Wait, M1 is a matrix. C00 gets M1, C11 gets M1.
        # So for each element of M1, it adds to the corresponding element in C00 and C11.
        # So C_contributions should reflect that the output block (0,0) gets coeff 1 and (1,1) gets coeff 1.
        
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)]), # A10+A11, B00 -> C10+1, C11+1?
        # C10 = M2 + M4. C11 = M1 - M2 + ...
        # So M2 contributes +1 to C10 and -1 to C11.
        
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,0,1), (0,1,1)]), # A00, B01-B11 -> C00+1, C01+1?
        # C00 = ... + M3? No.
        # C00 = M1 + M4 - M5 + M7.
        # C01 = M3 + M5.
        # So M3 contributes +1 to C01. Does it contribute to C00? No.
        # Wait, standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-verify the contributions.
        # M1 -> C00 (+1), C11 (+1)
        # M2 -> C10 (+1), C11 (-1)
        # M3 -> C01 (+1), C11 (+1) ? No.
        # Let's check C11 = M1 - M2 + M3 + M6.
        # So M3 contributes +1 to C11.
        # And C01 = M3 + M5. So M3 contributes +1 to C01.
        
        # M4 -> C00 (+1), C10 (+1)
        # M5 -> C00 (-1), C01 (+1)
        # M6 -> C11 (+1) ? No.
        # C11 = M1 - M2 + M3 + M6.
        # So M6 contributes +1 to C11.
        # Does M6 contribute elsewhere? No.
        
        # M7 -> C00 (+1) ?
        # C00 = M1 + M4 - M5 + M7.
        # So M7 contributes +1 to C00.
        # Does M7 contribute elsewhere? No.
    ]
    
    # Corrected Contributions:
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    # Let's re-verify M6 and M7.
    # M6 = (A10 - A00)(B00 + B01).
    # M7 = (A01 - A11)(B10 + B11).
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # This looks consistent.
    
    # We will build the full U, V, W.
    # Total rank will be 7 * rank_rec.
    total_rank = 7 * rank_rec
    
    U = [[0.0] * total_rank for _ in range(N*N)]
    V = [[0.0] * total_rank for _ in range(N*N)]
    W = [[0.0] * total_rank for _ in range(N*N)]
    
    # Define the 7 terms with their A, B, C block coefficients
    # (A_blocks, B_blocks, C_blocks)
    # Block index: (row_block, col_block)
    terms_def = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    for t_idx, (a_blocks, b_blocks, c_blocks) in enumerate(terms_def):
        # For each scalar multiplication in the recursive decomposition
        for r in range(rank_rec):
            col = t_idx * rank_rec + r
            
            # Process U (Input A)
            # U_rec[i, r] is the coefficient for A element i in the sub-problem
            # We need to map sub-problem element i to global A element.
            # The sub-problem A is formed by linear combination of blocks.
            # This is tricky: U_rec is for a specific A matrix.
            # Here, the "A" for this term is a linear combination of global blocks.
            # So, for each global block (bi, bj) with coeff c_a:
            #   For each local index i in the block:
            #       global_idx = block_idx(i)
            #       U[global_idx][col] += c_a * U_rec[i][r]
            
            # Similarly for V and W.
            
            # Build U
            for (bi, bj, coeff_a) in a_blocks:
                global_indices = get_block_indices(bi, bj)
                for k, gi in enumerate(global_indices):
                    U[gi][col] += coeff_a * U_rec[k][r]
            
            # Build V
            for (bi, bj, coeff_b) in b_blocks:
                global_indices = get_block_indices(bi, bj)
                for k, gi in enumerate(global_indices):
                    V[gi][col] += coeff_b * V_rec[k][r]
            
            # Build W
            for (bi, bj, coeff_c) in c_blocks:
                global_indices = get_block_indices(bi, bj)
                for k, gi in enumerate(global_indices):
                    W[gi][col] += coeff_c * W_rec[k][r]
    
    return U, V, W

def _schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1.0
                V[k * p + j][col] = 1.0
                W[j * n + i][col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Get rank-49 decomposition for 4x4x4
    U4, V4, W4 = _strassen_decomp(4)
    rank_4 = len(U4[0]) # Should be 49
    
    # 2. Get schoolbook decomposition for 4x4x5
    # We need to compute A * B_part2 where B_part2 is 4x5
    # This corresponds to the last 5 columns of B in the 4x4x9 problem.
    # The first 4 columns are handled by the 4x4x4 part.
    U5, V5, W5 = _schoolbook_decomp(4, 4, 5)
    rank_5 = len(U5[0]) # Should be 80
    
    # 3. Combine
    # The 4x4x9 problem:
    # A is 4x4. B is 4x9. C is 4x9.
    # Split B into B1 (4x4) and B2 (4x5).
    # C1 = A B1 (4x4). C2 = A B2 (4x5).
    # C = [C1, C2].
    
    # U, V, W for 4x4x9
    total_rank = rank_4 + rank_5
    U = [[0.0] * total_rank for _ in range(n * m)] # 16 x 129
    V = [[0.0] * total_rank for _ in range(m * p)] # 36 x 129
    W = [[0.0] * total_rank for _ in range(p * n)] # 36 x 129
    
    # Copy 4x4x4 part
    # U4 is 16x49. U is 16x129.
    for i in range(16):
        for c in range(rank_4):
            U[i][c] = U4[i][c]
            
    # V4 is 16x49. V is 36x129.
    # V4 corresponds to B1 (4x4).
    # In the 4x4x9 problem, B1 occupies the first 4 columns of B.
    # B index: k * p + j.
    # B1 is k=0..3, j=0..3.
    # So V4[k * 4 + j] maps to V[k * 9 + j].
    for k in range(4):
        for j in range(4):
            v_idx_4 = k * 4 + j
            v_idx_9 = k * 9 + j
            for c in range(rank_4):
                V[v_idx_9][c] = V4[v_idx_4][c]
                
    # W4 is 16x49. W is 36x129.
    # W4 corresponds to C1 (4x4).
    # C1 occupies the first 4 columns of C.
    # C index: j * n + i.
    # C1 is j=0..3, i=0..3.
    # So W4[j * 4 + i] maps to W[j * 9 + i].
    for j in range(4):
        for i in range(4):
            w_idx_4 = j * 4 + i
            w_idx_9 = j * 9 + i
            for c in range(rank_4):
                W[w_idx_9][c] = W4[w_idx_4][c]
                
    # Copy 4x4x5 part
    # U5 is 16x80.
    for i in range(16):
        for c in range(rank_5):
            U[i][rank_4 + c] = U5[i][c]
            
    # V5 is 16x80.
    # V5 corresponds to B2 (4x5).
    # B2 occupies the last 5 columns of B (j=4..8).
    # B index in 4x4x5: k * 5 + j_local (j_local 0..4).
    # B index in 4x4x9: k * 9 + j_global (j_global 4..8).
    for k in range(4):
        for j_local in range(5):
            v_idx_5 = k * 5 + j_local
            v_idx_9 = k * 9 + (j_local + 4)
            for c in range(rank_5):
                V[v_idx_9][rank_4 + c] = V5[v_idx_5][c]
                
    # W5 is 20x80.
    # W5 corresponds to C2 (4x5).
    # C2 occupies the last 5 columns of C (j=4..8).
    # C index in 4x4x5: j_local * 4 + i.
    # C index in 4x4x9: j_global * 4 + i.
    for j_local in range(5):
        for i in range(4):
            w_idx_5 = j_local * 4 + i
            w_idx_9 = (j_local + 4) * 4 + i
            for c in range(rank_5):
                W[w_idx_9][rank_4 + c] = W5[w_idx_5][c]
                
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U_s, V_s, W_s = _schoolbook_decomp(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
