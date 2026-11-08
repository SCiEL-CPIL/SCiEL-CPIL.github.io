"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Use Strassen's 2x2x2 decomposition (rank 7) applied to 8 blocks
    # A is 2x4, B is 4x8, C is 2x8
    # Split B into 4 blocks of 2x2 along the column dimension
    # For each (i_block, j_block) pair, we have a 2x2x2 subproblem
    # Total: 2 i-blocks * 2 j-blocks * 2 k-blocks = 8 subproblems
    # Each uses 7 rank-1 terms, total rank = 8 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen 2x2x2 decomposition patterns
    # For local 2x2 matrices: A[i,j], B[j,k], C[i,k] with i,j,k in {0,1}
    # Tensor index: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # The 7 Strassen terms (U_entries, V_entries, W_entries)
    # Each entry is (local_index, value)
    strassen_terms = [
        # Term 0: (A00+A11)(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # Term 1: (A10+A11)B00
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # Term 2: A00(B10-B11)
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # Term 3: A11(B01-B00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # Term 4: (A00+A01)B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # Term 5: (A10-A00)(B00+B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # Term 6: (A01-A11)(B10+B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Add a Strassen block at given offsets
    def add_strassen_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            current_rank = base_rank + r
            u_entries, v_entries, w_entries = strassen_terms[r]
            
            for idx, val in u_entries:
                local_i, local_j = divmod(idx, 2)
                gi = local_i + i_off
                gj = local_j + j_off
                U[gi * 4 + gj][current_rank] += val
            
            for idx, val in v_entries:
                local_j, local_k = divmod(idx, 2)
                gj = local_j + j_off
                gk = local_k + k_off
                V[gj * 8 + gk][current_rank] += val
            
            for idx, val in w_entries:
                local_k, local_i = divmod(idx, 2)
                gk = local_k + k_off
                gi = local_i + i_off
                W[gk * 2 + gi][current_rank] += val
    
    # 8 blocks: i_off in {0}, j_off in {0, 2}, k_off in {0, 2, 4, 6}
    # i is always 0-1 (n=2), so only one i-block
    block_count = 0
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            base_rank = block_count * 7
            add_strassen_block(base_rank, 0, j_off, k_off)
            block_count += 1
    
    # Verify the decomposition
    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify():
        # Fallback to schoolbook if something went wrong
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
