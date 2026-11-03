# EVOLVE-BLOCK-START
import time

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = 4, 4, 6
    
    # Rank: 49 (Strassen 4x4) + 8 (Schoolbook 4x4x2) = 57
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen Steps for 4x4 * 4x4 (cols 0-3)
    # A is 4x4, B is 4x4 (cols 0-3), C is 4x4 (cols 0-3)
    # Partition A into 2x2 blocks of size 2x2
    # Partition B into 2x2 blocks of size 2x2
    # Partition C into 2x2 blocks of size 2x2
    
    outer_steps = [
        # M1 = (A00 + A11) * (B00 + B11) -> C00 + C11
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        # M2 = (A10 + A11) * B00 -> C10 - C11
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        # M3 = A00 * (B01 - B11) -> C01 + C11
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        # M4 = A11 * (B10 - B00) -> C00 + C10
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        # M5 = (A00 + A01) * B11 -> C01 - C00
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        # M6 = (A10 - A00) * (B00 + B01) -> C11
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        # M7 = (A01 - A11) * (B10 + B11) -> C00
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
    ]
    
    # Inner Strassen for 2x2 * 2x2 (7 terms)
    # A_inner: 2x2 (r, s)
    # B_inner: 2x2 (s, k)
    # C_inner: 2x2 (r, k)
    inner_7 = [
        # S1: (A00+A11)(B00+B11) -> C00+C11
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        # S2: (A10+A11)B00 -> C10-C11
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        # S3: A00(B01-B11) -> C01+C11
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        # S4: A11(B10-B00) -> C00+C10
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        # S5: (A00+A01)B11 -> C01-C00
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
        # S6: (A10-A00)(B00+B01) -> C11
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        # S7: (A01-A11)(B10+B11) -> C00
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    ]
    
    # Helper to map local block coordinates to global indices
    def get_global_A(block_id, r, s):
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J

    def get_global_B(block_id, s, k):
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 2 + k
        return J, K

    def get_global_C(block_id, r, k):
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 2 + k
        return I, K

    # Construct the decomposition for cols 0-3 (Rank 49)
    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base_rank = i * 7
        
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_7):
            c_global_rank = base_rank + j
            
            # Update U
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            # Update V
            for (s, k, coeff) in B_inner:
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            # Update W: W index is K * n + I
            for (r, k, coeff) in C_inner:
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

    # Construct the decomposition for cols 4-5 (Rank 8)
    # A is 4x4, B is 4x2 (cols 4,5), C is 4x2 (cols 4,5)
    # Schoolbook: C[r, k] = sum_s A[r, s] B[s, k]
    # We use 4 terms for col 4 and 4 terms for col 5.
    # Term for col 4, s-th multiplier:
    # U[r, s] = 1, V[s, 4] = 1, W[4, r] = 1?
    # W index is K * n + I = 4 * 4 + r = 16 + r.
    
    for s in range(4):
        # Col 4
        c_rank_4 = 49 + s
        for r in range(4):
            U[r * m + s][c_rank_4] = 1
        V[s * p + 4][c_rank_4] = 1
        for r in range(4):
            W[4 * n + r][c_rank_4] = 1
            
        # Col 5
        c_rank_5 = 53 + s
        for r in range(4):
            U[r * m + s][c_rank_5] = 1
        V[s * p + 5][c_rank_5] = 1
        for r in range(4):
            W[5 * n + r][c_rank_5] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
