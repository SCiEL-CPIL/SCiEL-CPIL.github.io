import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def get_189_solution():
        # Kronecker product of Strassen (2x2, Rank 7) and Naive (3x3, Rank 27)
        # This is safe and known to work.
        # Rank = 7 * 27 = 189
        
        # Strassen 2x2
        U2 = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        V2 = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        W2 = [
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        # Naive 3x3
        # Rank 27
        U3 = []
        V3 = []
        W3 = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U[i, j, r]
                    # V[j, k, r]
                    # W[k, i, r]
                    # r is the multiplication index
                    # Naive: C[i,k] += A[i,j] * B[j,k]
                    # So U[i,j] = 1, V[j,k] = 1, W[k,i] = 1
                    # Index r = i*9 + j*3 + k ? No, r is the column index.
                    # We just need to generate the vectors.
                    pass
                    
        # Simpler construction for 189:
        # Use the block approach.
        # 6x6 as 2x2 blocks of 3x3.
        # Strassen on 2x2 blocks.
        # Each "multiplication" is a 3x3 naive multiplication.
        
        U = [[0] * 189 for _ in range(36)]
        V = [[0] * 189 for _ in range(36)]
        W = [[0] * 189 for _ in range(36)]
        
        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        col_idx = 0
        for I in range(2):
            for J in range(2):
                for K in range(2):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        # This loop structure in the previous code was wrong for 6x6.
                        # The 6x6 is 2x2 blocks of 3x3.
                        # So we iterate over the 2x2 block indices I, J, K.
                        # And for each Strassen pattern, we apply the 3x3 naive decomposition.
                        
                        # Actually, the previous 189 code was:
                        # blocks_per_dim = 3, block_size = 2.
                        # That was 3x3 blocks of 2x2.
                        # Rank = 3^3 * 7 = 189.
                        # This is also valid.
                        
                        pass
        
        # I will reuse the logic from the previous 189 solution which was 3x3 blocks of 2x2 Strassen.
        # That is safer.
        
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c][col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c][col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i][col_idx] += val
                        col_idx += 1
        return {"U": U, "V": V, "W": W}

    def get_161_solution():
        # Kronecker product of 3x3 (Rank 23) and 2x2 (Rank 7)
        # Requires Bini et al. coefficients.
        # If not available, return None.
        return None

    sol_161 = get_161_solution()
    if sol_161 is not None:
        U, V, W = sol_161['U'], sol_161['V'], sol_161['W']
        rank = len(U[0])
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                C_calc[i, k] = s
                
        if np.allclose(C_calc, C_true):
            return sol_161
        else:
            pass

    return get_189_solution()
# EVOLVE-BLOCK-END
