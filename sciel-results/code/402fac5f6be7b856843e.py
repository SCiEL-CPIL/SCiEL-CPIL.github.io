"""
Optimized 6x6 Matrix Multiplication Tensor Decomposition.

Strategy:
- Block decomposition: 6x6 = 2x2 blocks of 3x3 matrices.
- Outer: Strassen (rank 7) for 2x2 block multiplication.
- Inner: Schoolbook (rank 27) for 3x3 block multiplication.
- Total rank = 7 * 27 = 189.

This is a valid improvement over schoolbook rank 216.
Score = 216/189 ≈ 1.1429.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # Safety margin before 150s wall-clock kill

def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    dim = n  # 6
    
    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook():
        rank = n * m * p  # 216
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Hybrid: 2x2 Strassen (Outer) x 3x3 Schoolbook (Inner) (Rank 189)
    def _hybrid_strassen_schoolbook():
        block_size = 3
        rank_inner = 27   # Schoolbook rank for 3x3
        rank_outer = 7    # Strassen rank for 2x2
        total_rank = rank_inner * rank_outer  # 189
        
        U = [[0] * total_rank for _ in range(dim * dim)]
        V = [[0] * total_rank for _ in range(dim * dim)]
        W = [[0] * total_rank for _ in range(dim * dim)]
        
        col = 0
        
        # Strassen steps for 2x2 outer
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        steps = [
            # M1
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
            # M2
            {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
            # M3
            {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
            # M4
            {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
            # M5
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
            # M6
            {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
            # M7
            {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}}
        ]
        
        for step in steps:
            A_coeffs = step['A']
            B_coeffs = step['B']
            C_coeffs = step['C']
            
            for u in range(block_size):
                for v in range(block_size):
                    for w in range(block_size):
                        # U: X[u,v] = sum over A blocks
                        for (I, J), ca in A_coeffs.items():
                            i_global = I * block_size + u
                            j_global = J * block_size + v
                            idx_u = i_global * dim + j_global
                            U[idx_u][col] += ca
                        
                        # V: Y[v,w] = sum over B blocks
                        for (J, K), cb in B_coeffs.items():
                            j_global = J * block_size + v
                            k_global = K * block_size + w
                            idx_v = j_global * dim + k_global
                            V[idx_v][col] += cb
                        
                        # W: M[u,w] contributes to C blocks
                        for (I, K), cc in C_coeffs.items():
                            i_global = I * block_size + u
                            k_global = K * block_size + w
                            # W[k*n+i, r] convention
                            idx_w = k_global * dim + i_global
                            W[idx_w][col] += cc
                        
                        col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try Hybrid Strassen-Schoolbook (189)
    try:
        if time.time() < _DEADLINE:
            res = _hybrid_strassen_schoolbook()
            
            # Verify using numpy for speed
            U_np = np.array(res["U"], dtype=float)
            V_np = np.array(res["V"], dtype=float)
            W_np = np.array(res["W"], dtype=float)
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    u_row = U_np[i * m + j, :]
                    for k in range(p):
                        v_row = V_np[j * p + k, :]
                        w_row = W_np[k * n + i, :]
                        s = np.dot(u_row, v_row * w_row)
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-10:
                return res
    except Exception:
        pass
    
    # Fallback to Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
