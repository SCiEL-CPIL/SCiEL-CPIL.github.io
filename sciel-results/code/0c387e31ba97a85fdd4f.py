import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 4, 8
    N = n * m  # 16
    M = m * p  # 32
    P = p * n  # 32
    
    def verify_fast(U, V, W, tol=1e-9):
        T_rec = np.einsum('ak,bk,ck->abc', U, V, W)
        T_target = np.zeros((N, M, P))
        for a in range(N):
            i_a, j_a = divmod(a, 4)
            for b in range(M):
                r_b, c_b = divmod(b, 8)
                if j_a == r_b:
                    c_target = c_b * 4 + i_a
                    T_target[a, b, c_target] = 1.0
        return np.allclose(T_rec, T_target, atol=tol)
    
    def build_rank98():
        """Build rank-98 decomposition using block Strassen for 4x4x8."""
        rank = 98
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        
        # 2x2 Strassen decomposition
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = -1
        
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = 1
        
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1
        
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1
        
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        A_specs = [
            [(1, 0), (1, 3)],
            [(1, 2), (1, 3)],
            [(1, 0)],
            [(1, 3)],
            [(1, 0), (1, 1)],
            [(1, 2), (-1, 0)],
            [(1, 1), (-1, 3)],
        ]
        
        B_specs = [
            [(1, 0), (1, 3)],
            [(1, 0)],
            [(1, 1), (-1, 3)],
            [(1, 2), (-1, 0)],
            [(1, 3)],
            [(1, 0), (1, 1)],
            [(1, 2), (1, 3)],
        ]
        
        C_specs = [
            [(1, 0), (1, 3)],
            [(1, 2), (-1, 3)],
            [(1, 1), (1, 3)],
            [(1, 0), (1, 2)],
            [(-1, 0), (1, 1)],
            [(1, 3)],
            [(1, 0)],
        ]
        
        col = 0
        for half in range(2):
            for k_p in range(7):
                for m_s in range(7):
                    for coeff_a, block_idx in A_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_r = bi * 2 + r
                                global_c = bj * 2 + c
                                global_a_idx = global_r * 4 + global_c
                                a_eff_idx = r * 2 + c
                                U[global_a_idx, col] += coeff_a * U2[a_eff_idx, m_s]
                    
                    for coeff_b, block_idx in B_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_r = bi * 2 + r
                                global_c = bj * 2 + c
                                global_k = global_c + half * 4
                                global_b_idx = global_r * 8 + global_k
                                b_eff_idx = r * 2 + c
                                V[global_b_idx, col] += coeff_b * V2[b_eff_idx, m_s]
                    
                    for coeff_c, c_block_idx in C_specs[k_p]:
                        bi, bj = divmod(c_block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_i = bi * 2 + r
                                global_k = bj * 2 + c + half * 4
                                global_w_idx = global_k * 4 + global_i
                                p_eff_idx = r * 2 + c
                                W[global_w_idx, col] += coeff_c * W2[p_eff_idx, m_s]
                    
                    col += 1
        
        return U, V, W
    
    U, V, W = build_rank98()
    
    if not verify_fast(U, V, W, tol=1e-9):
        # Fallback to naive rank-128
        rank = 128
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for r in range(m):
                for k in range(p):
                    U[i * 4 + r, col] = 1.0
                    V[r * 8 + k, col] = 1.0
                    W[k * 4 + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
