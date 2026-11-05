"""
Optimized 6x6 Matrix Multiplication Tensor Decomposition.

Strategy:
1. Primary: Hybrid Strassen (2x2 outer) + Schoolbook (3x3 inner) decomposition.
   Rank = 7 * 27 = 189.
   This is significantly better than Schoolbook (216) and is a verified valid
   decomposition using the Kronecker product structure of block decomposition.

2. Fallback: Schoolbook 6x6.
   Rank = 216.

The implementation uses vectorized NumPy operations to construct the factor
matrices efficiently and verifies the result before returning. A strict
deadline guard ensures the program returns within the evaluator's time limit.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # Safety margin before 150s wall-clock kill

def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Hybrid: 2x2 Strassen (Outer) x 3x3 Schoolbook (Inner) (Rank 189)
    def _hybrid_strassen_schoolbook():
        block_size = 3
        dim = 6
        
        rank_inner = 27  # Schoolbook rank for 3x3
        rank_outer = 7   # Strassen rank for 2x2
        total_rank = rank_inner * rank_outer
        
        U = np.zeros((dim * dim, total_rank))
        V = np.zeros((dim * dim, total_rank))
        W = np.zeros((dim * dim, total_rank))
        
        # Strassen steps for 2x2 outer
        # Format: list of dicts with 'A', 'B', 'C' as {(i,j): coeff}
        strassen_steps = [
            {'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 }},
            {'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 }},
            {'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 }},
            {'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 }},
            {'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 }},
            {'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 }},
            {'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 }}
        ]
        
        # Schoolbook steps for 3x3
        # For each i, j, k in 0..2
        schoolbook_steps = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    schoolbook_steps.append({
                        'A': { (i,j): 1 },
                        'B': { (j,k): 1 },
                        'C': { (i,k): 1 }
                    })
        
        col = 0
        for s_step in strassen_steps:
            A_coeffs_outer = s_step['A']
            B_coeffs_outer = s_step['B']
            C_coeffs_outer = s_step['C']
            
            for sb_step in schoolbook_steps:
                A_coeffs_inner = sb_step['A']
                B_coeffs_inner = sb_step['B']
                C_coeffs_inner = sb_step['C']
                
                # Build U
                for (I, J), ca_out in A_coeffs_outer.items():
                    for (u, v), ca_in in A_coeffs_inner.items():
                        i_global = I * block_size + u
                        j_global = J * block_size + v
                        idx_u = i_global * dim + j_global
                        U[idx_u, col] += ca_out * ca_in
                
                # Build V
                for (J, K), cb_out in B_coeffs_outer.items():
                    for (v, w), cb_in in B_coeffs_inner.items():
                        j_global = J * block_size + v
                        k_global = K * block_size + w
                        idx_v = j_global * dim + k_global
                        V[idx_v, col] += cb_out * cb_in
                
                # Build W
                for (I, K), cc_out in C_coeffs_outer.items():
                    for (u, w), cc_in in C_coeffs_inner.items():
                        i_global = I * block_size + u
                        k_global = K * block_size + w
                        idx_w = k_global * dim + i_global
                        W[idx_w, col] += cc_out * cc_in
                
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try Hybrid Strassen-Schoolbook (189)
    try:
        if time.time() < _DEADLINE:
            res = _hybrid_strassen_schoolbook()
            # Verify
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
