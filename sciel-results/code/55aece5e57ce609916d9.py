"""
Optimized 6x6 Matrix Multiplication Tensor Decomposition.

Strategy:
1. Primary: Hybrid Strassen (2x2 outer) + Schoolbook (3x3 inner) decomposition.
   Rank = 7 * 27 = 189.
   This is a valid and verified improvement over the Schoolbook rank of 216.
   The previous attempt at a "Laderman" hybrid was flawed due to incorrect rank assumptions
   (3x3 rank is not 7). We stick to the robust Strassen + Schoolbook structure.

2. Fallback: Schoolbook 6x6.
   Rank = 216.

The key is to correctly construct the tensor factors U, V, W such that
T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
for all i, j, k, where n=m=p=6.
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
        num_blocks = 2
        dim = n
        
        rank_inner = 27  # Schoolbook rank for 3x3
        rank_outer = 7   # Strassen rank for 2x2
        total_rank = rank_inner * rank_outer
        
        U = np.zeros((dim * dim, total_rank))
        V = np.zeros((dim * dim, total_rank))
        W = np.zeros((dim * dim, total_rank))
        
        col = 0
        
        # Strassen steps for 2x2 outer
        # Standard Strassen algorithm for 2x2 matrices:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each step is defined by linear combinations of A blocks, B blocks, and C blocks.
        # Format: {'A': {(I,J): coeff}, 'B': {(J,K): coeff}, 'C': {(I,K): coeff}}
        # Where I, J, K are block indices (0 or 1).
        
        strassen_steps = [
            # M1
            {'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 }},
            # M2
            {'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 }},
            # M3
            {'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 }},
            # M4
            {'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 }},
            # M5
            {'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 }},
            # M6
            {'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 }},
            # M7
            {'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 }}
        ]
        
        # Schoolbook steps for 3x3 inner
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
        
        for s_step in strassen_steps:
            A_coeffs_outer = s_step['A']
            B_coeffs_outer = s_step['B']
            C_coeffs_outer = s_step['C']
            
            for sb_step in schoolbook_steps:
                A_coeffs_inner = sb_step['A']
                B_coeffs_inner = sb_step['B']
                C_coeffs_inner = sb_step['C']
                
                # Build U
                # U index: (I * block_size + u) * dim + (J * block_size + v)
                for (I, J), ca_out in A_coeffs_outer.items():
                    for (u, v), ca_in in A_coeffs_inner.items():
                        i_global = I * block_size + u
                        j_global = J * block_size + v
                        idx_u = i_global * dim + j_global
                        U[idx_u, col] += ca_out * ca_in
                
                # Build V
                # V index: (J * block_size + v) * dim + (K * block_size + w)
                for (J, K), cb_out in B_coeffs_outer.items():
                    for (v, w), cb_in in B_coeffs_inner.items():
                        j_global = J * block_size + v
                        k_global = K * block_size + w
                        idx_v = j_global * dim + k_global
                        V[idx_v, col] += cb_out * cb_in
                
                # Build W
                # W index: (K * block_size + w) * dim + (I * block_size + u)
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
            # We can vectorize the check for speed
            # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # We need T[i,j,k] == 1 for all i,j,k.
            
            # Let's do a partial check or full check if fast enough.
            # 6*6*6 = 216 entries. Rank 189.
            # 216 * 189 operations is trivial.
            
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
            else:
                # If verification fails, something is wrong with the logic.
                # But Strassen+Schoolbook is a known valid construction.
                # We will still return it if the error is small, or fall back.
                # Given the strictness, we fall back to schoolbook if it fails.
                pass
    except Exception:
        pass
        
    # Fallback to Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
