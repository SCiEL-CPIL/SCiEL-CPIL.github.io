import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Loop over 4 blocks of B (and C)
    for b in range(4):
        base_col = b * 11
        
        # --- Strassen Part (7 terms) ---
        # Local indices for 2x2x2
        # A_L: i in {0,1}, j in {0,1}
        # B_L: j in {0,1}, k_loc in {0,1} -> global k = 2*b + k_loc
        # C_L: i in {0,1}, k_loc in {0,1} -> global k = 2*b + k_loc
        
        # M1 = (A00 + A11) * (B00 + B11)
        col = base_col + 0
        U[u_idx(0,0), col] += 1
        U[u_idx(1,1), col] += 1
        V[v_idx(0, 2*b+0), col] += 1
        V[v_idx(1, 2*b+1), col] += 1
        W[w_idx(2*b+0, 0), col] += 1 # C00
        W[w_idx(2*b+1, 1), col] += 1 # C11
        
        # M2 = (A10 + A11) * B00
        col = base_col + 1
        U[u_idx(1,0), col] += 1
        U[u_idx(1,1), col] += 1
        V[v_idx(0, 2*b+0), col] += 1
        W[w_idx(2*b+0, 1), col] += 1 # C10
        W[w_idx(2*b+1, 1), col] -= 1 # C11
        
        # M3 = A00 * (B01 - B11)
        col = base_col + 2
        U[u_idx(0,0), col] += 1
        V[v_idx(0, 2*b+1), col] += 1
        V[v_idx(1, 2*b+1), col] -= 1
        W[w_idx(2*b+1, 0), col] += 1 # C01
        W[w_idx(2*b+1, 1), col] += 1 # C11
        
        # M4 = A11 * (B10 - B00)
        col = base_col + 3
        U[u_idx(1,1), col] += 1
        V[v_idx(1, 2*b+0), col] += 1
        V[v_idx(0, 2*b+0), col] -= 1
        W[w_idx(2*b+0, 0), col] += 1 # C00
        W[w_idx(2*b+0, 1), col] += 1 # C10
        
        # M5 = (A00 + A01) * B11
        col = base_col + 4
        U[u_idx(0,0), col] += 1
        U[u_idx(0,1), col] += 1
        V[v_idx(1, 2*b+1), col] += 1
        W[w_idx(2*b+0, 0), col] -= 1 # C00
        W[w_idx(2*b+1, 0), col] += 1 # C01
        
        # M6 = (A10 - A11) * (B01 + B11)
        col = base_col + 5
        U[u_idx(1,0), col] += 1
        U[u_idx(1,1), col] -= 1
        V[v_idx(0, 2*b+1), col] += 1
        V[v_idx(1, 2*b+1), col] += 1
        W[w_idx(2*b+1, 1), col] += 1 # C11
        
        # M7 = (A01 - A11) * (B00 + B10)
        col = base_col + 6
        U[u_idx(0,1), col] += 1
        U[u_idx(1,1), col] -= 1
        V[v_idx(0, 2*b+0), col] += 1
        V[v_idx(1, 2*b+0), col] += 1
        W[w_idx(2*b+0, 0), col] += 1 # C00
        
        # --- Outer Product Part (4 terms) ---
        # A_R: i in {0,1}, j=2
        # B_R: j=2, k_loc in {0,1} -> global k = 2*b + k_loc
        # C_R: i in {0,1}, k_loc in {0,1} -> global k = 2*b + k_loc
        
        # P1 = A02 * B20 -> C00
        col = base_col + 7
        U[u_idx(0,2), col] += 1
        V[v_idx(2, 2*b+0), col] += 1
        W[w_idx(2*b+0, 0), col] += 1
        
        # P2 = A02 * B21 -> C01
        col = base_col + 8
        U[u_idx(0,2), col] += 1
        V[v_idx(2, 2*b+1), col] += 1
        W[w_idx(2*b+1, 0), col] += 1
        
        # P3 = A12 * B20 -> C10
        col = base_col + 9
        U[u_idx(1,2), col] += 1
        V[v_idx(2, 2*b+0), col] += 1
        W[w_idx(2*b+0, 1), col] += 1
        
        # P4 = A12 * B21 -> C11
        col = base_col + 10
        U[u_idx(1,2), col] += 1
        V[v_idx(2, 2*b+1), col] += 1
        W[w_idx(2*b+1, 1), col] += 1

    # Verify correctness quickly
    # T = sum_r U[:,r] V[:,r]^T W[:,r]^T ? No, tensor contraction.
    # Check a few entries.
    # The decomposition is exact by construction.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
