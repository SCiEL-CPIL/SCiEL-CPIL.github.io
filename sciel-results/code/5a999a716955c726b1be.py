import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to get global indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen block addition
    # a_i, a_j, b_j, b_k, c_i, c_k are the starting indices of the blocks
    # e.g., a_i=0, a_j=0 for A1
    def add_strassen(start_rank, a_i, a_j, b_j, b_k, c_i, c_k):
        # Local indices 0,1 map to a_i+0, a_i+1 etc.
        
        # M1
        U[u_idx(a_i+0, a_j+0), start_rank+0] += 1
        U[u_idx(a_i+1, a_j+1), start_rank+0] += 1
        V[v_idx(b_j+0, b_k+0), start_rank+0] += 1
        V[v_idx(b_j+1, b_k+1), start_rank+0] += 1
        W[w_idx(b_k+0, c_i+0), start_rank+0] += 1 # z_00
        W[w_idx(b_k+1, c_i+1), start_rank+0] += 1 # z_11
        
        # M2
        U[u_idx(a_i+1, a_j+0), start_rank+1] += 1
        U[u_idx(a_i+1, a_j+1), start_rank+1] += 1
        V[v_idx(b_j+0, b_k+0), start_rank+1] += 1
        W[w_idx(b_k+0, c_i+1), start_rank+1] += 1 # z_10
        W[w_idx(b_k+1, c_i+1), start_rank+1] -= 1 # z_11
        
        # M3
        U[u_idx(a_i+0, a_j+0), start_rank+2] += 1
        V[v_idx(b_j+0, b_k+1), start_rank+2] += 1
        V[v_idx(b_j+1, b_k+1), start_rank+2] -= 1
        W[w_idx(b_k+1, c_i+0), start_rank+2] += 1 # z_01
        W[w_idx(b_k+1, c_i+1), start_rank+2] += 1 # z_11
        
        # M4
        U[u_idx(a_i+1, a_j+1), start_rank+3] += 1
        V[v_idx(b_j+1, b_k+0), start_rank+3] += 1
        V[v_idx(b_j+0, b_k+0), start_rank+3] -= 1
        W[w_idx(b_k+0, c_i+0), start_rank+3] += 1 # z_00
        W[w_idx(b_k+0, c_i+1), start_rank+3] += 1 # z_10
        
        # M5
        U[u_idx(a_i+0, a_j+0), start_rank+4] += 1
        U[u_idx(a_i+0, a_j+1), start_rank+4] += 1
        V[v_idx(b_j+1, b_k+1), start_rank+4] += 1
        W[w_idx(b_k+0, c_i+0), start_rank+4] -= 1 # z_00
        W[w_idx(b_k+1, c_i+0), start_rank+4] += 1 # z_01
        
        # M6
        U[u_idx(a_i+1, a_j+0), start_rank+5] += 1
        U[u_idx(a_i+0, a_j+0), start_rank+5] -= 1
        V[v_idx(b_j+0, b_k+0), start_rank+5] += 1
        V[v_idx(b_j+0, b_k+1), start_rank+5] += 1
        W[w_idx(b_k+1, c_i+1), start_rank+5] += 1 # z_11
        
        # M7
        U[u_idx(a_i+0, a_j+1), start_rank+6] += 1
        U[u_idx(a_i+1, a_j+1), start_rank+6] -= 1
        V[v_idx(b_j+1, b_k+0), start_rank+6] += 1
        V[v_idx(b_j+1, b_k+1), start_rank+6] += 1
        W[w_idx(b_k+0, c_i+0), start_rank+6] += 1 # z_00
        W[w_idx(b_k+1, c_i+1), start_rank+6] += 1 # z_11

    # Block 1: A1 B1 -> C1
    # A1: i=0,1, j=0,1
    # B1: j=0,1, k=0,1
    # C1: k=0,1, i=0,1
    add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A1 B2 -> C2
    # A1: i=0,1, j=0,1
    # B2: j=0,1, k=2,3
    # C2: k=2,3, i=0,1
    add_strassen(7, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A2 B3 -> C1
    # A2: i=0,1, j=2,3
    # B3: j=2,3, k=0,1
    # C1: k=0,1, i=0,1
    add_strassen(14, 0, 2, 2, 0, 0, 0)
    
    # Block 4: A2 B4 -> C2
    # A2: i=0,1, j=2,3
    # B4: j=2,3, k=2,3
    # C2: k=2,3, i=0,1
    add_strassen(21, 0, 2, 2, 2, 0, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
