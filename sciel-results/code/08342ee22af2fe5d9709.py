import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 4, 8
    rank = 56

    U_2x2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=int)
    
    V_2x2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=int)
    
    W_2x2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)

    # Pre-compute indices
    # r_idx is 0..55
    r_indices = np.arange(rank)
    
    # ab, bg, s
    # r = ab*4*7 + bg*7 + s
    # ab = r // 28
    # rem = r % 28
    # bg = rem // 7
    # s = rem % 7
    
    abs_ = (r_indices // 28).astype(int)
    rems = r_indices % 28
    bgs = (rems // 7).astype(int)
    ss = (rems % 7).astype(int)
    
    # U indices
    # U[i*m + ab*2 + jp, r] = U_2x2[s, i*2 + jp]
    # i in 0,1; jp in 0,1
    # We need to assign for all i, jp.
    # Let's create arrays for the row indices of U for each r.
    # For a given r, ab is fixed.
    # Rows: i*m + ab*2 + jp for i in 0,1 and jp in 0,1
    # i=0, jp=0 -> ab*2
    # i=0, jp=1 -> ab*2 + 1
    # i=1, jp=0 -> m + ab*2
    # i=1, jp=1 -> m + ab*2 + 1
    
    # We can use broadcasting.
    # U_rows = (np.array([0, 0, 1, 1]) * m + abs_[:, None] * 2 + np.array([0, 1, 0, 1])[None, :]).flatten()
    # This gives 4 rows per r. Total 4*56 = 224 entries.
    # U_vals = U_2x2[ss, np.array([0,1,2,3])[None,:]].flatten()
    
    i_jp_pairs = np.array([[0,0], [0,1], [1,0], [1,1]])
    i_vals = i_jp_pairs[:, 0]
    jp_vals = i_jp_pairs[:, 1]
    
    # For each r, the 4 rows are:
    # i*m + ab*2 + jp
    U_row_indices = (i_vals[None, :] * m + abs_[:, None] * 2 + jp_vals[None, :]).flatten()
    U_val_indices = (ss[:, None] * 4 + i_vals[None, :] * 2 + jp_vals[None, :]).flatten()
    
    U[U_row_indices, np.repeat(np.arange(rank), 4)] = U_2x2.flatten()[U_val_indices]
    
    # V indices
    # V[(ab*2+jp)*p + 2*bg + kp, r] = V_2x2[s, jp*2 + kp]
    # jp in 0,1; kp in 0,1
    jp_kp_pairs = np.array([[0,0], [0,1], [1,0], [1,1]])
    jp_vals_v = jp_kp_pairs[:, 0]
    kp_vals_v = jp_kp_pairs[:, 1]
    
    V_row_indices = ((abs_[:, None] * 2 + jp_vals_v[None, :]) * p + bgs[:, None] * 2 + kp_vals_v[None, :]).flatten()
    V_val_indices = (ss[:, None] * 4 + jp_vals_v[None, :] * 2 + kp_vals_v[None, :]).flatten()
    
    V[V_row_indices, np.repeat(np.arange(rank), 4)] = V_2x2.flatten()[V_val_indices]
    
    # W indices
    # W[(2*bg+kp)*n + i, r] = W_2x2[i*2+kp, s]
    # i in 0,1; kp in 0,1
    i_kp_pairs = np.array([[0,0], [0,1], [1,0], [1,1]])
    i_vals_w = i_kp_pairs[:, 0]
    kp_vals_w = i_kp_pairs[:, 1]
    
    W_row_indices = ((bgs[:, None] * 2 + kp_vals_w[None, :]) * n + i_vals_w[None, :]).flatten()
    # W_2x2 is 4x7. Index is i*2+kp, s.
    W_val_indices = (i_vals_w[None, :] * 2 + kp_vals_w[None, :]) * 7 + ss[:, None]
    W_val_indices = W_val_indices.flatten()
    
    W[W_row_indices, np.repeat(np.arange(rank), 4)] = W_2x2.flatten()[W_val_indices]
    
    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
