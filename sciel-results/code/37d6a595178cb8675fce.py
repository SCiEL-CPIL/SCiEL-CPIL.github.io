import numpy as np
import time

def construct():
    """Construct a rank-28 tensor decomposition for 2x3x5 matrix multiplication.
    
    Uses a data-driven approach: define blocks as lists of (u_indices, v_indices, w_indices, w_signs) tuples,
    then assemble the U, V, W matrices.
    
    Block decomposition:
    - Block A: j=0, all k (2x1x5 schoolbook) -> rank 10
    - Block B: j={1,2}, k={0,1} (2x2x2 Strassen) -> rank 7
    - Block C: j={1,2}, k=2 (2x2x1 schoolbook) -> rank 4
    - Block D: j={1,2}, k={3,4} (2x2x2 Strassen) -> rank 7
    Total: 28
    """
    n, m, p = 2, 3, 5
    
    # Define Strassen 7-term pattern for 2x2x2
    # Each term: (u_local, v_local, w_local, w_sign)
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # u = i*2+j, v = j*2+k, w = k*2+i
    strassen_pattern = [
        # (u_idx, v_idx, w_idx, w_sign)
        ([0, 3], [0, 3], [0, 3], [1, 1]),       # M1: (A00+A11)(B00+B11) -> C00 + C11
        ([2, 3], [0], [1, 3], [1, -1]),         # M2: (A10+A11)B00 -> C10 - C11
        ([0], [1, 3], [2, 3], [1, 1]),          # M3: A00(B01-B11) -> C01 + C11
        ([3], [2, 0], [0, 1], [1, 1]),          # M4: A11(B10-B00) -> C00 + C10
        ([0, 1], [3], [0, 2], [-1, 1]),         # M5: (A00+A01)B11 -> -C00 + C01
        ([2, 0], [0, 1], [3], [1, 1]),          # M6: (A10-A00)(B00+B01) -> C11
        ([1, 3], [2, 3], [0], [1, 1]),          # M7: (A01-A11)(B10+B11) -> C00
    ]
    
    # Define all blocks
    # Each block: list of (u_global_indices, v_global_indices, w_global_indices, w_signs)
    blocks = []
    
    # Block A: j=0, k=0..4 (schoolbook 2x1x5)
    for i in range(n):
        for k in range(p):
            blocks.append(([i*m + 0], [0*p + k], [k*n + i], [1]))
    
    # Block B: j={1,2}, k={0,1} (Strassen 2x2x2)
    # Local j' = j-1, so j=1->j'=0, j=2->j'=1
    # Local k' = k, so k=0->k'=0, k=1->k'=1
    for u_loc, v_loc, w_loc, w_sign in strassen_pattern:
        u_global = [u_loc[0]//2 * m + (u_loc[0]%2 + 1) for u_loc in [u_loc[0]]]  # i*2+j' -> i*m+(j'+1)
        # Actually let me compute this more carefully
        u_global = []
        for ul in u_loc:
            i_loc = ul // 2
            j_loc = ul % 2
            u_global.append(i_loc * m + (j_loc + 1))
        v_global = []
        for vl in v_loc:
            j_loc = vl // 2
            k_loc = vl % 2
            v_global.append((j_loc + 1) * p + k_loc)
        w_global = []
        for wl in w_loc:
            k_loc = wl // 2
            i_loc = wl % 2
            w_global.append(k_loc * n + i_loc)
        blocks.append((u_global, v_global, w_global, w_sign))
    
    # Block C: j={1,2}, k=2 (schoolbook 2x2x1)
    for i in range(n):
        for j in [1, 2]:
            blocks.append(([i*m + j], [j*p + 2], [2*n + i], [1]))
    
    # Block D: j={1,2}, k={3,4} (Strassen 2x2x2)
    # Local k' = k-3, so k=3->k'=0, k=4->k'=1
    for u_loc, v_loc, w_loc, w_sign in strassen_pattern:
        u_global = []
        for ul in u_loc:
            i_loc = ul // 2
            j_loc = ul % 2
            u_global.append(i_loc * m + (j_loc + 1))
        v_global = []
        for vl in v_loc:
            j_loc = vl // 2
            k_loc = vl % 2
            v_global.append((j_loc + 1) * p + (k_loc + 3))
        w_global = []
        for wl in w_loc:
            k_loc = wl // 2
            i_loc = wl % 2
            w_global.append((k_loc + 3) * n + i_loc)
        blocks.append((u_global, v_global, w_global, w_sign))
    
    # Assemble U, V, W
    rank = len(blocks)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col, (u_indices, v_indices, w_indices, w_signs) in enumerate(blocks):
        for idx in u_indices:
            U[idx, col] += 1
        for idx in v_indices:
            V[idx, col] += 1
        for idx, sign in zip(w_indices, w_signs):
            W[idx, col] += sign
    
    return {"U": U, "V": V, "W": W}
