# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    n = m = p = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_4x4_4x4():
    """Construct 4x4 * 4x4 -> 4x4 decomposition using Strassen. Rank 49."""
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    u_in, v_in, w_in = _strassen_2x2()
    
    # Outer Strassen steps
    # A_coeff: [A11, A12, A21, A22]
    # B_coeff: [B11, B12, B21, B22]
    # W_coeff: [C11, C12, C21, C22]
    steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    col_offset = 0
    for a_coeff, b_coeff, w_coeff in steps:
        for c in range(7):
            # For each i, j in 4x4
            for i in range(4):
                for j in range(4):
                    # Determine block index
                    bi = i // 2
                    bj = j // 2
                    b_idx = bi * 2 + bj # 0:11, 1:12, 2:21, 3:22
                    
                    # Local indices in 2x2 block
                    li = i % 2
                    lj = j % 2
                    l_idx = li * 2 + lj
                    
                    # U contribution
                    u_val = u_in[l_idx, c] * a_coeff[b_idx]
                    if u_val != 0:
                        U[i * 4 + j, col_offset + c] += u_val
                        
            for j in range(4):
                for k in range(4):
                    bi = j // 2
                    bk = k // 2
                    b_idx = bi * 2 + bk
                    li = j % 2
                    lk = k % 2
                    l_idx = li * 2 + lk
                    v_val = v_in[l_idx, c] * b_coeff[b_idx]
                    if v_val != 0:
                        V[j * 4 + k, col_offset + c] += v_val
                        
            for k in range(4):
                for i in range(4):
                    bi = k // 2
                    bi_i = i // 2
                    b_idx = bi * 2 + bi_i
                    li = k % 2
                    li_i = i % 2
                    l_idx = li * 2 + li_i
                    w_val = w_in[l_idx, c] * w_coeff[b_idx]
                    if w_val != 0:
                        W[k * 4 + i, col_offset + c] += w_val
                        
            col_offset += 7
            
    return U, V, W

def _construct_4x4_4x3():
    """Construct 4x4 * 4x3 -> 4x3 decomposition using schoolbook. Rank 48."""
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((16, rank))
    V = np.zeros((12, rank))
    W = np.zeros((12, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build 4x4 * 4x4 part
    U_444, V_444, W_444 = _construct_4x4_4x4()
    
    # Build 4x4 * 4x3 part
    U_443, V_443, W_443 = _construct_4x4_4x3()
    
    # Combine
    rank = 49 + 48
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    U[:, 0:49] = U_444
    V[:, 0:49] = V_444
    W[:, 0:49] = W_444
    
    # For the 4x4 * 4x3 part, the V and W matrices have different sizes.
    # V_443 is 12x48 (m*p = 4*3=12 for the right part? No, m=4, p=3 for the subproblem).
    # But in the full problem, p=7.
    # The 4x4 * 4x3 part corresponds to columns k=4,5,6 of the full B.
    # So V_full[j, k] for k in 4..6.
    # V_443[j, k'] where k' in 0..2 maps to V_full[j, k'+4].
    
    # U_443 is 16x48.
    U[:, 49:97] = U_443
    
    # V_443 is 12x48 (rows j*3+k).
    # We need to place it in V[:, 49:97] which is 28x48.
    # Rows 0..11 of V_443 correspond to j=0..3, k=0..2.
    # In full V, these are j=0..3, k=4..6.
    # Index in full V: j*7 + k.
    # Index in V_443: j*3 + k.
    
    for j in range(4):
        for k in range(3):
            v_idx_443 = j * 3 + k
            v_idx_full = j * 7 + (k + 4)
            V[v_idx_full, 49:97] = V_443[v_idx_443, :]
            
    # W_443 is 12x48 (rows k*4+i).
    # In full W, these are k=4..6, i=0..3.
    # Index in full W: k*4 + i.
    # Index in W_443: k'*4 + i where k'=k-4.
    
    for k in range(3):
        for i in range(4):
            w_idx_443 = k * 4 + i
            w_idx_full = (k + 4) * 4 + i
            W[w_idx_full, 49:97] = W_443[w_idx_443, :]
            
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
