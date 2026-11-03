import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 2, 4, 4
    # We aim for rank 28 using block Strassen
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(r, c): return r * m + c
    def v_idx(r, c): return r * p + c
    # W index: W[k*n+i, r] where i is row of C (0..n-1), k is col of C (0..p-1)
    # So index = k*n + i
    def w_idx(i, k): return k * n + i
    
    current_rank = 0
    
    # Blocks: (ar, ac, br, bc, ci, ck)
    # ar, ac: A block start row, col
    # br, bc: B block start row, col
    # ci, ck: C block start row, col
    # A is 2x4, B is 4x4, C is 2x4.
    # We use 2x2 blocks for A, B, C?
    # A has 1x2 blocks of 2x2?
    # A[0:2, 0:2] is A0. A[0:2, 2:4] is A1.
    # B[0:2, 0:2] is B00. B[0:2, 2:4] is B01.
    # B[2:4, 0:2] is B10. B[2:4, 2:4] is B11.
    # C[0:2, 0:2] is C0. C[0:2, 2:4] is C1.
    
    blocks = [
        (0, 0, 0, 0, 0, 0), # A0, B00 -> C0
        (0, 2, 2, 0, 0, 0), # A1, B10 -> C0
        (0, 0, 0, 2, 0, 2), # A0, B01 -> C1
        (0, 2, 2, 2, 0, 2)  # A1, B11 -> C1
    ]
    
    for (ar, ac, br, bc, ci, ck) in blocks:
        # 2x2 indices
        u11 = u_idx(ar, ac)
        u12 = u_idx(ar, ac + 1)
        u21 = u_idx(ar + 1, ac)
        u22 = u_idx(ar + 1, ac + 1)
        
        v11 = v_idx(br, bc)
        v12 = v_idx(br, bc + 1)
        v21 = v_idx(br + 1, bc)
        v22 = v_idx(br + 1, bc + 1)
        
        # C block is 2x2 starting at (ci, ck)
        w11 = w_idx(ci, ck)
        w12 = w_idx(ci, ck + 1)
        w21 = w_idx(ci + 1, ck)
        w22 = w_idx(ci + 1, ck + 1)
        
        # Strassen 7
        # M1
        U[u11, current_rank] += 1; U[u22, current_rank] += 1
        V[v11, current_rank] += 1; V[v22, current_rank] += 1
        W[w11, current_rank] += 1; W[w22, current_rank] += 1
        current_rank += 1
        
        # M2
        U[u21, current_rank] += 1; U[u22, current_rank] += 1
        V[v11, current_rank] += 1
        W[w21, current_rank] += 1; W[w22, current_rank] -= 1
        current_rank += 1
        
        # M3
        U[u11, current_rank] += 1
        V[v12, current_rank] += 1; V[v22, current_rank] -= 1
        W[w12, current_rank] += 1; W[w22, current_rank] += 1
        current_rank += 1
        
        # M4
        U[u22, current_rank] += 1
        V[v21, current_rank] += 1; V[v11, current_rank] -= 1
        W[w11, current_rank] += 1; W[w21, current_rank] += 1
        current_rank += 1
        
        # M5
        U[u11, current_rank] += 1; U[u12, current_rank] += 1
        V[v22, current_rank] += 1
        W[w11, current_rank] -= 1; W[w12, current_rank] += 1
        current_rank += 1
        
        # M6
        U[u21, current_rank] += 1; U[u11, current_rank] -= 1
        V[v11, current_rank] += 1; V[v12, current_rank] += 1
        W[w22, current_rank] += 1
        current_rank += 1
        
        # M7
        U[u12, current_rank] += 1; U[u22, current_rank] -= 1
        V[v21, current_rank] += 1; V[v22, current_rank] += 1
        W[w11, current_rank] += 1
        current_rank += 1
        
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
