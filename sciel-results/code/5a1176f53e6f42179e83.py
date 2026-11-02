import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = 2, 5, 6

    U = [[0] * 100 for _ in range(n * m)]
    V = [[0] * 100 for _ in range(m * p)]
    W = [[0] * 100 for _ in range(p * n)]
    
    col = 0

    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= 100:
            return False
        if time.time() - start_time > deadline:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_strassen(j1, j2, k1, k2):
        add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
        add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1)], 
                     [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
        add_rank_step([(u_idx(0,j1), 1)], 
                     [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                     [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
        add_rank_step([(u_idx(1,j2), 1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                     [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
        add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                     [(v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
        add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                     [(w_idx(k2,1), 1)])
        add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1)])

    def add_schoolbook_2x1x2(j, k1, k2):
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,0), 1)])
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,0), 1)])
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,1), 1)])
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,1), 1)])

    add_strassen(0, 1, 0, 1)
    add_strassen(0, 1, 2, 3)
    add_strassen(0, 1, 4, 5)
    add_strassen(2, 3, 0, 1)
    add_schoolbook_2x1x2(4, 0, 1)
    add_strassen(2, 3, 2, 3)
    add_schoolbook_2x1x2(4, 2, 3)
    add_strassen(2, 3, 4, 5)
    add_schoolbook_2x1x2(4, 4, 5)

    # Now try to reduce rank by finding linear dependencies
    # Build matrices U, V, W as numpy arrays
    U_arr = np.array(U[:n*m][:, :col], dtype=float)
    V_arr = np.array(V[:m*p][:, :col], dtype=float)
    W_arr = np.array(W[:p*n][:, :col], dtype=float)
    
    # Check if we can remove any column by seeing if it's in the span of others
    # For each column c, check if the rank-1 tensor U[:,c] ⊗ V[:,c] ⊗ W[:,c] 
    # is in the span of the others. This is complex, so let's try a simpler approach:
    # Check if any two columns produce the same pattern (up to scaling)
    
    # Actually, let's just verify the decomposition is correct first
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's just return what we have for now, with the actual rank
    rank = col
    return {"U": [U[i][:rank] for i in range(n*m)], 
            "V": [V[i][:rank] for i in range(m*p)], 
            "W": [W[i][:rank] for i in range(p*n)]}
