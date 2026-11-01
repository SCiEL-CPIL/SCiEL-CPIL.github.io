import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145

def _time_exceeded():
    return time.time() - _start_time > TIME_LIMIT

def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Shapes: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # 7 Strassen terms
    # u[i*2+j], v[j*2+k], w[k*2+i]
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),   # M2
        ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),   # M3
        ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),   # M4
        ([1,0,1,0], [0,0,0,1], [-1,0,1,0]),   # M5
        ([-1,0,0,1], [1,0,1,0], [0,0,0,1]),   # M6
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),   # M7
    ]
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    for t, (u, v, w) in enumerate(terms):
        U[:, t] = u
        V[:, t] = v
        W[:, t] = w
    return U, V, W

def construct():
    n, m, p = 2, 2, 15
    
    # Structural Change: Use a list of blocks to define the decomposition.
    # Each block is (size, rank, U_block, V_block, W_block)
    # For 2x2x2 Strassen
    U_s, V_s, W_s = _get_strassen_2x2x2()
    
    # For 2x2x1 Schoolbook
    U_1 = np.ones((4, 1), dtype=int)
    V_1 = np.ones((4, 1), dtype=int)
    W_1 = np.ones((4, 1), dtype=int)
    
    # Define blocks for p=15
    # 7 blocks of size 2, 1 block of size 1
    blocks = []
    for _ in range(7):
        blocks.append((2, U_s, V_s, W_s))
    blocks.append((1, U_1, V_1, W_1))
    
    total_rank = sum(len(U_b[1]) for U_b in blocks)
    
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    rank_offset = 0
    col_offset = 0
    
    for size, U_b, V_b, W_b in blocks:
        r_b = U_b.shape[1]
        
        # U is same for all blocks (only depends on i,j)
        U[:, rank_offset:rank_offset+r_b] = U_b
        
        # V depends on column offset
        # V indices: j*p + k
        # Block covers columns col_offset to col_offset + size - 1
        for j in range(m):
            for k in range(size):
                V[j*p + (col_offset + k), rank_offset:rank_offset+r_b] = V_b[j*2 + k, :]
                
        # W depends on column offset
        # W indices: k*n + i
        for k in range(size):
            for i in range(n):
                W[(col_offset + k)*n + i, rank_offset:rank_offset+r_b] = W_b[k*2 + i, :]
                
        rank_offset += r_b
        col_offset += size
        
    # Verification
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
                
    # Compute decomposition
    # computed = sum_t U[:,t] * V[:,t] * W[:,t]
    computed = np.zeros((n*m, m*p, p*n), dtype=float)
    for t in range(total_rank):
        computed += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        
    if not np.allclose(computed, T, atol=1e-10):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_fb = np.zeros((n*m, rank_school), dtype=int)
        V_fb = np.zeros((m*p, rank_school), dtype=int)
        W_fb = np.zeros((p*n, rank_school), dtype=int)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col_fb] = 1
                    V_fb[j*p+k, col_fb] = 1
                    W_fb[k*n+i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
