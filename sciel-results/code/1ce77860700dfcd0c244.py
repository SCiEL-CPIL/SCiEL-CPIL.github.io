import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = idx_u(i, j)
                    b = idx_v(j, k)
                    c = idx_w(k, i)
                    s = np.sum(W[c, :] * U[a, :] * V[b, :])
                    if abs(s - 1.0) > tol:
                        return False
        # Check zeros
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Check if (a,b,c) corresponds to a valid multiplication term
                    # a = i*m+j, b = j*p+k, c = k*n+i
                    # Valid if j from a matches j from b, and k from b matches k from c, and i from c matches i from a
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue # This is a "one" index, already checked
                    
                    s = np.sum(W[c, :] * U[a, :] * V[b, :])
                    if abs(s) > tol:
                        return False
        return True

    # --- Try Bini Rank 23 ---
    R = 23
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    # Bini Terms (u, v, w) where u, v, w are sparse vectors
    # Term format: (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    bini_terms = [
        # 1
        ([idx_u(0,0), idx_u(1,1)], [1, 1], [idx_v(0,0), idx_v(1,1)], [1, 1], [idx_w(0,0), idx_w(1,1)], [1, 1]),
        # 2
        ([idx_u(0,0), idx_u(1,1)], [1, -1], [idx_v(0,0), idx_v(1,1)], [1, -1], [idx_w(0,0), idx_w(1,1)], [1, 1]),
        # 3
        ([idx_u(0,1), idx_u(1,0)], [1, 1], [idx_v(0,1), idx_v(1,0)], [1, 1], [idx_w(1,0), idx_w(0,1)], [1, 1]),
        # 4
        ([idx_u(0,1), idx_u(1,0)], [1, -1], [idx_v(0,1), idx_v(1,0)], [1, -1], [idx_w(1,0), idx_w(0,1)], [1, 1]),
        # 5
        ([idx_u(0,0), idx_u(0,1)], [1, 1], [idx_v(1,0), idx_v(1,1)], [1, 1], [idx_w(0,0), idx_w(1,0)], [1, 1]),
        # 6
        ([idx_u(1,0), idx_u(1,1)], [1, 1], [idx_v(0,0), idx_v(0,1)], [1, 1], [idx_w(0,1), idx_w(1,1)], [1, 1]),
        # 7
        ([idx_u(0,0), idx_u(0,1)], [1, -1], [idx_v(1,1), idx_v(1,0)], [1, -1], [idx_w(0,0), idx_w(1,0)], [1, 1]),
        # 8
        ([idx_u(1,1), idx_u(1,0)], [1, -1], [idx_v(0,0), idx_v(0,1)], [1, -1], [idx_w(0,1), idx_w(1,1)], [1, 1]),
        # 9
        ([idx_u(0,0), idx_u(0,2)], [1, 1], [idx_v(1,0), idx_v(1,2)], [1, 1], [idx_w(0,0), idx_w(2,0)], [1, 1]),
        # 10
        ([idx_u(1,0), idx_u(1,2)], [1, 1], [idx_v(0,0), idx_v(0,2)], [1, 1], [idx_w(0,1), idx_w(2,1)], [1, 1]),
        # 11
        ([idx_u(0,0), idx_u(0,2)], [1, -1], [idx_v(1,2), idx_v(1,0)], [1, -1], [idx_w(0,0), idx_w(2,0)], [1, 1]),
        # 12
        ([idx_u(1,2), idx_u(1,0)], [1, -1], [idx_v(0,0), idx_v(0,2)], [1, -1], [idx_w(0,1), idx_w(2,1)], [1, 1]),
        # 13
        ([idx_u(0,1), idx_u(0,2)], [1, 1], [idx_v(1,1), idx_v(1,2)], [1, 1], [idx_w(1,0), idx_w(2,0)], [1, 1]),
        # 14
        ([idx_u(1,1), idx_u(1,2)], [1, 1], [idx_v(0,1), idx_v(0,2)], [1, 1], [idx_w(1,1), idx_w(2,1)], [1, 1]),
        # 15
        ([idx_u(0,1), idx_u(0,2)], [1, -1], [idx_v(1,2), idx_v(1,1)], [1, -1], [idx_w(1,0), idx_w(2,0)], [1, 1]),
        # 16
        ([idx_u(1,2), idx_u(1,1)], [1, -1], [idx_v(0,1), idx_v(0,2)], [1, -1], [idx_w(1,1), idx_w(2,1)], [1, 1]),
        # 17
        ([idx_u(0,0), idx_u(1,0)], [1, 1], [idx_v(0,1), idx_v(1,1)], [1, 1], [idx_w(0,0), idx_w(1,0)], [1, 1]),
        # 18
        ([idx_u(0,1), idx_u(1,1)], [1, 1], [idx_v(0,0), idx_v(1,0)], [1, 1], [idx_w(0,1), idx_w(1,1)], [1, 1]),
        # 19
        ([idx_u(0,0), idx_u(1,0)], [1, -1], [idx_v(1,1), idx_v(0,1)], [1, -1], [idx_w(0,0), idx_w(1,0)], [1, 1]),
        # 20
        ([idx_u(1,1), idx_u(0,1)], [1, -1], [idx_v(0,0), idx_v(1,0)], [1, -1], [idx_w(0,1), idx_w(1,1)], [1, 1]),
        # 21
        ([idx_u(0,0), idx_u(1,2)], [1, 1], [idx_v(0,2), idx_v(1,2)], [1, 1], [idx_w(0,0), idx_w(2,0)], [1, 1]),
        # 22
        ([idx_u(0,2), idx_u(1,2)], [1, 1], [idx_v(0,0), idx_v(1,0)], [1, 1], [idx_w(0,2), idx_w(2,2)], [1, 1]), # Correction: w indices for c02, c22?
        # 23
        ([idx_u(0,0), idx_u(1,2)], [1, -1], [idx_v(1,2), idx_v(0,2)], [1, -1], [idx_w(0,0), idx_w(2,0)], [1, 1]),
    ]
    
    for r, term in enumerate(bini_terms):
        u_idx, u_val, v_idx, v_val, w_idx, w_val = term
        U[u_idx, r] = u_val
        V[v_idx, r] = v_val
        W[w_idx, r] = w_val
        
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # --- Fallback Rank 26 ---
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))
    
    # Strassen terms for 2x2 core
    U26[idx_u(0,0), 0] = 1; U26[idx_u(1,1), 0] = 1
    V26[idx_v(0,0), 0] = 1; V26[idx_v(1,1), 0] = 1
    W26[idx_w(0,0), 0] = 1; W26[idx_w(1,1), 0] = 1
    
    U26[idx_u(1,0), 1] = 1; U26[idx_u(1,1), 1] = 1
    V26[idx_v(0,0), 1] = 1
    W26[idx_w(0,1), 1] = 1; W26[idx_w(1,1), 1] = -1
    
    U26[idx_u(0,0), 2] = 1
    V26[idx_v(0,1), 2] = 1; V26[idx_v(1,1), 2] = -1
    W26[idx_w(1,0), 2] = 1; W26[idx_w(1,1), 2] = 1
    
    U26[idx_u(1,1), 3] = 1
    V26[idx_v(1,0), 3] = 1; V26[idx_v(0,0), 3] = -1
    W26[idx_w(0,0), 3] = 1; W26[idx_w(0,1), 3] = 1
    
    U26[idx_u(0,0), 4] = 1; U26[idx_u(0,1), 4] = 1
    V26[idx_v(1,1), 4] = 1
    W26[idx_w(0,0), 4] = -1; W26[idx_w(1,0), 4] = 1
    
    U26[idx_u(1,0), 5] = 1; U26[idx_u(0,0), 5] = -1
    V26[idx_v(0,0), 5] = 1; V26[idx_v(0,1), 5] = 1
    W26[idx_w(1,1), 5] = 1
    
    U26[idx_u(0,1), 6] = 1; U26[idx_u(1,1), 6] = -1
    V26[idx_v(1,0), 6] = 1; V26[idx_v(1,1), 6] = 1
    W26[idx_w(0,0), 6] = 1
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1
                
    if verify(U26, V26, W26):
        return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
        
    # Fallback Schoolbook 27
    R27 = 27
    U27 = np.zeros((nm, R27))
    V27 = np.zeros((mp, R27))
    W27 = np.zeros((pn, R27))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[idx_u(i,j), c] = 1
                V27[idx_v(j,k), c] = 1
                W27[idx_w(k,i), c] = 1
                c += 1
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
