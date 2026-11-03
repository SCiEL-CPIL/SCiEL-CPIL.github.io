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
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = idx_u(i, j)
                    b = idx_v(j, k)
                    c = idx_w(k, i)
                    s = np.sum(W[c, :] * U[a, :] * V[b, :])
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    s = np.sum(W[c, :] * U[a, :] * V[b, :])
                    if abs(s) > tol:
                        return False
        return True

    # --- Bini Rank 23 ---
    R = 23
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    # Bini's rank-23 decomposition of 3x3x3 matmul
    # Each term: (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    terms = [
        # Term 0: (0,0,0) + (1,1,1)
        ([idx_u(0,0), idx_u(1,1)], [1, 1], [idx_v(0,0), idx_v(1,1)], [1, 1], [idx_w(0,0), idx_w(1,1)], [1, 1]),
        # Term 1: (0,0,0) - (1,1,1)  
        ([idx_u(0,0), idx_u(1,1)], [1, -1], [idx_v(0,0), idx_v(1,1)], [1, -1], [idx_w(0,0), idx_w(1,1)], [1, 1]),
        # Term 2: (0,1,0) + (1,0,1)
        ([idx_u(0,1), idx_u(1,0)], [1, 1], [idx_v(0,1), idx_v(1,0)], [1, 1], [idx_w(0,0), idx_w(1,1)], [1, 1]),
        # Term 3: (0,1,0) - (1,0,1)
        ([idx_u(0,1), idx_u(1,0)], [1, -1], [idx_v(0,1), idx_v(1,0)], [1, -1], [idx_w(0,0), idx_w(1,1)], [1, 1]),
        # Term 4: (0,0,1) + (1,1,0)
        ([idx_u(0,0), idx_u(1,1)], [1, 1], [idx_v(0,1), idx_v(1,0)], [1, 1], [idx_w(0,1), idx_w(1,0)], [1, 1]),
        # Term 5: (0,0,1) - (1,1,0)
        ([idx_u(0,0), idx_u(1,1)], [1, -1], [idx_v(0,1), idx_v(1,0)], [1, -1], [idx_w(0,1), idx_w(1,0)], [1, 1]),
        # Term 6: (0,1,1) + (1,0,0)
        ([idx_u(0,1), idx_u(1,0)], [1, 1], [idx_v(0,0), idx_v(1,1)], [1, 1], [idx_w(0,1), idx_w(1,0)], [1, 1]),
        # Term 7: (0,1,1) - (1,0,0)
        ([idx_u(0,1), idx_u(1,0)], [1, -1], [idx_v(0,0), idx_v(1,1)], [1, -1], [idx_w(0,1), idx_w(1,0)], [1, 1]),
        # Term 8: (0,2,0) + (2,0,0)
        ([idx_u(0,2), idx_u(2,0)], [1, 1], [idx_v(0,0), idx_v(2,0)], [1, 1], [idx_w(0,0), idx_w(0,2)], [1, 1]),
        # Term 9: (0,2,0) - (2,0,0)
        ([idx_u(0,2), idx_u(2,0)], [1, -1], [idx_v(0,0), idx_v(2,0)], [1, -1], [idx_w(0,0), idx_w(0,2)], [1, 1]),
        # Term 10: (0,0,2) + (2,2,0)
        ([idx_u(0,0), idx_u(2,2)], [1, 1], [idx_v(0,2), idx_v(2,0)], [1, 1], [idx_w(0,2), idx_w(0,0)], [1, 1]),
        # Term 11: (0,0,2) - (2,2,0)
        ([idx_u(0,0), idx_u(2,2)], [1, -1], [idx_v(0,2), idx_v(2,0)], [1, -1], [idx_w(0,2), idx_w(0,0)], [1, 1]),
        # Term 12: (0,2,2) + (2,0,2)
        ([idx_u(0,2), idx_u(2,0)], [1, 1], [idx_v(0,2), idx_v(2,2)], [1, 1], [idx_w(0,2), idx_w(2,0)], [1, 1]),
        # Term 13: (0,2,2) - (2,0,2)
        ([idx_u(0,2), idx_u(2,0)], [1, -1], [idx_v(0,2), idx_v(2,2)], [1, -1], [idx_w(0,2), idx_w(2,0)], [1, 1]),
        # Term 14: (1,2,0) + (2,1,0)
        ([idx_u(1,2), idx_u(2,1)], [1, 1], [idx_v(0,0), idx_v(1,0)], [1, 1], [idx_w(0,1), idx_w(0,2)], [1, 1]),
        # Term 15: (1,2,0) - (2,1,0)
        ([idx_u(1,2), idx_u(2,1)], [1, -1], [idx_v(0,0), idx_v(1,0)], [1, -1], [idx_w(0,1), idx_w(0,2)], [1, 1]),
        # Term 16: (1,0,2) + (2,2,2)
        ([idx_u(1,0), idx_u(2,2)], [1, 1], [idx_v(0,2), idx_v(2,2)], [1, 1], [idx_w(0,1), idx_w(2,2)], [1, 1]),
        # Term 17: (1,0,2) - (2,2,2)
        ([idx_u(1,0), idx_u(2,2)], [1, -1], [idx_v(0,2), idx_v(2,2)], [1, -1], [idx_w(0,1), idx_w(2,2)], [1, 1]),
        # Term 18: (1,2,2) + (2,0,1)
        ([idx_u(1,2), idx_u(2,0)], [1, 1], [idx_v(0,2), idx_v(2,1)], [1, 1], [idx_w(0,1), idx_w(1,2)], [1, 1]),
        # Term 19: (1,2,2) - (2,0,1)
        ([idx_u(1,2), idx_u(2,0)], [1, -1], [idx_v(0,2), idx_v(2,1)], [1, -1], [idx_w(0,1), idx_w(1,2)], [1, 1]),
        # Term 20: (1,1,2) + (2,1,1)
        ([idx_u(1,1), idx_u(2,1)], [1, 1], [idx_v(0,2), idx_v(1,1)], [1, 1], [idx_w(0,1), idx_w(1,2)], [1, 1]),
        # Term 21: (1,1,2) - (2,1,1)
        ([idx_u(1,1), idx_u(2,1)], [1, -1], [idx_v(0,2), idx_v(1,1)], [1, -1], [idx_w(0,1), idx_w(1,2)], [1, 1]),
        # Term 22: (2,2,1) + (1,2,1)
        ([idx_u(2,2), idx_u(1,2)], [1, 1], [idx_v(0,1), idx_v(2,1)], [1, 1], [idx_w(1,2), idx_w(1,1)], [1, 1]),
    ]
    
    # Try scaling U by 0.5
    for r, term in enumerate(terms):
        u_idx, u_val, v_idx, v_val, w_idx, w_val = term
        for i, val in zip(u_idx, u_val):
            U[i, r] = val * 0.5
        for i, val in zip(v_idx, v_val):
            V[i, r] = val
        for i, val in zip(w_idx, w_val):
            W[i, r] = val
            
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try scaling V by 0.5
    U[:] = 0
    V[:] = 0
    W[:] = 0
    for r, term in enumerate(terms):
        u_idx, u_val, v_idx, v_val, w_idx, w_val = term
        for i, val in zip(u_idx, u_val):
            U[i, r] = val
        for i, val in zip(v_idx, v_val):
            V[i, r] = val * 0.5
        for i, val in zip(w_idx, w_val):
            W[i, r] = val
            
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try scaling W by 0.5
    U[:] = 0
    V[:] = 0
    W[:] = 0
    for r, term in enumerate(terms):
        u_idx, u_val, v_idx, v_val, w_idx, w_val = term
        for i, val in zip(u_idx, u_val):
            U[i, r] = val
        for i, val in zip(v_idx, v_val):
            V[i, r] = val
        for i, val in zip(w_idx, w_val):
            W[i, r] = val * 0.5
            
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # --- Fallback: Strassen-based rank 26 ---
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))
    
    # 7 Strassen terms for the 2x2 block (indices 0,1)
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
    
    # Remaining 19 terms for index 2 (schoolbook for those)
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
