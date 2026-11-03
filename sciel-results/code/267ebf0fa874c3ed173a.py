import numpy as np
import time
import random

def construct():
    n, m, p = 3, 4, 7
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Greedy Peeling Algorithm with Random Sparse u
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    start_time = time.time()
    deadline = start_time + 140
    
    def is_zero(R, tol=0):
        return np.all(R == 0)

    def inner_product(R, u, v, w):
        return np.sum(R * np.outer(np.outer(u, v), w))

    while not is_zero(R):
        if time.time() > deadline:
            break
            
        best_score = -1
        best_u, best_v, best_w = None, None, None
        
        # Try a mix of basis vectors and random sparse vectors
        # 1. Basis vectors
        for a_idx in range(nm):
            u = np.zeros(nm, dtype=np.int32)
            u[a_idx] = 1
            
            M_u = R[a_idx, :, :]
            U_svd, s, Vt_svd = np.linalg.svd(M_u, full_matrices=False)
            if len(s) == 0: continue
            v = U_svd[:, 0]
            w = Vt_svd[0, :]
            
            v_int = np.round(v).astype(np.int32)
            w_int = np.round(w).astype(np.int32)
            
            score = inner_product(R, u, v_int, w_int)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u, v_int, w_int
        
        # 2. Random sparse vectors (support size 2 or 3)
        for _ in range(50): # Try 50 random u's
            if time.time() > deadline: break
            support_size = random.choice([2, 3])
            support = random.sample(range(nm), support_size)
            u = np.zeros(nm, dtype=np.int32)
            for idx in support:
                u[idx] = random.choice([-1, 1])
            
            M_u = np.tensordot(u, R, axes=([0], [0])) # Shape (mp, pn)
            U_svd, s, Vt_svd = np.linalg.svd(M_u, full_matrices=False)
            if len(s) == 0: continue
            v = U_svd[:, 0]
            w = Vt_svd[0, :]
            
            v_int = np.round(v).astype(np.int32)
            w_int = np.round(w).astype(np.int32)
            
            score = inner_product(R, u, v_int, w_int)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u, v_int, w_int
        
        if best_score <= 0:
            # Fallback
            idx = np.unravel_index(np.argmax(np.abs(R)), R.shape)
            best_u = np.zeros(nm, dtype=np.int32)
            best_u[idx[0]] = 1
            best_v = np.zeros(mp, dtype=np.int32)
            best_v[idx[1]] = 1
            best_w = np.zeros(pn, dtype=np.int32)
            best_w[idx[2]] = 1
            best_score = R[idx]
        
        R -= np.outer(np.outer(best_u, best_v), best_w)
        
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if is_zero(R):
            break

    if is_zero(R):
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
