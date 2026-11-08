import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 4
    
    # Build the matrix multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def extract_rank1(R, max_iter=200, seed=0):
        np.random.seed(seed)
        u = np.random.randn(n*m)
        v = np.random.randn(m*p)
        w = np.random.randn(p*n)
        
        best_u, best_v, best_w = u, v, w
        best_err = np.inf
        
        for it in range(max_iter):
            vw = np.outer(v, w)
            u_new = np.einsum('ijk,jk->i', R, vw)
            nu = np.linalg.norm(u_new)
            if nu < 1e-12: break
            u_new /= nu
            u = u_new
            
            uw = np.outer(u, w)
            v_new = np.einsum('ijk,ik->j', R, uw)
            nv = np.linalg.norm(v_new)
            if nv < 1e-12: break
            v_new /= nv
            v = v_new
            
            uv = np.outer(u, v)
            w_new = np.einsum('ijk,ij->k', R, uv)
            nw = np.linalg.norm(w_new)
            if nw < 1e-12: break
            w_new /= nw
            w = w_new
            
            # Check convergence
            uvw = np.einsum('i,j,k->ijk', u, v, w)
            err = np.linalg.norm(R - uvw)
            if err < best_err:
                best_err = err
                best_u, best_v, best_w = u, v, w
            if err < 1e-10:
                break
        
        # Optimal scaling
        uvw = np.einsum('i,j,k->ijk', best_u, best_v, best_w)
        denom = np.dot(uvw.flatten(), uvw.flatten())
        if abs(denom) < 1e-15:
            return best_u, best_v, best_w
        alpha = np.dot(R.flatten(), uvw.flatten()) / denom
        if abs(alpha) < 1e-15:
            return best_u, best_v, best_w
        scale = np.sqrt(abs(alpha))
        sign = 1 if alpha > 0 else -1
        return best_u * scale, best_v * scale, best_w * sign * scale
    
    def verify(U_mat, V_mat, W_mat, rank, atol=1e-7):
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U_mat[:,r], V_mat[:,r], W_mat[:,r])
        return np.allclose(T_recon, T, atol=atol)
    
    best_result = None
    
    for target_rank in [20, 21]:
        if time.time() > deadline - 30:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            R = T.copy()
            U_cols, V_cols, W_cols = [], [], []
            
            success = True
            for r in range(target_rank):
                if time.time() > deadline - 10:
                    success = False
                    break
                u, v, w = extract_rank1(R, max_iter=150, seed=attempt * 100 + r)
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
                R -= np.einsum('i,j,k->ijk', u, v, w)
            
            if not success:
                continue
            
            if np.linalg.norm(R) < 1e-5:
                U_mat = np.column_stack(U_cols)
                V_mat = np.column_stack(V_cols)
                W_mat = np.column_stack(W_cols)
                if verify(U_mat, V_mat, W_mat, target_rank):
                    best_result = (target_rank, U_mat, V_mat, W_mat)
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        rank, U_mat, V_mat, W_mat = best_result
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    
    # Fallback: rank-22 Strassen construction
    rank = 22
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx, col] += val
        for idx, val in v_indices:
            V[idx, col] += val
        for idx, val in w_indices:
            W[idx, col] += val
    
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
