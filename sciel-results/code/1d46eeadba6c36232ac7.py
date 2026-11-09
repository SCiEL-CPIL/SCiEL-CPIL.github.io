import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R
    
    U_fb, V_fb, W_fb, R_fb = schoolbook()
    
    # Target tensor: T[i, j, k] = 1
    T_target = np.ones((n, m, p))
    
    def tensor_contract(T, vec_j, vec_k, mode=0):
        """Contract T with vec_j (mode 1) and vec_k (mode 2) to get vec_i (mode 0)."""
        # T is (n, m, p)
        # vec_j is (m,)
        # vec_k is (p,)
        # Result is (n,)
        # Result[i] = sum_{j,k} T[i,j,k] * vec_j[j] * vec_k[k]
        # This is T @ vec_j @ vec_k
        return np.einsum('ijk,j,k->i', T, vec_j, vec_k)

    def tensor_contract_mode1(T, vec_i, vec_k):
        """Contract T with vec_i (mode 0) and vec_k (mode 2) to get vec_j (mode 1)."""
        return np.einsum('ijk,i,k->j', T, vec_i, vec_k)

    def tensor_contract_mode2(T, vec_i, vec_j):
        """Contract T with vec_i (mode 0) and vec_j (mode 1) to get vec_k (mode 2)."""
        return np.einsum('ijk,i,j->k', T, vec_i, vec_j)

    def norm(vec):
        return np.linalg.norm(vec)

    def best_rank1_approx(T, num_starts=10):
        """Find best rank-1 approximation of T using power iteration."""
        best_u, best_v, best_w = None, None, None
        best_score = -1
        
        for s in range(num_starts):
            # Random initialization
            u = np.random.randn(n)
            v = np.random.randn(m)
            w = np.random.randn(p)
            u /= norm(u)
            v /= norm(v)
            w /= norm(w)
            
            for it in range(50):
                # Update u
                u_new = tensor_contract(T, v, w, mode=0)
                if norm(u_new) < 1e-10:
                    break
                u_new /= norm(u_new)
                
                # Update v
                v_new = tensor_contract_mode1(T, u_new, w)
                if norm(v_new) < 1e-10:
                    break
                v_new /= norm(v_new)
                
                # Update w
                w_new = tensor_contract_mode2(T, u_new, v_new)
                if norm(w_new) < 1e-10:
                    break
                w_new /= norm(w_new)
                
                if np.allclose(u, u_new) and np.allclose(v, v_new) and np.allclose(w, w_new):
                    break
                
                u, v, w = u_new, v_new, w_new
            
            # Score: inner product of T with u v w
            score = np.sum(T * np.outer(np.outer(u, v), w))
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u, v, w
        
        return best_u, best_v, best_w, best_score

    def snap_and_verify(U, V, W, R, tol=1e-6):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        U_3d = U_snap.reshape(n, m, R)
        V_3d = V_snap.reshape(m, p, R)
        W_3d = W_snap.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.sum((T - 1.0)**2)
        if err < tol:
            return U_snap, V_snap, W_snap
        
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        U_3d = U_int.reshape(n, m, R)
        V_3d = V_int.reshape(m, p, R)
        W_3d = W_int.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.sum((T - 1.0)**2)
        if err < tol:
            return U_int, V_int, W_int
            
        return None

    # Greedy Search
    best_rank = n * m * p
    best_factors = None
    
    # We will try to build the decomposition term by term
    # Start with a high rank limit and see if we can reach 0 error
    # Since we don't know the rank, we iterate R from 50 to 62
    
    for R in range(50, 63):
        if time.time() > deadline:
            break
            
        T_res = T_target.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for r in range(R):
            if time.time() > deadline:
                break
            
            # Find best rank-1 approx of T_res
            u, v, w, score = best_rank1_approx(T_res, num_starts=5)
            
            if score < 1e-8:
                # Residual is effectively zero
                if norm(T_res) < 1e-8:
                    # Success
                    U = np.column_stack(U_list) if U_list else np.zeros((n*m, 0))
                    V = np.column_stack(V_list) if V_list else np.zeros((m*p, 0))
                    W = np.column_stack(W_list) if W_list else np.zeros((p*n, 0))
                    
                    snapped = snap_and_verify(U, V, W, len(U_list))
                    if snapped:
                        U_s, V_s, W_s = snapped
                        best_rank = len(U_list)
                        best_factors = (U_s, V_s, W_s)
                        break
                else:
                    break
            
            # Subtract
            T_res -= np.outer(np.outer(u, v), w)
            
            # Store
            U_list.append(u * m) # Wait, U is (n*m, R). u is (n,). We need to expand?
            # No, the tensor is T[i,j,k].
            # Rank 1 term is u[i] * v[j] * w[k].
            # U[i*m+j, r] = u[i]? No.
            # The decomposition is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
            # If we have rank-1 terms u_r, v_r, w_r, then:
            # U[i*m+j, r] should be u_r[i]? But U depends on i and j.
            # In the standard tensor decomposition T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
            # The rank-1 term is U_r[i,j] * V_r[j,k] * W_r[k,i].
            # My power iteration found u[i], v[j], w[k] such that sum_r u_r[i] v_r[j] w_r[k] approx T[i,j,k].
            # This corresponds to U_r[i,j] = u_r[i], V_r[j,k] = v_r[j], W_r[k,i] = w_r[k].
            # So U[i*m+j, r] = u_r[i].
            
            # Flatten u to (n*m,)? No, U is (n*m, R).
            # U[i*m+j, r] = u_r[i].
            # So for a fixed r, the column U[:, r] has values u_r[i] repeated m times?
            # Let's check:
            # Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # = Sum_r (u_r[i]) (v_r[j]) (w_r[k])
            # Yes.
            
            U_col = np.zeros(n * m)
            for i in range(n):
                for j in range(m):
                    U_col[i * m + j] = u[i]
            U_list.append(U_col)
            
            V_col = np.zeros(m * p)
            for j in range(m):
                for k in range(p):
                    V_col[j * p + k] = v[j]
            V_list.append(V_col)
            
            W_col = np.zeros(p * n)
            for k in range(p):
                for i in range(n):
                    W_col[k * n + i] = w[k]
            W_list.append(W_col)
            
            if norm(T_res) < 1e-8:
                U = np.column_stack(U_list)
                V = np.column_stack(V_list)
                W = np.column_stack(W_list)
                snapped = snap_and_verify(U, V, W, len(U_list))
                if snapped:
                    U_s, V_s, W_s = snapped
                    best_rank = len(U_list)
                    best_factors = (U_s, V_s, W_s)
                    break
        
        if best_factors:
            break

    if best_factors:
        U, V, W = best_factors
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
