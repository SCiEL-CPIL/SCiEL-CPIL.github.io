import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Construct the target tensor T of shape (nm, mp, pn)
    # T[u, v, w] = 1 if j(u)==j(v), k(v)==k(w), i(w)==i(u)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T = np.zeros((nm, mp, pn))
    for u in range(nm):
        i_u = u // m
        j_u = u % m
        for v in range(mp):
            j_v = v // p
            k_v = v % p
            if j_u != j_v:
                continue
            for w in range(pn):
                k_w = w // n
                i_w = w % n
                if k_v != k_w or i_u != i_w:
                    continue
                T[u, v, w] = 1.0
                
    # Function to verify if U, V, W exactly decompose T
    def verify(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Reconstruct T_rec[u, v, w] = sum_k U[u,k] V[v,k] W[w,k]
        # We can do this efficiently:
        # T_rec = np.einsum('uk,vk,wk->uvw', U, V, W)
        # But this is O(r * nm * mp * pn). For r=12, 12*6*12*8 = 6912 ops. Very fast.
        T_rec = np.einsum('uk,vk,wk->uvw', U, V, W)
        # Check if T_rec is close to T
        if not np.allclose(T_rec, T, atol=1e-9):
            return False
        # Check if entries are integer or half-integer
        # Multiply by 2 and check if integer
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if not np.allclose(U2, np.round(U2), atol=1e-9):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-9):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-9):
            return False
        return True

    # ALS Solver
    def solve_als(r, max_iter=1000, tol=1e-12):
        # Initialize
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        prev_err = np.inf
        for it in range(max_iter):
            # Update U
            # T_rec = sum_k U[:,k] (V[:,k] * W[:,k])^T ? No.
            # T[u,v,w] = sum_k U[u,k] V[v,k] W[w,k]
            # For fixed V, W, U[u,k] is determined by minimizing || T - ... ||^2
            # This is a linear system for U[:,k]? No, U, V, W are coupled in the sum.
            # Actually, for fixed V, W, the term involving U is:
            # sum_k U[u,k] (sum_{v,w} T[u,v,w] V[v,k] W[w,k] - sum_{k'} U[u,k'] sum_{v,w} V[v,k'] W[w,k'] V[v,k] W[w,k] )?
            # This is complex.
            # Simpler: Treat it as a matrix factorization problem for each slice?
            # No, standard ALS for 3-way tensor rank-1 sum:
            # Minimize || T - sum_k u_k v_k^T w_k^T ||_F^2
            # Fix v_k, w_k. Then for each k, u_k is the best rank-1 approximation of the residual?
            # No, u_k appears in all terms.
            # Actually, if we fix V and W, the problem for U is:
            # Minimize || T - sum_k u_k (v_k \otimes w_k)^T ||^2
            # This is a linear least squares problem for the vectorization of U?
            # Let M_k = v_k \otimes w_k (size nm*mp*pn? No, v_k is mp, w_k is pn. M_k is mp*pn).
            # T is nm x mp x pn.
            # We can reshape T to nm x (mp*pn).
            # Then T = sum_k u_k M_k^T?
            # T[u, v, w] = sum_k U[u,k] V[v,k] W[w,k]
            # Let M_k[v, w] = V[v,k] W[w,k].
            # Then T[u, :, :] = sum_k U[u,k] M_k.
            # So for each u, T[u] is a linear combination of M_k with coefficients U[u,:].
            # We can solve for U by stacking M_k.
            # Let M be a matrix of size (mp*pn) x r, where column k is M_k.flatten().
            # Then T.flatten(1) (size nm x mp*pn) = U M^T?
            # T[u, v, w] = sum_k U[u,k] M_k[v,w]
            # So T_mat (nm x mp*pn) = U @ M_mat.T
            # We want to find U such that U M_mat.T approx T_mat.
            # This is a linear system: M_mat U^T = T_mat^T.
            # U^T = M_mat^+ T_mat^T.
            
            M_mat = np.zeros((mp * pn, r))
            for k in range(r):
                M_k = np.outer(V[:, k], W[:, k])
                M_mat[:, k] = M_k.flatten()
                
            T_mat = T.reshape(nm, mp * pn)
            # Solve U^T = M_mat^+ T_mat^T
            # Use lstsq
            U_new, _, _, _ = np.linalg.lstsq(M_mat, T_mat.T, rcond=None)
            U = U_new.T
            
            # Update V
            # T[u, v, w] = sum_k V[v,k] (U[u,k] W[w,k])
            # For each v, T[:, v, :] is linear combination of N_k[u,w] = U[u,k] W[w,k].
            # N_mat size (nm*pn) x r.
            N_mat = np.zeros((nm * pn, r))
            for k in range(r):
                N_k = np.outer(U[:, k], W[:, k])
                N_mat[:, k] = N_k.flatten()
            T_mat_v = T.reshape(mp, nm * pn) # T[v, u, w] ? No, T is (u, v, w).
            # We need T[:, v, :] which is T[0:nm, v, 0:pn].
            # Reshape T to (mp, nm*pn) by moving axis 1 to 0.
            T_mat_v = np.moveaxis(T, 1, 0).reshape(mp, nm * pn)
            V_new, _, _, _ = np.linalg.lstsq(N_mat, T_mat_v.T, rcond=None)
            V = V_new.T
            
            # Update W
            # T[u, v, w] = sum_k W[w,k] (U[u,k] V[v,k])
            # P_k[u,v] = U[u,k] V[v,k].
            P_mat = np.zeros((nm * mp, r))
            for k in range(r):
                P_k = np.outer(U[:, k], V[:, k])
                P_mat[:, k] = P_k.flatten()
            T_mat_w = np.moveaxis(T, 2, 0).reshape(pn, nm * mp)
            W_new, _, _, _ = np.linalg.lstsq(P_mat, T_mat_w.T, rcond=None)
            W = W_new.T
            
            # Calculate error
            T_rec = np.einsum('uk,vk,wk->uvw', U, V, W)
            err = np.linalg.norm(T_rec - T, 'fro')
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, err

    # Time guard
    start_time = time.time()
    deadline = start_time + 110
    
    # Try ranks from 12 upwards
    for r in range(12, 25):
        if time.time() > deadline:
            break
            
        # Try multiple restarts
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            U, V, W, err = solve_als(r, max_iter=500)
            
            # Check error
            if err < 1e-8:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    # Convert to lists
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
