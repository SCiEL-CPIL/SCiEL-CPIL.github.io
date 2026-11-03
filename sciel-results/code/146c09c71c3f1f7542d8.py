import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Target Tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    T = np.zeros((num_u, num_v, num_w), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    schoolbook_rank = n * m * p
    
    def get_schoolbook():
        U = np.zeros((num_u, schoolbook_rank))
        V = np.zeros((num_v, schoolbook_rank))
        W = np.zeros((num_w, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1.0
                    V[v, col] = 1.0
                    W[w, col] = 1.0
                    col += 1
        return U, V, W, schoolbook_rank

    # Try to find lower rank
    # We will try ranks from schoolbook_rank - 1 down to some limit
    # Using ALS (Alternating Least Squares)
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = get_schoolbook()
    
    # Limit search to avoid timeout
    # ALS iterations per rank
    max_iters = 50
    tol = 1e-6
    
    # We will try a few ranks below 60
    # If rank is 60, we are done.
    # If rank is 59, 58, etc.
    
    # To save time, we only try a few ranks and a few random restarts
    # If we find a valid integer/half-integer solution, we return it.
    
    # Helper to check if solution is exact and has rational coefficients
    def check_solution(U, V, W, r):
        # Reconstruct tensor
        # T_recon[u,v,w] = sum_t U[u,t] V[v,t] W[w,t]
        # We can do this efficiently using einsum or loops
        # For small dims, loop is fine
        T_recon = np.zeros((num_u, num_v, num_w), dtype=np.float64)
        for t in range(r):
            T_recon += np.einsum('u,v,w->uvw', U[:, t], V[:, t], W[:, t])
        
        # Check error
        err = np.linalg.norm(T_recon - T)
        if err > 1e-5:
            return False
        
        # Check for integer/half-integer entries
        # Multiply by 2 and check if close to integer
        U2 = 2 * U
        V2 = 2 * V
        W2 = 2 * W
        
        if not np.allclose(U2, np.round(U2), atol=1e-5):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-5):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-5):
            return False
            
        return True

    def als_decompose(rank, max_iters=50, tol=1e-6):
        # Initialize
        U = np.random.randn(num_u, rank) * 0.1
        V = np.random.randn(num_v, rank) * 0.1
        W = np.random.randn(num_w, rank) * 0.1
        
        prev_err = np.inf
        
        for it in range(max_iters):
            # Update U
            # T_u = U_u Z^T where Z_{(v,w), t} = V_{v,t} W_{w,t}
            Z = np.einsum('vt,wt->vwt', V, W).reshape(num_v * num_w, rank)
            T_u_flat = T.reshape(num_u, num_v * num_w)
            # Solve U_u Z^T = T_u_flat
            # U_u = T_u_flat (Z^T)^+
            # Use lstsq
            U_new = np.zeros((num_u, rank))
            for u in range(num_u):
                U_new[u, :] = np.linalg.lstsq(Z.T, T_u_flat[u, :], rcond=None)[0]
            U = U_new
            
            # Update V
            # T_v = V_w Y^T where Y_{(u,w), t} = U_{u,t} W_{w,t}
            Y = np.einsum('ut,wt->uwt', U, W).reshape(num_u * num_w, rank)
            T_v_flat = T.transpose(1, 0, 2).reshape(num_v, num_u * num_w)
            V_new = np.zeros((num_v, rank))
            for v in range(num_v):
                V_new[v, :] = np.linalg.lstsq(Y.T, T_v_flat[v, :], rcond=None)[0]
            V = V_new
            
            # Update W
            # T_w = W_w X^T where X_{(u,v), t} = U_{u,t} V_{v,t}
            X = np.einsum('ut,vt->uvt', U, V).reshape(num_u * num_v, rank)
            T_w_flat = T.transpose(2, 0, 1).reshape(num_w, num_u * num_v)
            W_new = np.zeros((num_w, rank))
            for w in range(num_w):
                W_new[w, :] = np.linalg.lstsq(X.T, T_w_flat[w, :], rcond=None)[0]
            W = W_new
            
            # Calculate error
            T_recon = np.einsum('ut,vt,wt->uvw', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W

    # Search for lower rank
    # We try ranks 59, 58, 57...
    # To be safe, we limit the number of attempts
    search_ranks = [59, 58, 57, 56, 55]
    restarts = 3
    
    for r in search_ranks:
        if time.time() > deadline - 20:
            break
        for restart in range(restarts):
            if time.time() > deadline - 10:
                break
            U, V, W = als_decompose(r, max_iters=20, tol=1e-5)
            if check_solution(U, V, W, r):
                # Found a better solution
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        if best_rank < schoolbook_rank:
            break

    # If no better found, return schoolbook
    if best_rank == schoolbook_rank:
        best_U, best_V, best_W, _ = get_schoolbook()
        
    # Convert to lists of lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
