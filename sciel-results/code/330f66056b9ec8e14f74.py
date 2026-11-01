import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 120s limit, keep margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    dim_u = n * m  # 6
    dim_v = m * p  # 21
    dim_w = p * n  # 14
    
    # The tensor T has entries T[u, v, w] = 1 iff there exist i,j,k such that
    # u = i*m + j, v = j*p + k, w = k*n + i
    # For a given (u, v), there is at most one w that makes T[u,v,w] = 1.
    # Specifically, if u = i*m + j and v = j'*p + k, then we need j = j' for consistency.
    # If j != j', then T[u,v,w] = 0 for all w.
    # If j = j', then w must be k*n + i.
    
    # Precompute the "valid" (u, v) pairs and their corresponding w
    # For each (u, v), determine if it's a valid term and what w it corresponds to
    valid_pairs = {}  # (u, v) -> w
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_pairs[(u, v)] = w
    
    # For a random U, V with 0/1 entries, the system is:
    # For each (u, v, w): sum_t U[u,t] V[v,t] W[w,t] = T[u,v,w]
    # 
    # Key insight: For a fixed (u, v), if (u,v) is in valid_pairs, let w0 = valid_pairs[(u,v)].
    # Then T[u,v,w0] = 1 and T[u,v,w] = 0 for all w != w0.
    # If (u,v) is NOT in valid_pairs, then T[u,v,w] = 0 for all w.
    #
    # So the equations are:
    # For (u,v) in valid_pairs with w0:
    #   sum_t U[u,t] V[v,t] W[w0,t] = 1
    #   sum_t U[u,t] V[v,t] W[w,t] = 0 for all w != w0
    # For (u,v) not in valid_pairs:
    #   sum_t U[u,t] V[v,t] W[w,t] = 0 for all w
    #
    # Let c[u,v] = U[u,:] * V[v,:] (element-wise product, length r)
    # Then for each w, the equations involving W[w,:] are:
    #   c[u,v] dot W[w,:] = T[u,v,w] for all (u,v)
    #
    # So for each w, we have a linear system:
    #   A_w @ W[w,:] = b_w
    # where A_w has rows indexed by (u,v) and columns by t, with A_w[(u,v), t] = U[u,t]*V[v,t]
    # and b_w[(u,v)] = T[u,v,w]
    #
    # The matrix A_w is the same for all w! It's just the outer product structure.
    # Actually, A_w[(u,v), t] = U[u,t] * V[v,t], which doesn't depend on w.
    # So A is the same for all w.
    #
    # Therefore, for each w, we solve: A @ W[w,:] = b_w
    # where A is (dim_u * dim_v) x r, and b_w is a vector of length dim_u * dim_v.
    
    # Build A for a given U, V
    # A[(u,v), t] = U[u,t] * V[v,t]
    # This can be built as: A = reshape of (U[:, :, None] * V[None, :, :])? 
    # Actually, A is a matrix of size (dim_u * dim_v) x r.
    # A[(u * dim_v + v), t] = U[u, t] * V[v, t]
    
    # For each w, b_w[(u * dim_v + v)] = T[u, v, w]
    # T[u, v, w] = 1 iff (u, v) is in valid_pairs and valid_pairs[(u,v)] == w
    
    # Precompute b_w for all w
    b_w = np.zeros((dim_w, dim_u * dim_v))
    for (u, v), w in valid_pairs.items():
        b_w[w, u * dim_v + v] = 1.0
    
    # Search for lower ranks
    # Try ranks from 21 up to 41
    for r in range(21, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try random initializations for U and V
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
                
            # Random U and V with 0/1 entries
            U = np.random.randint(0, 2, size=(dim_u, r))
            V = np.random.randint(0, 2, size=(dim_v, r))
            
            # Build A: A[(u * dim_v + v), t] = U[u, t] * V[v, t]
            # A has shape (dim_u * dim_v, r)
            # We can build it efficiently:
            # For each t, A[:, t] = (U[:, t] as column) outer (V[:, t] as row) flattened
            # Actually, A[(u*dim_v + v), t] = U[u, t] * V[v, t]
            # So A[:, t] = np.kron(U[:, t], V[:, t])? No.
            # np.kron(U[:, t], V[:, t]) would give U[u,t]*V[v,t] for all u,v in order u=0..dim_u-1, v=0..dim_v-1
            # Which is exactly what we want since the index is u * dim_v + v.
            
            A = np.zeros((dim_u * dim_v, r))
            for t in range(r):
                A[:, t] = np.kron(U[:, t], V[:, t])
            
            # For each w, solve A @ W[w, :] = b_w[w, :]
            # Use least squares
            valid = True
            W = np.zeros((dim_w, r))
            for w in range(dim_w):
                try:
                    W[w, :], residuals, rank_sol, s = np.linalg.lstsq(A, b_w[w], rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if the solution is exact
            # Reconstruct T and compare
            exact = True
            for w in range(dim_w):
                # T_recon[:, :, w] = sum_t U[u,t] V[v,t] W[w,t]
                # = (U @ np.diag(W[w,:]) @ V.T)? No.
                # T_recon[u, v, w] = sum_t U[u,t] * V[v,t] * W[w,t]
                # = (U * W[w,:][None, :]) @ V.T? 
                # Let's compute: T_recon_w[u, v] = sum_t U[u,t] * W[w,t] * V[v,t]
                # = (U @ np.diag(W[w,:])) @ V.T? 
                # Actually, T_recon_w = (U * W[w,:][None, :]) @ V.T
                # Because (U * W[w,:][None, :])[u, t] = U[u,t] * W[w,t]
                # Then (U * W[w,:][None, :]) @ V.T has entry [u, v] = sum_t U[u,t]*W[w,t]*V[v,t]. Yes.
                
                T_recon_w = (U * W[w, :][None, :]) @ V.T
                T_recon_w_flat = T_recon_w.flatten()
                
                if not np.allclose(T_recon_w_flat, b_w[w], atol=1e-9):
                    exact = False
                    break
            
            if exact:
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break # Stop searching for this rank
        
        if best_rank < rank_school:
            break

    return best_result
