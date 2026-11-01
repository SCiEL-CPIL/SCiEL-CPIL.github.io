import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] corresponds to A[i, j] * B[j, k] * C[k, i] (trace convention W[k*n+i])
    # Actually, the tensor is T[i, j, k] = 1 if this term exists in matmul.
    # C[k, i] = sum_j A[i, j] B[j, k].
    # So T[i, j, k] = 1 for all i, j, k.
    # Wait, the tensor for matrix multiplication is T = sum_{i,j,k} e_{ij} (A) * e_{jk} (B) * e_{ki} (C).
    # So T[i, j, k] = 1 for all i, j, k in {0, 1, 2}.
    # Let's verify.
    # C_{ki} = sum_j A_{ij} B_{jk}.
    # Yes, every combination (i, j, k) appears exactly once in the sum.
    # So the target tensor is all ones?
    # No. The tensor is a multilinear form.
    # <T, A, B, C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i].
    # For matrix multiplication, we want <T, A, B, C> = sum_{i,j,k} A[i,j] B[j,k] C[k,i].
    # So T[i,j,k] = 1 for all i,j,k.
    
    T = np.ones((n, m, p))
    
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def flatten_idx(i, j, k):
        return (i * m + j, j * p + k, k * n + i)
        
    # Objective function
    def objective(params):
        # params is flattened U, V, W
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        # Compute tensor product sum
        # T_calc[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can vectorize this.
        # Reshape U, V, W to allow broadcasting?
        # U is (9, 23), V is (9, 23), W is (9, 23).
        # We need to sum over r for each (i, j, k).
        # This is a contraction.
        
        # Efficient way:
        # U_term = U[:, :, None] # (9, 23, 1)
        # V_term = V[:, :, None] # (9, 23, 1)
        # W_term = W[:, :, None] # (9, 23, 1)
        # This doesn't align indices easily.
        
        # Let's use np.einsum
        # U_idx = i*m+j
        # V_idx = j*p+k
        # W_idx = k*n+i
        # We need T_calc[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # Create index arrays
        i_idx, j_idx, k_idx = np.indices((n, m, p))
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        # Gather
        U_gathered = U_flat[u_idx, :] # (3, 3, 3, 23)
        V_gathered = V_flat[v_idx, :] # (3, 3, 3, 23)
        W_gathered = W_flat[w_idx, :] # (3, 3, 3, 23)
        
        # Sum over rank
        T_calc = np.sum(U_gathered * V_gathered * W_gathered, axis=3)
        
        # Residual
        residual = T_calc - T
        return residual.flatten()
        
    # Initial guess: Schoolbook (27 terms) truncated to 23?
    # Better: Random initialization with some structure.
    # Or use the 27 schoolbook terms and set 4 to zero, then optimize.
    
    best_params = None
    best_cost = np.inf
    
    # Try a few restarts
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W
        # Start with schoolbook terms
        U_init = np.zeros((n*m, rank))
        V_init = np.zeros((m*p, rank))
        W_init = np.zeros((p*n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U_init[i*m+j, col] = 1
                        V_init[j*p+k, col] = 1
                        W_init[k*n+i, col] = 1
                        col += 1
                    else:
                        break
        
        # Add random noise to the last few columns or all columns to break symmetry
        # Actually, just add small random noise to all
        U_init += np.random.normal(0, 0.1, U_init.shape)
        V_init += np.random.normal(0, 0.1, V_init.shape)
        W_init += np.random.normal(0, 0.1, W_init.shape)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(objective, x0, method='trf', max_nfev=1000)
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_params = result.x
                if cost < 1e-12:
                    break
        except Exception:
            continue
            
    if best_params is None:
        # Fallback to schoolbook if optimization fails (score 1.0)
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    # Extract best params
    U_best = best_params[:n*m*rank].reshape(n*m, rank)
    V_best = best_params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W_best = best_params[n*m*rank + m*p*rank :].reshape(p*n, rank)
    
    # Convert to lists of lists
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
