import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Index mappings
    # U: (n*m, rank) -> index i*m + j
    # V: (m*p, rank) -> index j*p + k
    # W: (p*n, rank) -> index k*n + i  (as per prompt/seed)
    
    # Target Tensor T (dense)
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
                
    # Precompute list of non-zero tensor entries for faster error calculation
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_indices.append((i*m+j, j*p+k, k*n+i))
    nz_indices = np.array(nz_indices)
    
    def tensor_error(params, rank):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:rank*size_u].reshape(rank, size_u).T
        V = params[rank*size_u:rank*(size_u+size_v)].reshape(rank, size_v).T
        W = params[rank*(size_u+size_v):].reshape(rank, size_w).T
        
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Calculate error only on non-zero entries to save time
        # T_comp at nz_indices should be 1
        # Other entries should be 0.
        # For small tensors, full sum is fine and safer.
        return np.sum((T_comp - T)**2)

    def check_and_round(U, V, W, rank):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_comp = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_comp, T, atol=1e-9):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
        return None

    def get_schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    schoolbook_U, schoolbook_V, schoolbook_W = get_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook_U, schoolbook_V, schoolbook_W

    # Strategy:
    # Focus on ranks 41, 40, 39.
    # Use more restarts for lower ranks.
    
    ranks_to_try = [41, 40, 39]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Increase restarts for rank 39
        num_restarts = 50 if rank == 39 else 30
        max_iter = 150
        
        found = False
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart * 1000 + rank)
            size_u = n * m
            size_v = m * p
            size_w = p * n
            total_size = rank * (size_u + size_v + size_w)
            
            # Initialize with small noise
            params = np.random.normal(0, 1.0, total_size)
            
            try:
                res = minimize(tensor_error, params, args=(rank,), method='L-BFGS-B', 
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    U = res.x[:rank*size_u].reshape(rank, size_u).T
                    V = res.x[rank*size_u:rank*(size_u+size_v)].reshape(rank, size_v).T
                    W = res.x[rank*(size_u+size_v):].reshape(rank, size_w).T
                    
                    result = check_and_round(U, V, W, rank)
                    if result:
                        U_r, V_r, W_r = result
                        best_rank = rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        found = True
                        break
            except Exception as e:
                pass
        
        if found:
            # Found a lower rank, continue to try even lower?
            # We are iterating 41, 40, 39. If 41 found, try 40, then 39.
            # If 39 found, we stop after loop.
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
