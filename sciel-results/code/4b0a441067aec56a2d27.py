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
    # T has 1s at (i*m+j, j*p+k, k*n+i)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_indices.append((i*m+j, j*p+k, k*n+i))
    nz_indices = np.array(nz_indices)
    nz_vals = np.ones(len(nz_indices))

    def tensor_error(params, rank):
        # params is a flat vector of size rank * (n*m + m*p + p*n)
        # Split into U, V, W
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:rank*size_u].reshape(rank, size_u).T
        V = params[rank*size_u:rank*(size_u+size_v)].reshape(rank, size_v).T
        W = params[rank*(size_u+size_v):].reshape(rank, size_w).T
        
        # Compute tensor sum: sum_r U[:,r] (x) V[:,r] (x) W[:,r]
        # Efficient computation using einsum
        # T_comp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is equivalent to np.einsum('ar,br,cr->abc', U, V, W)
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Calculate error only on non-zero entries to save time
        # T_comp at nz_indices should be 1
        # Other entries should be 0. 
        # Actually, checking all entries is safer for convergence, but checking nz is faster.
        # Let's check all entries but optimize the subtraction
        # T is sparse, so T_comp - T is T_comp where T=0, and T_comp-1 where T=1
        
        # A faster way:
        # err = 0
        # err += sum((T_comp[nz_indices] - 1)**2)
        # err += sum(T_comp[all_entries_not_in_nz]**2)
        # This is complex to index. Let's stick to full sum but maybe use a trick.
        # For small tensors, full sum is fine.
        # n*m = 6, m*p = 21, p*n = 14. Total size 6*21*14 = 1764. Very small.
        return np.sum((T_comp - T)**2)

    def check_and_round(U, V, W, rank):
        # Check if U, V, W entries are close to half-integers
        # And if the decomposition is exact
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness with rounded values
        T_comp = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_comp, T, atol=1e-9):
            # Ensure exact half-integer representation
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
        return None

    # First, return schoolbook if time is short or as fallback
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
    # 1. Try to find a decomposition with rank < 42.
    # 2. The tensor rank of 2x3x7 is likely around 30-40.
    # 3. We will use a "tensor slicing" inspired heuristic combined with optimization.
    #    However, direct optimization of rank is hard.
    #    Instead, we will try to find a decomposition for specific ranks by solving the system.
    
    # A more robust approach for small dimensions:
    # Try to find a basis of rank-1 tensors that span the target tensor.
    # This is equivalent to finding a low-rank factorization.
    
    # Let's try a grid search on a simplified parameter space? No, too large.
    # Let's try random initialization + L-BFGS-B as before, but with better initialization and more restarts.
    # The previous attempt failed likely due to poor initialization or local minima.
    
    # Improvement: Use a "structured" initialization based on schoolbook but with merged terms.
    # Or use a random Gaussian initialization with larger variance.
    
    # Let's try ranks from 41 down to 25.
    # For each rank, try multiple random restarts.
    
    num_restarts = 20
    max_iter = 200
    
    for rank in range(best_rank - 1, 25, -1):
        if time.time() > deadline - 15:
            break
            
        found = False
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize parameters
            np.random.seed(restart * 1000 + rank)
            size_u = n * m
            size_v = m * p
            size_w = p * n
            total_size = rank * (size_u + size_v + size_w)
            
            # Initialize with small noise
            params = np.random.normal(0, 1.0, total_size)
            
            # Try to optimize
            try:
                res = minimize(tensor_error, params, args=(rank,), method='L-BFGS-B', 
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
                
                # Check if error is small enough
                if res.fun < 1e-8:
                    # Extract U, V, W
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
            # Continue to try lower ranks
            pass

    # If no better rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
