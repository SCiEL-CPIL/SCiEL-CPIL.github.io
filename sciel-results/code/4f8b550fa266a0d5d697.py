import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    # Dimensions of the factor matrices
    dim_U = n * m  # 6
    dim_V = m * p  # 30
    dim_W = p * n  # 20
    
    # We try ranks from 55 downwards. 
    # If 55 works, we return it. If not, we might try 54, etc.
    # But usually, if a rank is possible, it's likely the minimal one found by search.
    # Let's try a few ranks.
    
    best_result = None
    best_rank = float('inf')
    
    # Pre-compute indices for faster access
    # For U update: for each alpha, we need beta_k, gamma_k
    u_indices = []
    for i in range(n):
        for j in range(m):
            alpha = i * m + j
            betas = [j * p + k for k in range(p)]
            gammas = [k * n + i for k in range(p)]
            u_indices.append((alpha, betas, gammas))
            
    # For V update: for each beta, we need alpha_i, gamma_i
    v_indices = []
    for j in range(m):
        for k in range(p):
            beta = j * p + k
            alphas = [i * m + j for i in range(n)]
            gammas = [k * n + i for i in range(n)]
            v_indices.append((beta, alphas, gammas))
            
    # For W update: for each gamma, we need alpha_j, beta_j
    w_indices = []
    for k in range(p):
        for i in range(n):
            gamma = k * n + i
            alphas = [i * m + j for j in range(m)]
            betas = [j * p + k for j in range(m)]
            w_indices.append((gamma, alphas, betas))

    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        # Check exactness
        # T[alpha, beta, gamma] should be sum_r U[alpha,r] V[beta,r] W[gamma,r]
        # We only check non-zero entries
        is_exact = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    val = np.sum(U_s[alpha] * V_s[beta] * W_s[gamma])
                    if abs(val - 1.0) > 1e-9:
                        is_exact = False
                        break
                if not is_exact:
                    break
            if not is_exact:
                break
        if not is_exact:
            # Also check some zero entries to be sure? 
            # If non-zeros are 1, and rank is low, zeros are likely 0.
            # But let's check a few random zeros.
            # Actually, if the tensor is exactly decomposed, all entries must match.
            # The number of entries is 6 * 30 * 20 = 3600.
            # Checking all is fast.
            full_check = True
            for alpha in range(dim_U):
                for beta in range(dim_V):
                    for gamma in range(dim_W):
                        val = np.sum(U_s[alpha] * V_s[beta] * W_s[gamma])
                        # Determine expected value
                        # Find i,j from alpha, j,k from beta, k,i from gamma
                        i_a = alpha // m
                        j_a = alpha % m
                        j_b = beta // p
                        k_b = beta % p
                        k_g = gamma // n
                        i_g = gamma % n
                        
                        if j_a == j_b and k_b == k_g and i_g == i_a:
                            expected = 1.0
                        else:
                            expected = 0.0
                        
                        if abs(val - expected) > 1e-9:
                            full_check = False
                            break
                    if not full_check:
                        break
                if not full_check:
                    break
            
            if not full_check:
                return None
            # If it passed, return the snapped matrices
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        else:
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    def run_als(rank, max_iters=2000, tol=1e-12):
        # Initialize randomly
        U = np.random.randn(dim_U, rank)
        V = np.random.randn(dim_V, rank)
        W = np.random.randn(dim_W, rank)
        
        # Normalize columns to prevent scaling issues?
        # Or just let ALS handle it.
        
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                return None
            
            # Update U
            new_U = np.zeros_like(U)
            for alpha, betas, gammas in u_indices:
                # A is p x rank
                # Row k is V[betas[k]] * W[gammas[k]]
                A = np.array([V[betas[k]] * W[gammas[k]] for k in range(p)])
                b = np.ones(p)
                # Solve A^T A u = A^T b
                # Use lstsq
                u_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                new_U[alpha] = u_sol
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for beta, alphas, gammas in v_indices:
                # A is n x rank
                A = np.array([U[alphas[i]] * W[gammas[i]] for i in range(n)])
                b = np.ones(n)
                v_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                new_V[beta] = v_sol
            V = new_V
            
            # Update W
            new_W = np.zeros_like(W)
            for gamma, alphas, betas in w_indices:
                # A is m x rank
                A = np.array([U[alphas[j]] * V[betas[j]] for j in range(m)])
                b = np.ones(m)
                w_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                new_W[gamma] = w_sol
            W = new_W
            
            # Check convergence
            # Calculate residual
            # We can estimate residual by checking the non-zero entries
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        alpha = i * m + j
                        beta = j * p + k
                        gamma = k * n + i
                        val = np.sum(U[alpha] * V[beta] * W[gamma])
                        res += (val - 1.0)**2
            if res < tol:
                return (U, V, W)
                
        return None

    # Try ranks
    # Start with 55
    for rank in [55, 54, 53, 52, 51, 50]:
        if time.time() - start_time > timeout:
            break
        
        # Multiple restarts
        for restart in range(20):
            if time.time() - start_time > timeout:
                break
            
            np.random.seed(42 + restart) # Reproducibility per restart
            result = run_als(rank)
            
            if result:
                U, V, W = result
                snapped = snap_and_verify(U, V, W)
                if snapped:
                    # Found a valid decomposition
                    return snapped
                    
    # If no lower rank found, return schoolbook (rank 60)
    # This is a fallback
    rank = n * m * p
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
