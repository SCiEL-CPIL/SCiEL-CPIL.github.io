import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Target tensor T[i, l, k] = 1 for all valid indices
    # Dimensions: i in [0, n), l in [0, m), k in [0, p)
    T = np.ones((n, m, p))
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                U_school[i * m + l, col] = 1
                V_school[l * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Lower bound for rank is max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Upper bound is 54
    # We search for ranks from 27 to 53
    
    # To make the search more effective, we use a structured approach:
    # We know that for matrix multiplication, the rank is at least the
    # maximum of the dimensions of the factors.
    
    # Let's try a different strategy: 
    # We can exploit the fact that one dimension is small (n=2).
    # The multiplication can be viewed as a sum of rank-1 tensors.
    
    # For n=2, we can try to find a decomposition where the W factors
    # have a specific structure.
    
    # Actually, let's use the idea from inspiration 2: random initialization
    # with ALS, but with a better validation and rounding step.
    
    # We'll iterate over possible ranks from low to high.
    # For each rank, we'll try multiple random initializations.
    
    # To speed up the residual check, we'll use vectorized operations.
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly reconstructs the tensor."""
        # T_recon[i, l, k] = sum_c U[i*m+l, c] * V[l*p+k, c] * W[k*n+i, c]
        # We can compute this using einsum or loops.
        # Since n, m, p are small, loops are fine, but let's try to be efficient.
        
        T_recon = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            # T_recon += u_c[i,l] * v_c[l,k] * w_c[k,i]
            # This is a sum over c.
            # We can compute this as:
            # T_recon[i,l,k] = sum_c u_c[i,l] v_c[l,k] w_c[k,i]
            # This is equivalent to the trace of u_c[i,:] @ v_c[:,k] @ w_c[k,i]? No.
            # It's element-wise.
            
            # Efficient way:
            # For each i, l: u_c[i,l] is a scalar.
            # For each l, k: v_c[l,k] is a scalar.
            # For each k, i: w_c[k,i] is a scalar.
            
            # Let's just loop, it's small.
            for i in range(n):
                for l in range(m):
                    for k in range(p):
                        T_recon[i, l, k] += u_c[i, l] * v_c[l, k] * w_c[k, i]
                        
        return np.max(np.abs(T - T_recon))
    
    def try_rank(rank, max_attempts=10):
        """Try to find a valid decomposition for the given rank."""
        if time.time() > deadline:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
                
            np.random.seed(1234 + rank * 100 + attempt)
            
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline:
                    return None
                    
                # Update V
                V_new = np.zeros_like(V)
                for l in range(m):
                    for k in range(p):
                        M_vk = np.zeros((n, rank))
                        for i in range(n):
                            M_vk[i, :] = U[i * m + l, :] * W[k * n + i, :]
                        t_vk = np.ones(n)
                        try:
                            v_vk, _, _, _ = np.linalg.lstsq(M_vk, t_vk, rcond=None)
                            V_new[l * p + k, :] = v_vk
                        except:
                            pass
                V = V_new
                
                # Update U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for l in range(m):
                        M_uk = np.zeros((p, rank))
                        for k in range(p):
                            M_uk[k, :] = V[l * p + k, :] * W[k * n + i, :]
                        t_uk = np.ones(p)
                        try:
                            u_uk, _, _, _ = np.linalg.lstsq(M_uk, t_uk, rcond=None)
                            U_new[i * m + l, :] = u_uk
                        except:
                            pass
                U = U_new
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        M_wk = np.zeros((m, rank))
                        for l in range(m):
                            M_wk[l, :] = U[i * m + l, :] * V[l * p + k, :]
                        t_wk = np.ones(m)
                        try:
                            w_wk, _, _, _ = np.linalg.lstsq(M_wk, t_wk, rcond=None)
                            W_new[k * n + i, :] = w_wk
                        except:
                            pass
                W = W_new
                
            # Check residual
            residual = check_decomposition(U, V, W, rank)
            
            if residual < 1e-8:
                # Try to round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify exactness with rounded values
                residual_round = check_decomposition(U_round, V_round, W_round, rank)
                if residual_round < 1e-9:
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    
        return None
    
    # Search from lower bound to upper bound
    # Lower bound: max(n*m, m*p, p*n) = 27
    # But we can start a bit higher to save time, or start lower and go up.
    # Let's start from 27.
    
    for rank in range(27, 54):
        if time.time() > deadline:
            break
            
        result = try_rank(rank, max_attempts=5)
        if result is not None:
            return result
            
    return best_result
