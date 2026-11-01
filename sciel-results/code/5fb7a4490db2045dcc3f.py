"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses Strassen per pair of columns as baseline (rank 42), then attempts
    to reduce rank by finding shared structure via optimization.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    nm = n * m  # 4
    mp = m * p  # 24
    pn = p * n  # 24
    
    def build_strassen_decomposition(rank):
        """Build the standard Strassen-per-pair decomposition with given rank."""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        num_pairs = p // 2
        if rank < num_pairs * 7:
            return None  # Can't fit
        
        strassen_U = [
            [(0, 1), (3, 1)],
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (0, -1)],
            [(1, 1), (3, -1)]
        ]
        strassen_V = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)]
        ]
        strassen_W = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(0, 0, -1), (1, 0, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)]
        ]
        
        for t in range(num_pairs):
            base_c = t * 7
            k0 = 2 * t
            for s in range(7):
                c = base_c + s
                for u_idx, coeff in strassen_U[s]:
                    U[u_idx, c] += coeff
                for j, k_loc, coeff in strassen_V[s]:
                    global_k = k0 + k_loc
                    v_idx = j * p + global_k
                    V[v_idx, c] += coeff
                for k_loc, i, coeff in strassen_W[s]:
                    global_k = k0 + k_loc
                    w_idx = global_k * n + i
                    W[w_idx, c] += coeff
        
        return U, V, W
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        # Compute tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should equal delta_{j,k mod 2 == j?} No, should equal 1 when j matches the column pair
        # Actually T[i,j,k] should be 1 if the (i,j) entry of A times (j,k) entry of B contributes
        # The multiplication tensor: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the identity in the j dimension for each (i,k) pair)
        # Wait, let me re-read: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
        # for all i in [0,n), j in [0,m), k in [0,p).
        
        # Let's verify: for each (i,j,k), compute the sum
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def tensor_error(U, V, W):
        """Compute the squared error of the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0) ** 2
        return err
    
    # Start with rank 42 (Strassen per pair)
    best_rank = 42
    best_U, best_V, best_W = build_strassen_decomposition(42)
    
    # Try to reduce rank by attempting rank 36, 35, 34, ...
    # For each candidate rank, try to find a valid decomposition
    for candidate_rank in range(36, 42):
        if time.time() > deadline - 30:
            break
        
        # Try optimization from the Strassen decomposition projected to this rank
        # This is tricky - let's try a different approach: 
        # Use the fact that we can try to merge some rank terms
        
        # Alternative: try a structured decomposition with rank < 42
        # For 2x2x12, we can think of it as 6 copies of 2x2x2.
        # The minimal rank of 2x2x2 is 7, so 6*7=42 is an upper bound.
        # But can we do better by sharing?
        
        # Try: use 7 terms for the first 5 pairs (35 terms) and try to handle
        # the 6th pair with fewer terms by sharing with others.
        
        # Actually, let's try a direct optimization approach for rank 36.
        # We'll parameterize U, V, W and minimize the tensor error.
        
        if time.time() > deadline - 20:
            break
        
        # For rank 36, we need to find U (4x36), V (24x36), W (24x36)
        # such that the tensor equation holds.
        
        # Initialize with the Strassen decomposition truncated/modified
        U_init = np.zeros((nm, candidate_rank))
        V_init = np.zeros((mp, candidate_rank))
        W_init = np.zeros((pn, candidate_rank))
        
        # Fill first 5 pairs with Strassen (35 terms)
        strassen_U = [
            [(0, 1), (3, 1)],
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (0, -1)],
            [(1, 1), (3, -1)]
        ]
        strassen_V = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)]
        ]
        strassen_W = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(0, 0, -1), (1, 0, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)]
        ]
        
        num_pairs_full = min(5, p // 2)  # 5 full pairs = 35 terms
        for t in range(num_pairs_full):
            base_c = t * 7
            k0 = 2 * t
            for s in range(7):
                c = base_c + s
                if c >= candidate_rank:
                    break
                for u_idx, coeff in strassen_U[s]:
                    U_init[u_idx, c] += coeff
                for j, k_loc, coeff in strassen_V[s]:
                    global_k = k0 + k_loc
                    v_idx = j * p + global_k
                    V_init[v_idx, c] += coeff
                for k_loc, i, coeff in strassen_W[s]:
                    global_k = k0 + k_loc
                    w_idx = global_k * n + i
                    W_init[w_idx, c] += coeff
        
        # The 36th term (index 35) is free - initialize with small random values
        np.random.seed(42)
        U_init[:, 35] = np.random.randn(nm) * 0.1
        V_init[:, 35] = np.random.randn(mp) * 0.1
        W_init[:, 35] = np.random.randn(pn) * 0.1
        
        # Define the optimization variables
        def flatten_params(x):
            U_f = x[:nm * candidate_rank].reshape(nm, candidate_rank)
            V_f = x[nm * candidate_rank: nm * candidate_rank + mp * candidate_rank].reshape(mp, candidate_rank)
            W_f = x[nm * candidate_rank + mp * candidate_rank:].reshape(pn, candidate_rank)
            return U_f, V_f, W_f
        
        def objective(x):
            U_f, V_f, W_f = flatten_params(x)
            return tensor_error(U_f, V_f, W_f)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Try L-BFGS-B optimization
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50})
            
            if result.fun < 1e-8:
                U_opt, V_opt, W_opt = flatten_params(result.x)
                # Verify
                if check_decomposition(U_opt, V_opt, W_opt, tol=1e-8):
                    best_rank = candidate_rank
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    break
        except Exception:
            pass
    
    if time.time() > deadline - 5:
        # Return best found so far
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
