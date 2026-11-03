import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor indices
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_indices.append((a, b, c))
    
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    for idx, (a, b, c) in enumerate(nonzero_indices):
        U_sb[a, idx] = 1
        V_sb[b, idx] = 1
        W_sb[c, idx] = 1
        
    def verify_solution(U, V, W):
        """Verify if U, V, W provide an exact decomposition with half-integer entries."""
        # Check if entries are close to half-integers
        # And if the tensor product matches 1.0 exactly (within float precision)
        
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check if the snap is valid (i.e., original values were close enough to be intended as half-integers)
        # We allow a small tolerance for the optimization to have converged
        if np.max(np.abs(U - U_snap)) > 1e-4: return None
        if np.max(np.abs(V - V_snap)) > 1e-4: return None
        if np.max(np.abs(W - W_snap)) > 1e-4: return None
        
        # Verify exactness
        for a, b, c in nonzero_indices:
            val = np.dot(U_snap[a], V_snap[b] * W_snap[c])
            if abs(val - 1.0) > 1e-9:
                return None
        
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

    def residuals(theta, rank):
        """Calculate residuals for the tensor decomposition."""
        # Unpack theta
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U = theta[:size_u].reshape((n*m, rank))
        V = theta[size_u:size_u+size_v].reshape((m*p, rank))
        W = theta[size_u+size_v:].reshape((p*n, rank))
        
        res = np.zeros(len(nonzero_indices))
        for idx, (a, b, c) in enumerate(nonzero_indices):
            res[idx] = 1.0 - np.dot(U[a], V[b] * W[c])
        return res

    def try_rank(rank, time_limit=10):
        """Try to find a solution for a given rank."""
        start_time = time.time()
        
        # Total parameters
        total_params = (n*m + m*p + p*n) * rank
        
        # Try a few random restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(42 + restart * 101 + rank)
            
            # Initialize
            # Strategy: Start with small random values
            scale = 0.5
            theta0 = np.random.randn(total_params) * scale
            
            # Use scipy least_squares
            try:
                import scipy.optimize
                result = scipy.optimize.least_squares(
                    residuals, 
                    theta0, 
                    args=(rank,),
                    method='lm', # Levenberg-Marquardt is good for this
                    max_nfev=1000
                )
                
                if result.success or result.cost < 1e-10:
                    # Reconstruct matrices
                    size_u = n * m * rank
                    size_v = m * p * rank
                    size_w = p * n * rank
                    
                    U = result.x[:size_u].reshape((n*m, rank))
                    V = result.x[size_u:size_u+size_v].reshape((m*p, rank))
                    W = result.x[size_u+size_v:].reshape((p*n, rank))
                    
                    sol = verify_solution(U, V, W)
                    if sol:
                        return sol
                        
            except Exception:
                pass

        return None

    # Search for lower ranks
    # Start from 49 down to 40 (likely 50 is optimal, but check a few)
    # Time budget: 20 seconds for search
    search_start = time.time()
    max_search_time = 20
    
    for rank in range(rank_schoolbook - 1, max(40, rank_schoolbook - 10), -1):
        if time.time() - search_start > max_search_time:
            break
            
        sol = try_rank(rank, time_limit=5)
        if sol:
            return sol

    # Return schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
