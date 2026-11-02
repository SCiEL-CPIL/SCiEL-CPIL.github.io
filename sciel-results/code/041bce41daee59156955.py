import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower-rank decomposition
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 47 down to 40
    for rank in range(rank_schoolbook - 1, 39, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook terms + random perturbations
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Use first rank schoolbook terms as a starting point
            for c in range(rank):
                U[:, c] = U_school[:, c]
                V[:, c] = V_school[:, c]
                W[:, c] = W_school[:, c]
            
            # Add small random perturbations
            U += np.random.randn(n*m, rank) * 0.1
            V += np.random.randn(m*p, rank) * 0.1
            W += np.random.randn(p*n, rank) * 0.1
            
            # Try to optimize: minimize the error
            # Use a simple gradient-free method
            error = None
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Compute error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-9:
                    error = 0.0
                    break
                
                if error is None or max_err < error:
                    error = max_err
                
                # Simple perturbation
                U += np.random.randn(n*m, rank) * 0.01
                V += np.random.randn(m*p, rank) * 0.01
                W += np.random.randn(p*n, rank) * 0.01
            
            if error is not None and error < 1e-9:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition for this rank
    
    # Convert to lists with integer/half-integer entries if possible
    def to_half_int(x):
        """Round to nearest half-integer."""
        return round(2*x) / 2.0
    
    U_list = [[to_half_int(best_U[i,j]) for j in range(best_U.shape[1])] for i in range(best_U.shape[0])]
    V_list = [[to_half_int(best_V[i,j]) for j in range(best_V.shape[1])] for i in range(best_V.shape[0])]
    W_list = [[to_half_int(best_W[i,j]) for j in range(best_W.shape[1])] for i in range(best_W.shape[0])]
    
    # Verify after rounding
    if not check_decomposition(np.array(U_list), np.array(V_list), np.array(W_list)):
        # Fallback to schoolbook
        U_list = [[int(U_school[i,j]) for j in range(rank_schoolbook)] for i in range(n*m)]
        V_list = [[int(V_school[i,j]) for j in range(rank_schoolbook)] for i in range(m*p)]
        W_list = [[int(W_school[i,j]) for j in range(rank_schoolbook)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
