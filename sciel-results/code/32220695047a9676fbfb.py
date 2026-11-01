"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 110.0  # Safety margin
    
    # Schoolbook decomposition as fallback (rank 36)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check if sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def check_half_integer(mat):
        """Check if all entries are close to half-integers (multiples of 0.5)."""
        doubled = mat * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=1e-4)

    # Try to find a lower rank decomposition
    # We search for ranks from 24 to 35
    for target_rank in range(24, 36):
        if time.time() > deadline:
            break
            
        # Number of variables: 9*r + 12*r + 12*r = 33*r
        num_vars = (n*m + m*p + p*n) * target_rank
        
        # Objective function
        def objective(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank:]
            
            U = U_flat.reshape((n*m, target_rank))
            V = V_flat.reshape((m*p, target_rank))
            W = W_flat.reshape((p*n, target_rank))
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (val - 1.0)**2
                        
            return error

        # Initial guess: random small values
        # We use multiple restarts for each rank
        best_result = None
        best_score = float('inf')
        
        # Limit restarts to stay within time
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with small random values, but maybe larger scale?
            # Tensor entries are often 1, -1, 0.5, etc.
            # Let's try initializing with random values in {-1, 0, 1} or similar
            # Actually, random normal with sigma=1 might be better than 0.1
            x0 = np.random.randn(num_vars)
            
            try:
                # Use L-BFGS-B with more iterations
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
                
                # Check if the raw error is low enough to potentially round
                if res.fun < 1e-4:
                    U_flat = res.x[:n*m*target_rank]
                    V_flat = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                    W_flat = res.x[n*m*target_rank + m*p*target_rank:]
                    
                    U = U_flat.reshape((n*m, target_rank))
                    V = V_flat.reshape((m*p, target_rank))
                    W = W_flat.reshape((p*n, target_rank))
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    # Verify the rounded solution
                    if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                        # Found a valid solution!
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        
            except Exception as e:
                pass
        
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
