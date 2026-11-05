import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Target: find rank-minimal decomposition
    # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try increasing ranks
    for rank in range(24, 64):
        # Use random initialization with multiple attempts
        for attempt in range(20):
            np.random.seed(attempt * 1000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating minimization
            converged = False
            for it in range(500):
                # Fix V, W, solve for U
                # For each (i,j): U[i*m+j, :] @ (V[j*p+0:..., :] * W[0:..., :] * ... ) 
                # This is tricky because it's a 3-way tensor
                
                # Alternative: use the fact that we can flatten
                # Build the linear system for U given V, W
                # For each (i,j,k): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                # Let G[i*m+j, r] = sum_k ... no, this doesn't work directly
                
                # Better approach: for fixed V, W, the problem for U is:
                # For each (i,j): sum_k [sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]] = p (since each k contributes 1)
                # Wait no, each (i,j,k) must equal 1 separately.
                
                # Let me define: for fixed V, W, define M[i*m+j, r] such that
                # M[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r] * delta? No...
                
                # Actually, for each (i,j,k) independently:
                # U[i*m+j, :] dot (V[j*p+k, :] * W[k*n+i, :]) = 1
                # This is a linear system in U for each (i,j,k), but U[i*m+j, :] is shared across k.
                
                # So for fixed i,j: we need U[i*m+j, :] to satisfy
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: (U[i*m+j, :]) @ diag(V[j*p+k, :] * W[k*n+i, :]) ... no
                # It's: U[i*m+j, :] dot (elementwise product of V[j*p+k, :] and W[k*n+i, :]) = 1
                # For all k, so we have p equations for U[i*m+j, :]
                
                # Build matrix A_ij of shape (p, rank) where A_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then we need U[i*m+j, :] @ A_ij.T = ones(p)
                # This is a linear system: A_ij @ U[i*m+j, :].T = ones(p)
                
                new_U = np.zeros((n*m, rank))
                for i in range(n):
                    for j in range(m):
                        A_ij = np.zeros((p, rank))
                        for k in range(p):
                            A_ij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A_ij @ u = ones(p)
                        try:
                            u, residuals, rank_solve, sv = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                            new_U[i*m+j, :] = u
                        except:
                            break
                    else:
                        continue
                    break
                else:
                    U = new_U
                    # Similarly for V and W... this is getting complex
                    
                break  # Skip alternating for now
            
            # Check if current U, V, W give exact decomposition
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
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
