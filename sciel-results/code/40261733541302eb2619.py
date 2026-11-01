"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook as fallback
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
    
    def verify(U, V, W, tol=1e-9):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    target = 1.0 if i == k else 0.0
                    if abs(val - target) > tol:
                        return False
        return True
    
    # Try to find rank 17 decomposition with structured approach
    # The tensor T[i,j,k] = delta_{i,k}
    # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,k}
    
    # For rank 17, we can try to find a solution where most entries are 0 or 1
    # with a few free parameters
    
    # Let's try a different approach: use the known fact that for 2x2x2, rank 7 works (Strassen)
    # For 2x3x3, maybe we can extend this idea
    
    # Actually, let me try a numerical search with better initialization
    # using the schoolbook solution as a starting point and trying to merge columns
    
    # Start with schoolbook rank 18
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U0[i * m + j, c] = 1
                V0[j * p + k, c] = 1
                W0[k * n + i, c] = 1
    
    # Try to find a rank-17 solution by setting up equations
    # This is a system of 18 equations (for each (i,j,k)) with 3*17*r variables
    # Too many variables, so let's try a different approach
    
    # Use randomized low-rank factorization search
    for target_rank in [17, 16, 15]:
        if time.time() > deadline - 30:
            break
        
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with random factors
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Use alternating least squares or similar
            # For each factor, solve a linear system
            for iteration in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), solve for U[i,j,:]
                for i in range(n):
                    for j in range(m):
                        # U[i,j,:] * sum_k V[j,k,:]^T * W[k,i,:] = delta_{i,k} for all k
                        # This is: U[i,j,:] * M = b where M is p x target_rank, b is p-vector
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            for c in range(target_rank):
                                M[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        b = np.array([1.0 if i == k else 0.0 for k in range(p)])
                        
                        # Solve U[i,j,:] * M = b => M^T * U[i,j,:]^T = b
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(M.T, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j,k,:]
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                M[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        b = np.array([1.0 if i == k else 0.0 for i in range(n)])
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(M.T, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k,i,:]
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                M[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        b = np.array([1.0 if i == k else 0.0 for _ in range(m)])
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(M.T, b, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                            target = 1.0 if i == k else 0.0
                            err += (val - target) ** 2
                
                if err < 1e-14:
                    # Found a solution, try to make it rational
                    U_r = np.round(U * 2) / 2.0
                    V_r = np.round(V * 2) / 2.0
                    W_r = np.round(W * 2) / 2.0
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    # Also try integer rounding
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if verify(U_i.tolist(), V_i.tolist(), W_i.tolist()):
                        return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
    
    return schoolbook()

# EVOLVE-BLOCK-END
