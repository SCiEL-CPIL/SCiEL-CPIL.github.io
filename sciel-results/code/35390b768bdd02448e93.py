import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matmul tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(r))
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        # Try multiple random initializations
        best = None
        for trial in range(50):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Iterative optimization
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Compute residuals
                # For each (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should = 1
                # Update W: W[k*n+i, r] += lr * (1 - sum_r' U[i*m+j,r'] V[j*p+k,r'] W[k*n+i,r']) * U[i*m+j,r] * V[j*p+k,r]
                
                # Simplified: use SVD-based approach
                # Reshape and do alternating least squares
                
                # For fixed U, V: solve for W
                # For each (k,i): W[k*n+i, :] should satisfy
                # sum_j U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :] = 1 for all j
                # This is: (sum_j U[i*m+j, r] V[j*p+k, r]) * W[k*n+i, r] = 1 (element-wise product summed)
                # Actually: for each r, the contribution is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # Sum over r: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all j
                
                # This is a linear system in W for fixed U,V
                # Let's build the system:
                # For each (i,k): sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) ... no
                # Actually the constraint is for each (i,j,k):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                
                # For fixed U, V, this is linear in W:
                # For each (i,j,k): sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                
                # Group by (k,i): for each (k,i), we have m equations (one for each j)
                # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for j=0..m-1
                
                # This is a system: A_{(k,i)} @ W[k*n+i, :] = 1_m
                # where A_{(k,i)}[j, r] = U[i*m+j, r] * V[j*p+k, r]
                
                # Solve for W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        # Solve A @ W[k*n+i, :] = ones(m)
                        try:
                            W_new[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
                W = W_new
                
                # Similarly update U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U_new[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                U = U_new
                
                # Similarly update V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V_new[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
                V = V_new
                
                # Check convergence
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(r))
                            err = max(err, abs(s - 1))
                if err < 1e-8:
                    break
            
            # Check if solution is valid
            if verify(U, V, W):
                # Round to integers or half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                if verify(U_round, V_round, W_round):
                    return (U_round, V_round, W_round)
                # Try rounding to integers
                U_round = np.round(U)
                V_round = np.round(V)
                W_round = np.round(W)
                if verify(U_round, V_round, W_round):
                    return (U_round, V_round, W_round)
        
        return None
    
    # Try ranks from low to high
    for r in range(10, 41):
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
