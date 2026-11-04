import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the tensor."""
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=tol, rtol=0.0):
            return False
        return True

    def solve_for_W(U, V):
        """Given U and V, solve for W using ALS."""
        R = U.shape[1]
        W = np.zeros((p * n, R))
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                w, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W[k*n+i, :] = w
        
        return W

    def solve_for_U(V, W):
        """Given V and W, solve for U using ALS."""
        R = V.shape[1]
        U = np.zeros((n * m, R))
        
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                
                u, residuals, rank, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i*m+j, :] = u
        
        return U

    def solve_for_V(U, W):
        """Given U and W, solve for V using ALS."""
        R = U.shape[1]
        V = np.zeros((m * p, R))
        
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                
                v, residuals, rank, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V[j*p+k, :] = v
        
        return V

    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    # Try ranks from low to high
    for rank in range(20, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Update W
                W_new = solve_for_W(U, V)
                # Update U
                U_new = solve_for_U(V, W_new)
                # Update V
                V_new = solve_for_V(U_new, W_new)
                
                # Check convergence
                U_3d = U_new.reshape(n, m, rank)
                V_3d = V_new.reshape(m, p, rank)
                W_3d = W_new.reshape(p, n, rank)
                T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                err = np.max(np.abs(T - 1.0))
                
                if err < 1e-8:
                    best_rank = rank
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    break
                
                # Update for next iteration
                U, V, W = U_new, V_new, W_new
                
            if best_rank == rank:
                break
        
        if best_rank == rank:
            break
    
    # Return the best found
    if best_U is not None:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        
        if check_decomposition(np.array(U_out), np.array(V_out), np.array(W_out)):
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    U_out = U_sb.tolist()
    V_out = V_sb.tolist()
    W_out = W_sb.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
