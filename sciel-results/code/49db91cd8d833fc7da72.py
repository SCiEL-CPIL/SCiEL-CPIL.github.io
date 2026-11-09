import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=3 rows of A, m=4 inner, p=6 cols of B
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    
    # Precompute M_full indices
    # M_full[i*m+j, j*p+k, k*n+i] = 1.0
    # We will build M_full once
    M_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M_full[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Search for lower rank
    # Try ranks from 72 down to 50
    for rank in range(72, 49, -1):
        if time.time() > deadline - 20:
            break
            
        found = False
        # More restarts for lower ranks? Or more for higher?
        # Higher ranks are easier. Start with 72.
        num_restarts = 20 if rank > 65 else 50
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(rank * 1000 + restart)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Normalize to avoid overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            converged = False
            for iteration in range(200): # Reduced iterations per restart to allow more restarts
                if time.time() > deadline - 5:
                    break
                
                # ALS Step
                # Update U
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                N_U = np.einsum('jkc,kic->ijc', V_r, W_r)
                VtV = V.T @ V
                WtW = W.T @ W
                G_U = VtV * WtW
                
                try:
                    N_U_flat = N_U.reshape(n*m, rank)
                    U_new = np.linalg.solve(G_U, N_U_flat.T).T
                except:
                    G_U_reg = G_U + np.eye(rank) * 1e-6
                    try:
                        U_new = np.linalg.solve(G_U_reg, N_U_flat.T).T
                    except:
                        break
                U = U_new
                
                # Update V
                U_r = U.reshape(n, m, rank)
                W_r = W.reshape(p, n, rank)
                N_V = np.einsum('ijc,kic->jk c', U_r, W_r)
                UtU = U.T @ U
                WtW = W.T @ W
                G_V = UtU * WtW
                
                try:
                    N_V_flat = N_V.reshape(m*p, rank)
                    V_new = np.linalg.solve(G_V, N_V_flat.T).T
                except:
                    G_V_reg = G_V + np.eye(rank) * 1e-6
                    try:
                        V_new = np.linalg.solve(G_V_reg, N_V_flat.T).T
                    except:
                        break
                V = V_new
                
                # Update W
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                N_W = np.einsum('ijc,jkc->ikc', U_r, V_r)
                UtU = U.T @ U
                VtV = V.T @ V
                G_W = UtU * VtV
                
                try:
                    N_W_flat = N_W.reshape(p*n, rank)
                    W_new = np.linalg.solve(G_W, N_W_flat.T).T
                except:
                    G_W_reg = G_W + np.eye(rank) * 1e-6
                    try:
                        W_new = np.linalg.solve(G_W_reg, N_W_flat.T).T
                    except:
                        break
                W = W_new
                
                # Check residual
                T_full = np.zeros((n*m, m*p, p*n))
                for c in range(rank):
                    T_full += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                
                err = np.max(np.abs(T_full - M_full))
                
                if err < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Try to round to integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                T_int = np.zeros((n*m, m*p, p*n))
                for c in range(rank):
                    T_int += np.einsum('i,j,k->ijk', U_int[:,c], V_int[:,c], W_int[:,c])
                
                err_int = np.max(np.abs(T_int - M_full))
                
                if err_int < 1e-8:
                    # Found integer solution
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                else:
                    # Found real solution, but not integer
                    # We might return this if we are desperate, but it might be invalid.
                    # For now, continue searching for integer solution.
                    # If we find no integer solution, we fall back to schoolbook.
                    pass
        
        # If we found a real solution but not integer, we could try to polish it further
        # but for now, we just continue to next rank or restart.
        
    # Fallback to schoolbook
    return get_schoolbook()
