import numpy as np
import time
import scipy.optimize

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # Leave 5s for cleanup/return
    
    # 1. Generate the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition
    def schoolbook_decomp():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook_decomp()
    best_rank = n * m * p
    
    # Try to find lower rank
    # We'll use a simple random search + local optimization for a few ranks.
    
    # Try ranks 47 down to 40
    for r in range(best_rank - 1, 39, -1):
        if time.time() - start_time > time_limit - 5:
            break
            
        found = False
        attempts = 10 # Increase attempts slightly
        
        for _ in range(attempts):
            # Random initialization with integers
            U = np.random.randint(-2, 3, (n * m, r))
            V = np.random.randint(-2, 3, (m * p, r))
            W = np.random.randint(-2, 3, (p * n, r))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_flat = params[:n*m*r].reshape(n*m, r)
                V_flat = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_flat = params[n*m*r+m*p*r:].reshape(p*n, r)
                T_approx = np.einsum('ac,bc,dc->abd', U_flat, V_flat, W_flat)
                return (T_approx - T).flatten()
                
            try:
                # Use 'lm' for small problems, it's fast
                res = scipy.optimize.least_squares(objective, x0, method='lm', max_nfev=200)
                if res.cost < 1e-10:
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Round to nearest integer or half-integer
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Verify rounded solution
                    T_check = np.einsum('ac,bc,dc->abd', U_round, V_round, W_round)
                    if np.allclose(T_check, T, atol=1e-9):
                        best_U, best_V, best_W = U_round, V_round, W_round
                        best_rank = r
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break # Found a lower rank, stop searching lower

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
