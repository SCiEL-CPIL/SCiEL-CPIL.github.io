import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s margin

    n, m, p = DIMENSIONS
    NM, MP, PN = n * m, m * p, p * n
    rank_schoolbook = NM * MP // PN * PN # No, n*m*p
    rank_schoolbook = n * m * p  # 72

    # Build the correct matrix multiplication tensor T
    # T is of shape (NM, MP, PN)
    # T[a, b, c] = 1 if a=(i,j), b=(j',k), c=(i',k') and i=i', j=j', k=k', j=j'
    T = np.zeros((NM, MP, PN), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * p + k  # Wait, c index in W is k*n+i?
                # In the code, W is reshaped to (p, n, rank).
                # So W index is k*n + i.
                # Let's check the fallback:
                # W[k*n+i][col] = 1.
                # So c = k * n + i.
                c = k * n + i
                T[a, b, c] = 1.0

    T_flat = T.flatten()
    
    def check_exact(U, V, W, rank):
        """
        U: (NM, rank)
        V: (MP, rank)
        W: (PN, rank)
        """
        # Compute T_rec
        # T_rec[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can do this efficiently using einsum or matrix mult
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # But this creates a large array (12*24*18 = 5184).
        # 5184 is small.
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-8)

    def get_schoolbook_solution():
        rank = rank_schoolbook
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def try_find_lower_rank(rank, num_restarts=10):
        """
        Try to find a decomposition of given rank using gradient descent.
        """
        for seed in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(seed)
            # Initialize
            U = np.random.randn(NM, rank) * 0.1
            V = np.random.randn(MP, rank) * 0.1
            W = np.random.randn(PN, rank) * 0.1
            
            # Use L-BFGS-B
            try:
                from scipy.optimize import minimize
                
                def objective(x):
                    U_x = x[:NM*rank].reshape(NM, rank)
                    V_x = x[NM*rank:NM*rank+MP*rank].reshape(MP, rank)
                    W_x = x[NM*rank+MP*rank:].reshape(PN, rank)
                    T_rec = np.einsum('ar,br,cr->abc', U_x, V_x, W_x)
                    err = T_rec.flatten() - T_flat
                    return np.sum(err**2)
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                # Run optimization
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-15})
                
                if res.success and res.fun < 1e-8:
                    x_opt = res.x
                    U_opt = x_opt[:NM*rank].reshape(NM, rank)
                    V_opt = x_opt[NM*rank:NM*rank+MP*rank].reshape(MP, rank)
                    W_opt = x_opt[NM*rank+MP*rank:].reshape(PN, rank)
                    
                    # Round to integers/half-integers if possible
                    # Check exactness
                    if check_exact(U_opt, V_opt, W_opt, rank):
                        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                        
                    # Try rounding
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    if check_exact(U_round, V_round, W_round, rank):
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if check_exact(U_int, V_int, W_int, rank):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                pass
        
        return None

    # Search for lower rank
    best_result = None
    
    # Start from 71 down to 60 (just a few trials)
    # If rank 71 is not found quickly, fall back to 72.
    for rank in range(rank_schoolbook - 1, rank_schoolbook - 10, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_find_lower_rank(rank, num_restarts=5)
        if result is not None:
            best_result = result
            break

    if best_result is None:
        best_result = get_schoolbook_solution()

    return best_result
