"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor entries: T[i,j,k] = 1 for valid (i,j,k)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Collect all (i,j,k) triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k))
    
    def error_fn(params, rank):
        """Compute the error for a given parameter vector and rank."""
        U = params[:rank * nm].reshape(rank, nm).T  # (nm, rank)
        V = params[rank * nm:2 * rank * nm].reshape(rank, mp).T  # (mp, rank)
        W = params[2 * rank * nm:3 * rank * nm].reshape(rank, pn).T  # (pn, rank)
        
        total_err = 0.0
        for (i, j, k) in triples:
            ui = U[i * m + j]
            vj = V[j * p + k]
            wk = W[k * n + i]
            val = np.dot(ui, vj * wk)  # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            total_err += (val - 1.0) ** 2
        return total_err
    
    def try_rank(rank, n_restarts=5, timeout_frac=0.3):
        """Try to find a valid decomposition at the given rank."""
        start_time = time.time()
        max_time = 120 * timeout_frac
        best_err = float('inf')
        best_params = None
        
        total_params = 3 * rank * nm if False else 3 * rank * max(nm, mp, pn)  # This is wrong, let me fix
        
        # Actually: U has nm*rank, V has mp*rank, W has pn*rank
        total_params = nm * rank + mp * rank + pn * rank
        
        for restart in range(n_restarts):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with random values
            params = np.random.randn(total_params) * 0.1
            
            # Use L-BFGS-B
            try:
                result = minimize(
                    error_fn, params, args=(rank,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.fun < best_err:
                    best_err = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        if best_params is not None and best_err < 1e-6:
            # Extract the matrices
            U = best_params[:nm * rank].reshape(nm, rank)
            V = best_params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
            W = best_params[nm * rank + mp * rank:].reshape(pn, rank)
            
            # Verify
            max_err = 0.0
            for (i, j, k) in triples:
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-6:
                return U, V, W
        
        return None
    
    # Try ranks from lower to higher
    # Lower bound: at least max(nm, mp, pn) but realistically higher
    # Start from a reasonable lower bound
    min_possible = max(nm, mp, pn)  # 30
    
    # Try a range of ranks
    for rank in range(min_possible, 91):
        result = try_rank(rank, n_restarts=3, timeout_frac=0.05)
        if result is not None:
            U, V, W = result
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
