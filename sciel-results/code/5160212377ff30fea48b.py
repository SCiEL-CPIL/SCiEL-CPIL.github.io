"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def build_tensor_indices(n, m, p):
        """Return list of (i,j,k) tuples for the multiplication tensor."""
        return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i3 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i == i3 and j == j2 and k == k2:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k2][c] * W[k*n+i][c]
                                if abs(s) > tol:
                                    return False
        return True
    
    def rank25_decomposition(n, m, p):
        """Known rank-25 structured decomposition."""
        rank = m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for j0 in range(m):
            for k0 in range(p):
                c = j0 * p + k0
                for i in range(n):
                    for j in range(m):
                        if j == j0:
                            U[i * m + j][c] = 1
                V[j0 * p + k0][c] = 1
                for i in range(n):
                    W[k0 * n + i][c] = 1
        return U, V, W
    
    def to_numpy(U, V, W):
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)
    
    def from_numpy(U, V, W):
        return [[float(x) for x in row] for row in U], \
               [[float(x) for x in row] for row in V], \
               [[float(x) for x in row] for row in W]
    
    def objective(params, n, m, p, rank):
        """Compute sum of squared errors for tensor decomposition."""
        U_np = params[0:rank * (n*m)].reshape(rank, n*m)
        V_np = params[rank*(n*m):rank*(n*m)+rank*(m*p)].reshape(rank, m*p)
        W_np = params[rank*(n*m)+rank*(m*p):].reshape(rank, p*n)
        
        # Reshape to proper indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        U_mat = U_np.T  # (n*m, rank)
        V_mat = V_np.T  # (m*p, rank)
        W_mat = W_np.T  # (p*n, rank)
        
        total_err = 0.0
        # Diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                    total_err += (s - 1.0) ** 2
        # Off-diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i3 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i == i3 and j == j2 and k == k2:
                                    continue
                                s = np.dot(U_mat[i*m+j], V_mat[j2*p+k2] * W_mat[k*n+i3])
                                total_err += s ** 2
        return total_err
    
    def try_rank(rank, n, m, p, time_budget=30.0):
        """Try to find a valid decomposition of given rank via optimization."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize from rank-25 solution, truncate to desired rank
        U25, V25, W25 = rank25_decomposition(n, m, p)
        U25_np, V25_np, W25_np = to_numpy(U25, V25, W25)
        
        # For rank < 25, we can't simply truncate. Use random init mixed with structure.
        if rank >= 25:
            U_init = U25_np[:, :rank] if rank <= 25 else np.hstack([U25_np, np.zeros((n*m, rank-25))])
            V_init = V25_np[:, :rank] if rank <= 25 else np.hstack([V25_np, np.zeros((m*p, rank-25))])
            W_init = W25_np[:, :rank] if rank <= 25 else np.hstack([W25_np, np.zeros((p*n, rank-25))])
        else:
            # Try to find a lower-rank solution: use SVD-like compression or random init
            # Random initialization with small perturbation
            np.random.seed(42)
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
        
        # Flatten for optimization
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def obj(x):
            if time.time() > deadline - 3:
                return 1e10
            return objective(x, n, m, p, rank)
        
        # Try L-BFGS-B
        try:
            result = minimize(obj, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'maxfun': 1000, 'gtol': 1e-8})
        except Exception:
            return None
        
        if result.fun < 1e-8:
            U_opt = result.x[0:rank*(n*m)].reshape(rank, n*m).T
            V_opt = result.x[rank*(n*m):rank*(n*m)+rank*(m*p)].reshape(rank, m*p).T
            W_opt = result.x[rank*(n*m)+rank*(m*p):].reshape(rank, p*n).T
            
            # Round to half-integers
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            U_list, V_list, W_list = from_numpy(U_round, V_round, W_round)
            
            if verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return U_list, V_list, W_list
        
        return None
    
    # First, verify and return the known rank-25 solution
    U25, V25, W25 = rank25_decomposition(n, m, p)
    if verify(U25, V25, W25, n, m, p):
        best_result = {"U": U25, "V": V25, "W": W25}
        best_rank = 25
    else:
        # Fallback: schoolbook rank 50
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try to find lower rank solutions
    for target_rank in range(best_rank - 1, 14, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank, n, m, p, time_budget=20.0)
        if result is not None:
            U_found, V_found, W_found = result
            best_result = {"U": U_found, "V": V_found, "W": W_found}
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
