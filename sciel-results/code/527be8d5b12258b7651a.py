"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)  
    # W: (p*n, rank) = (6, rank)
    
    def tensor_residual(U, V, W):
        """Compute residual efficiently using matrix operations."""
        rank = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    total += (s - 1.0) ** 2
        return total
    
    def is_half_integer(arr, tol=1e-9):
        """Check if all entries are integer or half-integer."""
        flat = arr.flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)
    
    def round_to_half(arr):
        return np.round(arr * 2) / 2
    
    def verify_solution(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact and has half-integer entries."""
        if not (is_half_integer(U, 1e-8) and is_half_integer(V, 1e-8) and is_half_integer(W, 1e-8)):
            return False
        res = tensor_residual(U, V, W)
        return res < tol
    
    # Schoolbook as fallback
    def schoolbook():
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
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def try_rank(target_rank, max_attempts=10):
        """Try to find a rank-minimal decomposition for given target_rank."""
        nonlocal best, best_rank
        if target_rank >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                return False
            
            # Different initialization strategies
            if attempt % 4 == 0:
                # Sparse random initialization
                np.random.seed(42 + target_rank * 1000 + attempt)
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                # Place random non-zero entries
                for r in range(target_rank):
                    for idx in range(np.random.randint(1, n*m+1)):
                        U[np.random.randint(n*m), r] = np.random.choice([-1, 1]) / 2
                    for idx in range(np.random.randint(1, m*p+1)):
                        V[np.random.randint(m*p), r] = np.random.choice([-1, 1]) / 2
                    for idx in range(np.random.randint(1, p*n+1)):
                        W[np.random.randint(p*n), r] = np.random.choice([-1, 1]) / 2
            elif attempt % 4 == 1:
                # Dense random half-integer
                np.random.seed(1000 + target_rank * 100 + attempt)
                U = np.random.randint(-2, 3, size=(n*m, target_rank)) / 2.0
                V = np.random.randint(-2, 3, size=(m*p, target_rank)) / 2.0
                W = np.random.randint(-1, 2, size=(p*n, target_rank)) / 2.0
            elif attempt % 4 == 2:
                # Structured: try to exploit symmetry
                np.random.seed(2000 + target_rank * 50 + attempt)
                U = np.random.randint(-1, 2, size=(n*m, target_rank)) / 2.0
                V = np.random.randint(-1, 2, size=(m*p, target_rank)) / 2.0
                W = np.random.randint(-1, 2, size=(p*n, target_rank)) / 2.0
            else:
                # Try known structured patterns for small matrices
                np.random.seed(3000 + target_rank * 25 + attempt)
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                # For 2x3*3x3, try rank-15 or lower structured solutions
                if target_rank <= 15:
                    # Try to build from schoolbook with some zeros
                    base_rank = min(target_rank, 18)
                    for r in range(base_rank):
                        i = r // (m*p)
                        rem = r % (m*p)
                        j = rem // p
                        k = rem % p
                        if i < n and j < m and k < p:
                            U[i*m+j, r] = 1.0
                            V[j*p+k, r] = 1.0
                            W[k*n+i, r] = 1.0
                    
                    # Add some random perturbation
                    if target_rank > base_rank:
                        for r in range(base_rank, target_rank):
                            U[np.random.randint(n*m), r] = np.random.choice([-1, 0, 1]) / 2
                            V[np.random.randint(m*p), r] = np.random.choice([-1, 0, 1]) / 2
                            W[np.random.randint(p*n), r] = np.random.choice([-1, 0, 1]) / 2
                else:
                    U = np.random.randint(-1, 2, size=(n*m, target_rank)) / 2.0
                    V = np.random.randint(-1, 2, size=(m*p, target_rank)) / 2.0
                    W = np.random.randint(-1, 2, size=(p*n, target_rank)) / 2.0
            
            # Quick check before expensive optimization
            res = tensor_residual(U, V, W)
            if res < 1e-4:
                if verify_solution(U, V, W, 1e-6):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    return True
                # Try rounding
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                if verify_solution(U_r, V_r, W_r, 1e-6):
                    best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = target_rank
                    return True
            
            # Use scipy optimization if not too close already
            if res < 10.0:  # Only optimize if reasonably close
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def obj(x):
                    U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_flat = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    return tensor_residual(U_flat, V_flat, W_flat)
                
                # Time-limited optimization
                max_iter = 500 if target_rank > 14 else 300
                res_opt = minimize(obj, x0, method='Nelder-Mead', 
                                  options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
                
                if res_opt.fun < 1e-8:
                    U_sol = res_opt.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = res_opt.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = res_opt.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    if verify_solution(U_sol, V_sol, W_sol, 1e-6):
                        best = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                        best_rank = target_rank
                        return True
                    else:
                        U_r = round_to_half(U_sol)
                        V_r = round_to_half(V_sol)
                        W_r = round_to_half(W_sol)
                        if verify_solution(U_r, V_r, W_r, 1e-6):
                            best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                            best_rank = target_rank
                            return True
        
        return False
    
    # Try ranks from low to high
    for target_rank in range(12, rank_sb):
        if time.time() > deadline - 20:
            break
        found = try_rank(target_rank, max_attempts=8)
        if found:
            # Found a solution, try to go lower if time permits
            if time.time() > deadline - 25:
                break
    
    return best
# EVOLVE-BLOCK-END
