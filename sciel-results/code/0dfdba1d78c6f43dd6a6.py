"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W form a valid decomposition of the tensor T[i,j,k] = 1.
    The tensor T is decomposed as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(rank):
                    lhs += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(lhs - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(arr, tol=1e-9):
    for row in arr:
        for val in row:
            v = val * 2
            if abs(v - round(v)) > tol:
                return False
    return True

def _round_to_half(arr):
    return [[round(val * 2) / 2.0 for val in row] for row in arr]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)
    # W: (p*n, rank) = (6, rank)
    
    best = _schoolbook()
    best_rank = n * m * p
    
    def compute_residual(U_flat, V_flat, W_flat, rank):
        U = np.array(U_flat).reshape(n * m, rank)
        V = np.array(V_flat).reshape(m * p, rank)
        W = np.array(W_flat).reshape(p * n, rank)
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    total += (s - 1.0) ** 2
        return total

    def half_integer_penalty(U_flat, V_flat, W_flat):
        penalty = 0.0
        for arr in [U_flat, V_flat, W_flat]:
            for val in arr:
                v = val * 2
                penalty += (v - round(v)) ** 2
        return penalty

    def objective(x, rank):
        U_flat = x[:n * m * rank]
        V_flat = x[n * m * rank : n * m * rank + m * p * rank]
        W_flat = x[n * m * rank + m * p * rank : n * m * rank + m * p * rank + p * n * rank]
        
        res = compute_residual(U_flat, V_flat, W_flat, rank)
        pen = half_integer_penalty(U_flat, V_flat, W_flat)
        return res + 0.1 * pen

    # Try ranks lower than schoolbook
    for target_rank in range(15, best_rank):
        if time.time() > deadline - 15:
            break
            
        found = False
        # Multiple random attempts
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + target_rank * 100 + attempt)
            
            # Initialize with small random values
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            size_W = p * n * target_rank
            x0 = np.random.normal(0, 1, size=size_U + size_V + size_W)
            
            try:
                res = minimize(objective, x0, args=(target_rank,), method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
                
                if res.fun < 1e-6:
                    U_sol = res.x[:size_U].reshape(n * m, target_rank)
                    V_sol = res.x[size_U : size_U + size_V].reshape(m * p, target_rank)
                    W_sol = res.x[size_U + size_V :].reshape(p * n, target_rank)
                    
                    # Check if close to half-integers
                    if _is_half_integer(U_sol.tolist(), 1e-3) and \
                       _is_half_integer(V_sol.tolist(), 1e-3) and \
                       _is_half_integer(W_sol.tolist(), 1e-3):
                        
                        # Round to half-integers
                        U_r = _round_to_half(U_sol.tolist())
                        V_r = _round_to_half(V_sol.tolist())
                        W_r = _round_to_half(W_sol.tolist())
                        
                        # Verify exactness
                        if _verify(U_r, V_r, W_r):
                            best = {"U": U_r, "V": V_r, "W": W_r}
                            best_rank = target_rank
                            found = True
                            break
            except Exception:
                continue
        
        if found:
            break
            
    return best

# EVOLVE-BLOCK-END
