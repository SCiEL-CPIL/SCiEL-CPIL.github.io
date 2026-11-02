"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import itertools
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix using scipy optimization.
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    def check_time():
        return time.time() - start_time < DEADLINE
    
    def schoolbook():
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
    
    def verify(U, V, W, tol=1e-6):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rank = len(U[0]) if U else 0
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(np.dot(U_np[a], V_np[b]), W_np[c])
                    if abs(val - 1) > tol:
                        return False
        
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c // n
                    if i != i2 or j != j2 or k != k2:
                        val = np.dot(np.dot(U_np[a], V_np[b]), W_np[c])
                        if abs(val) > tol:
                            return False
        return True
    
    def full_verify(U, V, W, tol=1e-8):
        """Full verification with tighter tolerance."""
        return verify(U, V, W, tol=tol)
    
    def try_rank_scipy(r, seed=42, max_nfev=2000, num_zero_constraints=150):
        """
        Try to find a rank-r decomposition using scipy least_squares.
        Uses a subset of zero constraints for speed.
        """
        if not check_time():
            return None
        
        np.random.seed(seed)
        
        # Flatten all variables
        total_vars = (n*m + m*p + p*n) * r
        x0 = np.random.randn(total_vars) * 0.3
        
        # Precompute all inconsistent triples
        inconsistent_triples = []
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c // n
                    if i != i2 or j != j2 or k != k2:
                        inconsistent_triples.append((a, b, c))
        
        # Use a random subset
        if len(inconsistent_triples) > num_zero_constraints:
            idx = np.random.choice(len(inconsistent_triples), num_zero_constraints, replace=False)
            zero_triples = [inconsistent_triples[i] for i in idx]
        else:
            zero_triples = inconsistent_triples
        
        def residuals(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r:(n*m+m*p)*r].reshape(m*p, r)
            W = x[(n*m+m*p)*r:].reshape(p*n, r)
            
            res = []
            # Consistent constraints: should equal 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(np.dot(U[a], V[b]), W[c])
                        res.append(val - 1.0)
            
            # Subset of inconsistent constraints: should equal 0
            for a, b, c in zero_triples:
                val = np.dot(np.dot(U[a], V[b]), W[c])
                res.append(val)
            
            return np.array(res)
        
        try:
            result = least_squares(
                residuals, x0, 
                method='lm',
                max_nfev=max_nfev,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10
            )
            
            if not check_time():
                return None
            
            x = result.x
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r:(n*m+m*p)*r].reshape(m*p, r)
            W = x[(n*m+m*p)*r:].reshape(p*n, r)
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if full_verify(U_list, V_list, W_list, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
            
        except Exception as e:
            pass
        
        return None
    
    # First, get the schoolbook solution as fallback
    result = schoolbook()
    best_rank = n * m * p
    best_result = result
    
    # Try to find better ranks
    ranks_to_try = list(range(15, min(best_rank, 40)))
    
    for r in ranks_to_try:
        if not check_time():
            break
        
        found = False
        for seed in range(5):
            if not check_time():
                break
            res = try_rank_scipy(r, seed=seed)
            if res is not None:
                if full_verify(res["U"], res["V"], res["W"], tol=1e-6):
                    best_rank = r
                    best_result = res
                    found = True
                    break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
