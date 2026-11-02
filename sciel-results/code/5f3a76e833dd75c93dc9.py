import time
import random
import numpy as np
from fractions import Fraction
from math import gcd

DIMENSIONS = [2, 3, 7]
DEADLINE = 110.0

def _to_rational(val, max_denom=6):
    """Convert a float to the nearest rational with denominator <= max_denom."""
    # Try small denominators
    for d in range(1, max_denom + 1):
        num = round(val * d)
        if abs(val - num / d) < 1e-9:
            return num / d
    # Fallback: round to nearest half-integer
    return round(val * 2) / 2

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _try_random_rank(rank, n, m, p, max_trials=200):
    """Try to find a rank-rank decomposition via random U,V and solving for W."""
    num_vars = p * n * rank
    num_eqs = n * m * p
    
    for trial in range(max_trials):
        # Random U and V with small integer entries
        U_rand = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(n*m)]
        V_rand = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(m*p)]
        
        # Build linear system A x = b for W
        # Variables x are w_{k,i,c}
        # Index mapping: var_idx = (k*n + i)*rank + c
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        coeff = U_rand[i*m+j][c] * V_rand[j*p+k][c]
                        if coeff != 0:
                            var_idx = (k*n + i)*rank + c
                            A[eq_idx, var_idx] = coeff
                    eq_idx += 1
        
        # Solve A x = b
        try:
            x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            # Check residual
            if len(residuals) > 0 and residuals[0] < 1e-8:
                # Solution exists - convert to rational W
                W_rand = [[0.0]*rank for _ in range(p*n)]
                valid = True
                for k in range(p):
                    for i in range(n):
                        for c in range(rank):
                            val = x[(k*n + i)*rank + c]
                            # Try to find a rational approximation
                            w_val = _to_rational(val, max_denom=4)
                            W_rand[k*n+i][c] = w_val
                
                # Verify the solution
                if _verify(U_rand, V_rand, W_rand):
                    return {"U": U_rand, "V": V_rand, "W": W_rand}
        except:
            pass
    
    return None

def _try_rank_with_polish(rank, n, m, p, max_trials=50):
    """Try to find a rank-rank decomposition with a polish step."""
    num_vars = p * n * rank
    num_eqs = n * m * p
    
    for trial in range(max_trials):
        # Random U and V
        U_rand = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
        V_rand = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
        
        # Build linear system
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        coeff = U_rand[i*m+j][c] * V_rand[j*p+k][c]
                        if coeff != 0:
                            var_idx = (k*n + i)*rank + c
                            A[eq_idx, var_idx] = coeff
                    eq_idx += 1
        
        # Solve and polish
        try:
            x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and residuals[0] < 1e-8:
                # Polish: try nearby rational values
                W_rand = [[0.0]*rank for _ in range(p*n)]
                for k in range(p):
                    for i in range(n):
                        for c in range(rank):
                            val = x[(k*n + i)*rank + c]
                            # Try denominators 1, 2, 3
                            best_w = None
                            best_err = float('inf')
                            for d in [1, 2, 3]:
                                num = round(val * d)
                                w_candidate = num / d
                                err = abs(val - w_candidate)
                                if err < best_err:
                                    best_err = err
                                    best_w = w_candidate
                            W_rand[k*n+i][c] = best_w
                
                if _verify(U_rand, V_rand, W_rand):
                    return {"U": U_rand, "V": V_rand, "W": W_rand}
        except:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Create schoolbook decomposition
    U_sb = [[0]*schoolbook_rank for _ in range(n*m)]
    V_sb = [[0]*schoolbook_rank for _ in range(m*p)]
    W_sb = [[0]*schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower rank solutions
    # Search from schoolbook_rank - 1 down to a reasonable lower bound
    # For 2x3x7, known lower bounds suggest rank <= 30 or so
    min_rank_to_try = 20
    
    for rank in range(schoolbook_rank - 1, min_rank_to_try - 1, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Determine number of trials based on remaining time
        remaining_time = DEADLINE - 10 - (time.time() - start_time)
        if remaining_time < 2:
            break
        
        # Estimate time per trial
        trials_per_rank = min(100, max(10, int(remaining_time / 0.5)))
        
        # Try random search
        result = _try_random_rank(rank, n, m, p, max_trials=trials_per_rank)
        if result is not None:
            best_rank = rank
            best_result = result
            continue
        
        # If not found, try with polish
        if time.time() - start_time < DEADLINE - 5:
            result = _try_rank_with_polish(rank, n, m, p, max_trials=max(5, trials_per_rank // 4))
            if result is not None:
                best_rank = rank
                best_result = result
                continue
    
    return best_result
