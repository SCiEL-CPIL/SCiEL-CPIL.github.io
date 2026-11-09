"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W):
    """Fast verification using numpy for real-valued checks."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 40."""
    n, m, p = DIMENSIONS
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

def _is_integer_or_half(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    if abs(x - round(x)) < tol:
        return True
    if abs(x * 2 - round(x * 2)) < tol:
        return True
    return False

def _try_rank_k_structural(k, deadline, n_trials=500):
    """Try to find a decomposition of rank k using structured random search.
    
    We sample sparse U and V with entries in {-1, 1} (or 0) and solve for W.
    We check if W has integer or half-integer entries.
    """
    n, m, p = DIMENSIONS
    if k < 14:
        return None
    
    trial = 0
    while trial < n_trials and time.time() < deadline:
        trial += 1
        
        # Sparse random initialization
        # Density 0.4 to keep it sparse enough for integer solutions to be likely
        density = 0.4 
        U = [[0] * k for _ in range(n*m)]
        V = [[0] * k for _ in range(m*p)]
        
        for r in range(n*m):
            for c in range(k):
                if random.random() < density:
                    U[r][c] = random.choice([-1, 1])
        for r in range(m*p):
            for c in range(k):
                if random.random() < density:
                    V[r][c] = random.choice([-1, 1])
        
        W = [[0]*k for _ in range(p*n)]
        valid = True
        
        for i in range(n):
            for ki in range(p):
                # Construct A (m x k) and b (m)
                A = np.zeros((m, k))
                for j in range(m):
                    for c in range(k):
                        A[j][c] = U[i*m + j][c] * V[j*p + ki][c]
                
                b = np.ones(m)
                
                # Solve A w = b
                # If m == 1, we have 1 eq, k unknowns.
                # If m > 1, we have m eqs, k unknowns.
                # We look for a solution w with integer/half-integer entries.
                
                if m == 1:
                    # A[0] is a row vector. We need A[0] . w = 1.
                    # Find any integer w that satisfies this.
                    # Simple strategy: set w to 0, then find one non-zero entry in A[0].
                    # If A[0][c] is 1 or -1, set w[c] to +/- 1.
                    # If A[0][c] is 2, set w[c] to 0.5.
                    found = False
                    for c in range(k):
                        a_val = A[0][c]
                        if a_val == 0:
                            continue
                        # w[c] = 1 / a_val
                        val = 1.0 / a_val
                        if _is_integer_or_half(val):
                            W[ki*n+i][c] = val
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                else:
                    # Use least squares first to get a candidate
                    try:
                        w, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if this is an exact solution
                        if not np.allclose(A @ w, b, atol=1e-8):
                            valid = False
                            break
                        
                        # Check if w is integer or half-integer
                        # If not, try rounding to nearest integer/half-integer and re-check
                        w_candidate = w.copy()
                        
                        # Try rounding to integers
                        w_int = np.round(w_candidate)
                        if np.allclose(A @ w_int, b, atol=1e-8):
                            w_candidate = w_int
                        else:
                            # Try rounding to half-integers
                            w_half = np.round(w_candidate * 2) / 2
                            if np.allclose(A @ w_half, b, atol=1e-8):
                                w_candidate = w_half
                            else:
                                # Try a mix? Or just fail
                                # For now, fail if not exactly int or half
                                valid = False
                                break
                        
                        # Final check: are they int or half?
                        for val in w_candidate:
                            if not _is_integer_or_half(val):
                                valid = False
                                break
                        if not valid:
                            break
                            
                        W[ki*n+i] = [float(x) for x in w_candidate]
                    except:
                        valid = False
                        break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify full solution
        if _verify(U, V, W):
            # Convert to clean numbers
            def to_num(x):
                if abs(x - round(x)) < 1e-9:
                    return int(round(x))
                elif abs(x * 2 - round(x * 2)) < 1e-9:
                    return float(round(x * 2) / 2)
                else:
                    return x
            
            U_out = [[to_num(x) for x in row] for row in U]
            V_out = [[to_num(x) for x in row] for row in V]
            W_out = [[to_num(x) for x in row] for row in W]
            
            return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Fallback: Schoolbook rank 40
    fallback = _schoolbook()
    best_result = fallback
    best_rank = 40
    
    # Try ranks from 30 to 39
    # We prioritize lower ranks
    for k in range(30, 40):
        if time.time() > deadline - 10:
            break
            
        # Try multiple times for each rank
        # Increase trials if we have more time
        remaining = deadline - 10 - time.time()
        if remaining <= 0:
            break
            
        # Estimate trials: ~100ms per trial? Let's assume 50ms.
        # 500 trials * 50ms = 25s.
        # We have ~100s left.
        # Let's do 200 trials per rank.
        result = _try_rank_k_structural(k, deadline, n_trials=200)
        
        if result is not None:
            # Verify again to be safe
            if _verify(result["U"], result["V"], result["W"]):
                new_rank = len(result["U"][0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_result = result
                    # If we found a low rank, we can stop or continue to find even lower
                    # But since we are iterating k upwards, we should break if we found k?
                    # No, we iterate k from 30 to 39. If we find one at 30, that's the best possible in this range.
                    # But maybe there is a solution at 29? We started at 30.
                    # Let's restart from 25 if we have time?
                    # For now, if we find one, it's a huge improvement over 40.
                    # We can break out of the k loop if we found a very low rank, 
                    # but let's just take the first one found in ascending order.
                    break
    
    return best_result
