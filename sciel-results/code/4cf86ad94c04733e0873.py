"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m + j]
                v_row = V[j*p + k]
                w_row = W[k*n + i]
                s = np.dot(u_row, v_row * w_row)
                err += (s - 1.0) ** 2
    return err

def _snap_and_verify(U, V, W, n, m, p, tol=1e-6):
    """Try to snap to half-integers and verify. Returns dict if valid, else None."""
    # Try rounding to half-integers
    U_h = np.round(U * 2) / 2
    V_h = np.round(V * 2) / 2
    W_h = np.round(W * 2) / 2
    
    err_h = _compute_error(U_h, V_h, W_h, n, m, p)
    if err_h < 1e-9:
        return {
            "U": U_h.tolist(),
            "V": V_h.tolist(),
            "W": W_h.tolist()
        }
    
    # Try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    
    err_i = _compute_error(U_i, V_i, W_i, n, m, p)
    if err_i < 1e-9:
        return {
            "U": U_i.tolist(),
            "V": V_i.tolist(),
            "W": W_i.tolist()
        }
    
    # If raw error is very small but snap fails, check if raw is acceptable? 
    # The task requires integer/half-integer entries. So we must snap.
    # If snap fails, this trial is a failure.
    return None

def _gradient_descent(U, V, W, n, m, p, steps=200, lr=0.01, time_limit=10.0):
    """Perform gradient descent to minimize the error."""
    start = time.time()
    R = U.shape[1]
    
    for step in range(steps):
        if time.time() > start + time_limit:
            break
            
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m + j]
                    v_row = V[j*p + k]
                    w_row = W[k*n + i]
                    
                    s = np.dot(u_row, v_row * w_row)
                    err_term = 2 * (s - 1.0)
                    
                    dU[i*m + j] += err_term * v_row * w_row
                    dV[j*p + k] += err_term * u_row * w_row
                    dW[k*n + i] += err_term * u_row * v_row
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
    return U, V, W

def _try_rank_with_grad(n, m, p, target_rank, time_limit, num_tries=10):
    """Try to find a decomposition with target_rank using gradient descent."""
    start_time = time.time()
    
    for trial in range(num_tries):
        if time.time() > start_time + time_limit:
            break
            
        # Random initialization
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Run gradient descent
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
            
        U, V, W = _gradient_descent(U, V, W, n, m, p, steps=100, lr=0.05, time_limit=remaining * 0.8)
        
        # Check error
        err = _compute_error(U, V, W, n, m, p)
        
        # If error is very small, try to snap to half-integers
        if err < 1e-4:
            result = _snap_and_verify(U, V, W, n, m, p)
            if result is not None:
                return result
                
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _strassen_block_decomposition(n, m, p):
    """
    Try to find a decomposition using Strassen-like block strategies.
    For 3x5x5, we can view this as multiplying a 3x5 by 5x5.
    This is not a standard square block case, but we can try to embed it
    or use specific tensor identities.
    
    Given the small size, a direct random search might be more effective,
    but we can try to initialize with some structure.
    """
    # For now, return None to indicate this specific strategy isn't fully implemented
    # in a way that guarantees improvement over random search for this specific shape.
    # A true Strassen algorithm requires specific block sizes (powers of 2).
    # 3x5x5 is irregular.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Schoolbook fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    # Try to find better ranks
    # We try ranks from schoolbook_rank - 1 down to schoolbook_rank - 20
    # For 3x5x5, rank 75 is schoolbook.
    # Let's try 74, 73, 72, 71, 70...
    
    # We will iterate through ranks in descending order of likelihood (closest to schoolbook first)
    # but we want the minimum rank, so we should check if a lower rank is possible.
    # However, finding a rank R decomposition is NP-hard.
    # We will search from schoolbook_rank - 1 downwards.
    
    # To save time, we'll allocate a budget per rank.
    # If we find a valid one, we return it.
    # We should prioritize lower ranks if we have time, but higher ranks (closer to schoolbook)
    # are more likely to be found.
    
    # Strategy:
    # 1. Try rank 74. If found, return.
    # 2. Try rank 73. If found, return.
    # ...
    # This ensures we find the *highest* rank below schoolbook first? No, we want the *lowest* rank.
    # But if rank 70 works, rank 74 also works (by adding zero columns? No, rank is minimal).
    # Actually, if a decomposition of rank R exists, a decomposition of rank R+1 exists (add a zero term).
    # So if we find a valid decomposition of rank 70, we can return it.
    # We should search for the *smallest* possible rank.
    # But searching from low to high is risky if we don't find anything.
    # Searching from high to low (74, 73...) finds a valid one quickly, but it's not minimal.
    # The task asks for "rank-minimal exact tensor decomposition".
    # So we MUST find the minimal rank.
    
    # This implies we need to check if rank 1 is possible, then 2, etc.
    # For 3x5x5, the rank is likely close to 75.
    # We can use a binary search or just linear search from a reasonable lower bound.
    # However, without knowing the lower bound, we might waste time.
    
    # Let's try ranks from 74 down to 60.
    # If we find one at 74, we still need to check if 73 works.
    # So we should iterate from LOW to HIGH?
    # No, if we find 70, we can stop? No, 69 might work.
    # So we must check all ranks from 1 to 74? That's too slow.
    
    # Insight: The rank of matrix multiplication tensor is known to be at least n*m*p / 2 (for square) or similar bounds.
    # For 3x5x5, lower bound is likely around 60-70.
    
    # Let's try a strategy:
    # Try rank 74. If success, try 73. If success, try 72...
    # Stop when we fail to find a decomposition for rank R.
    # Then R+1 is the minimal rank?
    # Not necessarily. It's possible that rank 74 works, 73 fails, but 72 works?
    # No, if rank 72 works, rank 73 works (by adding a zero column).
    # So the set of valid ranks is an interval [R_min, infinity).
    # So we can binary search for R_min!
    
    # Low: 1, High: 74
    # If rank 74 works, check 73.
    # If rank 73 works, check 72.
    # ...
    # This is linear search from top down.
    # Binary search would be:
    # Check 37. If works, check 18. If fails, check 27...
    # But checking a rank takes time.
    
    # Given the time limit, let's try ranks 74, 73, 72, 71, 70.
    # If we find 70, we assume it's minimal? No, we should check 69, 68...
    # But if we run out of time, we return the best found (lowest rank).
    
    best_result = None
    best_rank = schoolbook_rank
    
    # We will search from schoolbook_rank - 1 down to max(1, schoolbook_rank - 20)
    # We prioritize lower ranks (better score) but need to verify them.
    # Since valid ranks form an interval, if R works, R+1 works.
    # So if we find R_min, we can stop.
    # We should search from LOW to HIGH to find the minimum?
    # No, if we search 1, 2, 3... it will fail for a long time.
    # We should search from HIGH to LOW to find a valid one, then go lower.
    
    # Let's try 74, 73, 72, 71, 70.
    # If 70 works, try 69, 68...
    
    for target_rank in range(schoolbook_rank - 1, max(1, schoolbook_rank - 20) - 1, -1):
        if time.time() > deadline - 10.0:
            break
            
        time_per_rank = min(20.0, deadline - time.time() - 5.0)
        if time_per_rank < 1.0:
            break
            
        result = _try_rank_with_grad(n, m, p, target_rank, time_per_rank, num_tries=5)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Continue searching for lower rank
        else:
            # If this rank failed, lower ranks are less likely to be found quickly?
            # Or maybe they are just harder?
            # Since valid ranks are contiguous, if R fails, R-1 might still work?
            # No, if R-1 works, R works.
            # So if R fails, R-1 MUST fail.
            # Therefore, we can STOP if a rank fails!
            # Wait, is that true?
            # If a decomposition of rank R-1 exists, we can pad it with zeros to get rank R.
            # So if R-1 is valid, R is valid.
            # Contrapositive: If R is invalid, R-1 is invalid.
            # So we can break!
            break
            
    if best_result is not None:
        return best_result
        
    return schoolbook

# EVOLVE-BLOCK-END
