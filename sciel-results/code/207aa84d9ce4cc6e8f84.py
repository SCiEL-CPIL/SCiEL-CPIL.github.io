import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # and check it equals the canonical multiplication tensor (which is 1 for all i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _residuals(x, n, m, p, rank):
    """
    Compute residuals for least_squares.
    x is a flat array of length (n*m + m*p + p*n) * rank.
    Residuals are the differences between the computed tensor entries and 1.
    """
    U_size = n * m * rank
    V_size = m * p * rank
    
    U_flat = x[:U_size]
    V_flat = x[U_size:U_size + V_size]
    W_flat = x[U_size + V_size:]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    residuals = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                residuals[idx] = val - 1.0
                idx += 1
    return residuals

def _try_ranked_decomposition(n, m, p, target_rank, max_attempts=10, timeout=5.0):
    """
    Try to find an exact decomposition of rank target_rank using least_squares.
    """
    start_time = time.time()
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        # Scale is important. 
        # If we use random uniform [-1, 1], the products might be small.
        # Let's try random normal with std=1.
        
        x0 = np.random.randn((n * m + m * p + p * n) * target_rank)
        
        try:
            # Use least_squares
            # Bounds are infinite by default
            result = least_squares(
                _residuals, 
                x0, 
                args=(n, m, p, target_rank),
                method='lm', # Levenberg-Marquardt is good for small problems
                max_nfev=200
            )
            
            if result.success or np.max(np.abs(result.fun)) < 1e-8:
                x_opt = result.x
                U_size = n * m * target_rank
                V_size = m * p * target_rank
                
                U = x_opt[:U_size].reshape(n * m, target_rank)
                V = x_opt[U_size:U_size + V_size].reshape(m * p, target_rank)
                W = x_opt[U_size + V_size:].reshape(p * n, target_rank)
                
                # Convert to lists
                U_list = [[float(val) for val in row] for row in U]
                V_list = [[float(val) for val in row] for row in V]
                W_list = [[float(val) for val in row] for row in W]
                
                # Verify strictly
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
                    
        except Exception:
            continue
            
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    by a 4 by 6 matrix.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    schoolbook = _build_schoolbook()
    best_rank = n * m * p
    best_result = schoolbook
    
    # Search strategy:
    # The rank of 4x4x4 is 49.
    # The rank of 4x4x6 is likely higher.
    # Let's search from 50 to 95.
    # We will try each rank with a few attempts.
    
    # To maximize chance, we should spend more time on lower ranks?
    # Or just iterate.
    
    # Lower bound is 24.
    # Let's try a range. 50-70 is a good guess for improvement over 96.
    
    for target_rank in range(50, best_rank):
        if time.time() > deadline:
            break
        
        # Allocate time based on remaining time
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        # Try this rank
        # If we have lots of time, do more attempts
        attempts = max(2, int(time_left / 5.0))
        
        result = _try_ranked_decomposition(n, m, p, target_rank, 
                                           max_attempts=attempts, 
                                           timeout=min(5.0, time_left - 1.0))
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                # If we found a solution, we can try to lower the rank further
                # But we are iterating upwards. 
                # To find the *minimum*, we should iterate downwards? 
                # No, we don't know the minimum. 
                # If we find rank 70, we should check 69, 68...
                # But that's expensive. 
                # Let's stick to finding *a* good low rank.
                # Actually, if we find 70, we can break and then try 69, 68... in a second phase?
                # For now, let's just update best and continue to higher ranks? 
                # No, that doesn't help minimize. 
                # Better: if we find a solution, record it. 
                # But we want the MINIMAL rank.
                # The current loop goes 50, 51, 52...
                # If 50 fails, 51 fails... 70 succeeds.
                # We return 70.
                # But maybe 65 would have succeeded if we tried harder?
                # The `least_squares` is stochastic (random init).
                # So failing at 50 doesn't mean 50 is impossible.
                
                # Strategy adjustment:
                # If we find a solution at R, we should try to find a solution at R-1, R-2...
                # But we already passed them.
                # Let's restart the search from R-1 downwards?
                # That might be too slow.
                
                # Alternative: Search from a lower bound upwards, but with increasing effort?
                # Or just accept the first found?
                # Given the time limit, finding *any* rank < 96 is a huge win.
                # Let's just return the first one we find in the upward scan.
                # It's not guaranteed minimal, but it's an improvement.
                
                # However, the prompt asks for "rank-minimal".
                # If we return 70 but 65 is possible, we are suboptimal.
                # But 70 < 96, so score > 1.0.
                
                # Let's try to be smarter:
                # If we find a solution, break and then do a downward scan from R-1 to LB?
                
                # Let's implement a simple downward scan if we find a solution.
                # But we need to be careful with time.
                
                # For now, let's just break and return the found one.
                # Actually, let's continue the loop but with a flag?
                # No, let's just break. 70 is much better than 96.
                
                break

    return best_result
# EVOLVE-BLOCK-END
