import time
import numpy as np

DIMENSIONS = [2, 3, 3]
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For each (i, j, k), the tensor entry is 1
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # For all other (a, b, c) combinations, the tensor entry is 0
    # We need to check all (a, b, c) that are NOT of the form (i*m+j, j*p+k, k*n+i)
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_set.add((i * m + j, j * p + k, k * n + i))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid_set:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True

def _try_rank(rank, n, m, p, time_limit):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    start = time.time()
    np.random.seed(42)
    
    # Try random initializations with local optimization
    # For exact decomposition, we need integer/half-integer entries
    # Start with random real values and try to converge
    
    best_err = float('inf')
    best_factors = None
    
    max_iters = 2000
    lr = 0.01
    
    for attempt in range(20):
        if time.time() - start > time_limit:
            break
        
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            # Compute error and gradients
            # err = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
            # This is expensive for full tensor, so sample
            
            # Use a subset of constraints for speed
            err = 0.0
            # Check valid entries
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        err += (val - 1.0) ** 2
            
            # Check a random subset of invalid entries
            n_invalid = n * m * m * p * p * n - n * m * p
            for _ in range(min(100, n_invalid)):
                a = np.random.randint(0, n * m)
                b = np.random.randint(0, m * p)
                c = np.random.randint(0, p * n)
                # Check if this is a valid entry
                valid = False
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if a == i*m+j and b == j*p+k and c == k*n+i:
                                valid = True
                                break
                        if valid:
                            break
                    if valid:
                        break
                if not valid:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err += val ** 2
            
            if err < best_err:
                best_err = err
                best_factors = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-8:
                break
            
            # Gradient step (simplified)
            # dU[a,r] = 2 * sum_{b,c} (pred[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # This is too expensive, so use a simpler update
            # Just do a small random perturbation biased by error
            break  # Skip gradient for now, just try more random inits
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-8 and best_factors is not None:
        U, V, W = best_factors
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify the rounded version
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor M_{2,3,3}.
    The rank is 18, which is the schoolbook rank.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition as fallback
    rank = schoolbook_rank
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
    
    # Try to find a lower-rank decomposition
    # Theoretical lower bound: max(n*m, m*p, p*n) = 9
    # Try ranks from 9 up to schoolbook_rank - 1
    for r in range(max(n * m, m * p, p * n), schoolbook_rank):
        remaining = _DEADLINE - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        result = _try_rank(r, n, m, p, time_limit=remaining - 2.0)
        if result is not None:
            U, V, W = result
            rank = r
            break
    
    return {"U": U, "V": V, "W": W}
