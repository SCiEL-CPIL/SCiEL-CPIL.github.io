import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 4, 5]
DEADLINE = 140.0

def _schoolbook():
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

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(W[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _random_integer_decomp(r, n, m, p, seed=0):
    """Generate a random decomposition with integer entries in {-1, 0, 1}."""
    rng = np.random.RandomState(seed)
    U = rng.randint(-1, 2, size=(n*m, r))
    V = rng.randint(-1, 2, size=(m*p, r))
    W = rng.randint(-1, 2, size=(p*n, r))
    return U.tolist(), V.tolist(), W.tolist()

def _local_search_integer(U, V, W, n, m, p, max_iters=1000, temp=1.0, cooling=0.99):
    """
    Simulated Annealing on the integer lattice.
    We represent the decomposition as a list of rank-1 terms (u, v, w).
    We try to modify terms to reduce the residual.
    """
    r = len(W[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def get_residual(U, V, W):
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    res += (val - 1.0)**2
        return res

    current_res = get_residual(U_arr, V_arr, W_arr)
    best_res = current_res
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    current_temp = temp
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
            
        # Pick a random term c
        c = np.random.randint(0, r)
        
        # Pick a random factor and entry to perturb
        factor_choice = np.random.randint(0, 3)
        if factor_choice == 0:
            idx = np.random.randint(0, n*m)
            arr = U_arr
        elif factor_choice == 1:
            idx = np.random.randint(0, m*p)
            arr = V_arr
        else:
            idx = np.random.randint(0, p*n)
            arr = W_arr
            
        old_val = arr[idx, c]
        # Perturb by -1, 0, or 1 (keeping it small)
        delta = np.random.choice([-1, 0, 1])
        new_val = old_val + delta
        if abs(new_val) > 2: # Keep values bounded
            continue
            
        arr[idx, c] = new_val
        
        new_res = get_residual(U_arr, V_arr, W_arr)
        
        # Acceptance criterion
        delta_res = new_res - current_res
        if delta_res < 0 or np.random.random() < np.exp(-delta_res / current_temp):
            current_res = new_res
            if current_res < best_res:
                best_res = current_res
                best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        else:
            arr[idx, c] = old_val
            
        current_temp *= cooling
        
        if best_res < 1e-6:
            break
            
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_res

def _try_integer_search(n, m, p, target_rank, time_limit=10.0, seed=0):
    start = time.time()
    for attempt in range(20):
        if time.time() - start > time_limit:
            break
            
        U, V, W = _random_integer_decomp(target_rank, n, m, p, seed=seed + attempt)
        
        # Initial check
        if _verify(U, V, W, n, m, p):
            return U, V, W
        
        # Local search
        U, V, W, res = _local_search_integer(U, V, W, n, m, p, max_iters=500)
        
        if res < 1e-6:
            if _verify(U, V, W, n, m, p):
                return U, V, W
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank using integer search
    # Start from best_rank - 1 down to a reasonable lower bound
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But for matrix mult, it's usually higher.
    
    # We prioritize checking ranks close to 60 first, as they are more likely to exist
    # and might be found by simple perturbations of the schoolbook solution.
    
    # Strategy: Take the schoolbook solution and try to merge terms or reduce rank
    # by setting some columns to zero and re-optimizing?
    # Or simply try random integer decompositions for r < 60.
    
    for r in range(best_rank - 1, 55, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        time_per_rank = min(5.0, (DEADLINE - 20 - (time.time() - start_time)) / 5.0)
        if time_per_rank < 1.0:
            break
            
        result = _try_integer_search(n, m, p, r, time_limit=time_per_rank, seed=r*13)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = r
            # If we found one, we can stop or continue to find even lower?
            # The task asks for rank-minimal. So we should continue searching for lower.
            # But time is limited. We break and return the best found so far if time is tight.
            # Let's try to go lower if time permits.
            pass 

    # If we haven't found anything better, return schoolbook
    if time.time() - start_time > DEADLINE - 5:
        return _schoolbook()
        
    return best
