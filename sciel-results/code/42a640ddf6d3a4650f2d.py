import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a dense array and a list of non-zero indices."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                T_indices.append((a, b, c))
    return T, T_indices

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0])
    
    # Convert to Fraction
    U_frac = [[Fraction(U[i][r]) for r in range(rank)] for i in range(nm)]
    V_frac = [[Fraction(V[i][r]) for r in range(rank)] for i in range(mp)]
    W_frac = [[Fraction(W[i][r]) for r in range(rank)] for i in range(pn)]
    
    # Check target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = Fraction(0)
                for r in range(rank):
                    s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                if s != 1:
                    return False
    
    # Check non-target entries
    T_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_set.add((a, b, c))
                
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in T_set:
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                    if s != 0:
                        return False
    return True

def _schoolbook(n, m, p):
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

def _try_rank(r, n, m, p, T, T_indices, time_limit):
    """Try to find a rank-r decomposition using local search."""
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    
    # Initialize with random half-integers
    # We use a set of candidate values
    candidates = [-1, -0.5, 0, 0.5, 1]
    
    # To make the search effective, we initialize with a "sparse" random pattern
    # and then refine.
    
    for attempt in range(100):
        if time.time() > start_time + time_limit:
            return None
            
        # Initialize U, V, W
        # Bias towards 0 to keep sparsity
        U = [[random.choice(candidates) if random.random() < 0.5 else 0 for _ in range(r)] for _ in range(nm)]
        V = [[random.choice(candidates) if random.random() < 0.5 else 0 for _ in range(r)] for _ in range(mp)]
        W = [[random.choice(candidates) if random.random() < 0.5 else 0 for _ in range(r)] for _ in range(pn)]
        
        # Convert to numpy for fast error calculation
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        best_error = float('inf')
        best_state = None
        
        for it in range(500):
            if time.time() > start_time + time_limit:
                return None
                
            # Calculate error
            # T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
            # This is expensive. Let's compute error only on non-zero T entries and a sample of zeros.
            
            error = 0.0
            # Check target entries
            for a, b, c in T_indices:
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                error += (val - 1.0) ** 2
            
            # Check some non-target entries
            # Sample 10 random non-target entries
            for _ in range(10):
                a = random.randint(0, nm - 1)
                b = random.randint(0, mp - 1)
                c = random.randint(0, pn - 1)
                if (a, b, c) not in T_indices:
                    val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                    error += val ** 2
            
            if error < best_error:
                best_error = error
                best_state = (U_arr.copy(), V_arr.copy(), W_arr.copy())
                
                if error < 1e-10:
                    # Verify exactly
                    U_exact = best_state[0].tolist()
                    V_exact = best_state[1].tolist()
                    W_exact = best_state[2].tolist()
                    # Snap to half-integers
                    U_exact = [[round(v * 2) / 2 for v in row] for row in U_exact]
                    V_exact = [[round(v * 2) / 2 for v in row] for row in V_exact]
                    W_exact = [[round(v * 2) / 2 for v in row] for row in W_exact]
                    
                    if _verify_decomposition_exact(U_exact, V_exact, W_exact, n, m, p):
                        return {"U": U_exact, "V": V_exact, "W": W_exact}
            
            # Perturb one entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                row = random.randint(0, nm - 1)
                arr = U_arr
            elif mat_idx == 1:
                row = random.randint(0, mp - 1)
                arr = V_arr
            else:
                row = random.randint(0, pn - 1)
                arr = W_arr
                
            col = random.randint(0, r - 1)
            # Try a small perturbation
            delta = random.choice([-0.5, 0.5, -1, 1])
            arr[row, col] += delta
            
    return None

def construct():
    n, m, p = DIMENSIONS
    T, T_indices = _build_target_tensor(n, m, p)
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Lower bound is max(n*m, m*p, p*n) = 18
    # We try ranks from 18 to 35
    # Allocate time wisely
    total_time_limit = 120
    start_time = time.time()
    
    # We will try to find a solution for each rank
    # If we find one, we return it immediately.
    # We spend a fraction of the time on each rank.
    # Let's spend 5 seconds per rank attempt, up to 30 seconds total for search.
    # If no solution found in 30 seconds, return schoolbook.
    
    search_time_limit = 30
    time_per_rank = 2.0
    
    for r in range(18, best_rank):
        if time.time() - start_time > search_time_limit:
            break
            
        sol = _try_rank(r, n, m, p, T, T_indices, time_per_rank)
        if sol is not None:
            return sol
            
    return best
