import numpy as np
import random
import time

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 120  # Leave margin
    
    # The tensor T[i,j,k] = 1 for all (i,j,k)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Start with schoolbook rank 60
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Build schoolbook
    rank0 = best_rank
    U0 = [[0]*rank0 for _ in range(n*m)]
    V0 = [[0]*rank0 for _ in range(m*p)]
    W0 = [[0]*rank0 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to find lower rank decompositions
    # Strategy: randomized local search with half-integer constraint
    
    def residual(U, V, W, r):
        """Compute the residual: max |T[i,j,k] - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]|"""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_ in range(r):
                        s += U[i*m+j][r_] * V[j*p+k][r_] * W[k*n+i][r_]
                    res = abs(1.0 - s)
                    if res > max_res:
                        max_res = res
        return max_res
    
    def to_half_integer(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2
    
    def try_rank(r, time_budget=10):
        """Try to find a rank-r decomposition using local search."""
        start_time = time.time()
        
        # Multiple restarts
        for restart in range(50):
            if time.time() - start_time > time_budget:
                break
            if time.time() > deadline:
                return None
            
            # Initialize with random half-integers in a reasonable range
            scale = 1.0
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
            
            # Local search: randomly perturb entries
            best_res = residual(U, V, W, r)
            for step in range(2000):
                if time.time() - start_time > time_budget:
                    break
                if time.time() > deadline:
                    return None
                
                # Pick a random matrix and entry
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, n*m-1)
                elif mat == 'V':
                    row = random.randint(0, m*p-1)
                else:
                    row = random.randint(0, p*n-1)
                col = random.randint(0, r-1)
                
                # Try a random perturbation
                old_val = {'U': U, 'V': V, 'W': W}[mat][row][col]
                new_val = to_half_integer(old_val + random.choice([-1, -0.5, 0.5, 1, 1.5, 2, -1.5, -2]))
                
                if mat == 'U':
                    U[row][col] = new_val
                elif mat == 'V':
                    V[row][col] = new_val
                else:
                    W[row][col] = new_val
                
                new_res = residual(U, V, W, r)
                if new_res < best_res:
                    best_res = new_res
                else:
                    # Revert
                    if mat == 'U':
                        U[row][col] = old_val
                    elif mat == 'V':
                        V[row][col] = old_val
                    else:
                        W[row][col] = old_val
                
                if best_res < 1e-10:
                    # Found exact solution! Verify and return
                    if residual(U, V, W, r) < 1e-10:
                        return (U, V, W)
            
            # Check if best_res is small enough
            if best_res < 1e-10:
                return (U, V, W)
        
        return None
    
    # Try ranks from 59 down to 50
    for r in range(59, 49, -1):
        if time.time() > deadline - 15:
            break
        result = try_rank(r, time_budget=min(10, max(1, (deadline - 15 - time.time()))))
        if result is not None:
            U, V, W = result
            # Verify
            if residual(U, V, W, r) < 1e-10:
                return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
