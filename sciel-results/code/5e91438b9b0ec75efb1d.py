import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _validate_full(U, V, W, n=3, m=3, p=5):
    """Validate that the decomposition exactly reproduces the matrix multiplication tensor.
    T[(i,j),(j',k),(k',i')] = 1 iff i==i', j==j', k==k', else 0.
    """
    rank = len(U[0])
    # Check all on-diagonal entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Check off-diagonal entries (should be 0)
    # U index: (i,j), V index: (j',k), W index: (k',i')
    # Off-diagonal when not (i==i' and j==j' and k==k')
    for i in range(n):
        for j in range(m):
            for jp in range(m):  # j'
                for k in range(p):
                    for kp in range(p):  # k'
                        for ip in range(n):  # i'
                            if i == ip and j == jp and k == kp:
                                continue  # skip on-diagonal
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j][r] * V[jp*p+k][r] * W[kp*n+ip][r]
                            if abs(s) > 1e-9:
                                return False
    return True

def _validate_half_integer(U, V, W, n=3, m=3, p=5):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return _validate_full(U, V, W, n, m, p)

def _compute_all_errors(U, V, W, n=3, m=3, p=5):
    """Compute all tensor entries and return max error."""
    rank = len(U[0])
    max_err = 0.0
    # On-diagonal: should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                max_err = max(max_err, abs(s - 1.0))
    # Off-diagonal: should be 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                continue
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j][r] * V[jp*p+k][r] * W[kp*n+ip][r]
                            max_err = max(max_err, abs(s))
    return max_err

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = 3, 3, 5
    
    # Schoolbook fallback
    U = [[0]*45 for _ in range(n*m)]
    V = [[0]*45 for _ in range(m*p)]
    W = [[0]*45 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    schoolbook = {"U": U, "V": V, "W": W}
    
    best_rank = 45
    best_sol = schoolbook
    
    # Strategy: Try to find lower rank decompositions using structured search
    # Key insight: The tensor has a specific structure. We can use the fact that
    # matrix multiplication can be decomposed using various algebraic tricks.
    
    # For 3x3x5, let's try ranks from 30 to 44
    rng = random.Random(42)
    
    # Try different rank values
    for r in range(30, 45):
        if _time_remaining() < 15.0:
            break
            
        found = False
        for attempt in range(30):
            if _time_remaining() < 15.0:
                break
                
            # Initialize with random sparse matrices with half-integer entries
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            
            # Use structured initialization: try to preserve some symmetry
            # Initialize with small random values
            for i in range(n*m):
                for rr in range(r):
                    if rng.random() < 0.3:
                        U[i][rr] = rng.choice([-1, -0.5, 0.5, 1])
            for j in range(m*p):
                for rr in range(r):
                    if rng.random() < 0.3:
                        V[j][rr] = rng.choice([-1, -0.5, 0.5, 1])
            for k in range(p*n):
                for rr in range(r):
                    if rng.random() < 0.3:
                        W[k][rr] = rng.choice([-1, -0.5, 0.5, 1])
                        
            # Local search with proper validation
            current_err = _compute_all_errors(U, V, W, n, m, p)
            
            for it in range(200):
                if _time_remaining() < 15.0:
                    break
                    
                if current_err < 1e-9:
                    if _validate_half_integer(U, V, W, n, m, p):
                        best_rank = r
                        best_sol = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
                        found = True
                        break
                    
                # Pick a random entry to modify
                mat_choice = rng.randint(0, 2)
                if mat_choice == 0:
                    idx_u = rng.randint(0, n*m-1)
                    idx_r = rng.randint(0, r-1)
                    old_val = U[idx_u][idx_r]
                    # Try small perturbations
                    new_val = old_val + rng.choice([-1, -0.5, 0.5, 1])
                    U[idx_u][idx_r] = new_val
                elif mat_choice == 1:
                    idx_v = rng.randint(0, m*p-1)
                    idx_r = rng.randint(0, r-1)
                    old_val = V[idx_v][idx_r]
                    new_val = old_val + rng.choice([-1, -0.5, 0.5, 1])
                    V[idx_v][idx_r] = new_val
                else:
                    idx_w = rng.randint(0, p*n-1)
                    idx_r = rng.randint(0, r-1)
                    old_val = W[idx_w][idx_r]
                    new_val = old_val + rng.choice([-1, -0.5, 0.5, 1])
                    W[idx_w][idx_r] = new_val
                    
                new_err = _compute_all_errors(U, V, W, n, m, p)
                
                if new_err < current_err:
                    current_err = new_err
                else:
                    # Revert
                    if mat_choice == 0:
                        U[idx_u][idx_r] = old_val
                    elif mat_choice == 1:
                        V[idx_v][idx_r] = old_val
                    else:
                        W[idx_w][idx_r] = old_val
                            
        if found:
            break
            
    return best_sol
