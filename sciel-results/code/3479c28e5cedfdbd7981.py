import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _validate(U, V, W, n=3, m=3, p=5):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _validate_half_integer(U, V, W, n=3, m=3, p=5):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return _validate(U, V, W, n, m, p)

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
    
    # Try to find a better rank using random search + local improvement
    # We aim for rank 30-40
    rng = random.Random(42)
    
    for r in range(30, 45):
        if _time_remaining() < 10.0:
            break
            
        found = False
        for attempt in range(50):
            if _time_remaining() < 10.0:
                break
                
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            
            # Random initialization
            for i in range(n*m):
                for rr in range(r):
                    if rng.random() < 0.5:
                        U[i][rr] = rng.choice([-1, 1])
            for j in range(m*p):
                for rr in range(r):
                    if rng.random() < 0.5:
                        V[j][rr] = rng.choice([-1, 1])
            for k in range(p*n):
                for rr in range(r):
                    if rng.random() < 0.5:
                        W[k][rr] = rng.choice([-1, 1])
                        
            # Local search
            for it in range(100):
                if _time_remaining() < 10.0:
                    break
                    
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(r):
                                s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                            errors.append(abs(s - 1.0))
                
                max_err = max(errors)
                if max_err < 1e-9:
                    if _validate_half_integer(U, V, W, n, m, p):
                        best_rank = r
                        best_sol = {"U": U, "V": V, "W": W}
                        found = True
                        break
                    
                # Pick a random error and try to fix it
                if not found:
                    err_idx = rng.randint(0, len(errors)-1)
                    i = err_idx // (m*p)
                    k = (err_idx % (m*p)) // m
                    j = err_idx % m
                    
                    # Try changing a random entry in U, V, or W
                    mat_choice = rng.randint(0, 2)
                    if mat_choice == 0:
                        idx_u = rng.randint(0, n*m-1)
                        idx_r = rng.randint(0, r-1)
                        old_val = U[idx_u][idx_r]
                        new_val = old_val + rng.choice([-1, 1])
                        U[idx_u][idx_r] = new_val
                    elif mat_choice == 1:
                        idx_v = rng.randint(0, m*p-1)
                        idx_r = rng.randint(0, r-1)
                        old_val = V[idx_v][idx_r]
                        new_val = old_val + rng.choice([-1, 1])
                        V[idx_v][idx_r] = new_val
                    else:
                        idx_w = rng.randint(0, p*n-1)
                        idx_r = rng.randint(0, r-1)
                        old_val = W[idx_w][idx_r]
                        new_val = old_val + rng.choice([-1, 1])
                        W[idx_w][idx_r] = new_val
                        
                    # Check if it improved
                    new_errors = []
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                s = 0.0
                                for rr in range(r):
                                    s += U[ii*m+jj][rr] * V[jj*p+kk][rr] * W[kk*n+ii][rr]
                                new_errors.append(abs(s - 1.0))
                    
                    new_max_err = max(new_errors)
                    if new_max_err < max_err:
                        pass # Keep change
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
