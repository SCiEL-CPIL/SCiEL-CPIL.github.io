"""Find rank-minimal tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _validate(U, V, W, n=3, m=3, p=5):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
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
    """Check entries are half-integers and decomposition is exact."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return _validate(U, V, W, n, m, p)

def _compute_all_sums(U, V, W, n, m, p, r):
    """Compute all tensor entry sums efficiently."""
    sums = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                sums[(i,j,k)] = s
    return sums

def _try_rank_fast(r, n=3, m=3, p=5, max_iters=20000, seed=42):
    """Try to find a rank-r decomposition using fast local search with incremental updates."""
    if _time_remaining() < 3.0:
        return None
    
    rng = random.Random(seed)
    
    value_sets = [
        [-1, 0, 1],
        [-1.5, -1, -0.5, 0, 0.5, 1, 1.5],
        [-2, -1, -0.5, 0, 0.5, 1, 2],
        [-1, 0, 1, 2],
    ]
    
    for values in value_sets:
        if _time_remaining() < 5.0:
            break
        
        n_restarts = max(5, min(50, int(50 * _time_remaining() / max(1, 20-r))))
        
        for restart in range(n_restarts):
            if _time_remaining() < 5.0:
                break
            
            # Random initialization
            U = [[rng.choice(values) for _ in range(r)] for _ in range(n*m)]
            V = [[rng.choice(values) for _ in range(r)] for _ in range(m*p)]
            W = [[rng.choice(values) for _ in range(r)] for _ in range(p*n)]
            
            # Compute initial sums
            sums = _compute_all_sums(U, V, W, n, m, p, r)
            max_err = max(abs(s - 1.0) for s in sums.values())
            
            if max_err < 1e-9:
                if _validate_half_integer(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                continue
            
            improved = True
            iters = 0
            while improved and iters < max_iters and _time_remaining() > 5.0:
                improved = False
                iters += 1
                
                # Try random entry changes with incremental update
                for _ in range(50):
                    if _time_remaining() < 5.0:
                        break
                    
                    mat_choice = rng.randint(0, 2)
                    if mat_choice == 0:
                        row = rng.randint(0, n*m-1)
                        col = rng.randint(0, r-1)
                        old_val = U[row][col]
                        new_val = rng.choice(values)
                        if new_val == old_val:
                            continue
                        delta = new_val - old_val
                        
                        # Compute new max_err using incremental update
                        # U[row, col] affects sums where i*m+j == row
                        i = row // m
                        j = row % m
                        new_max_err = max_err
                        for k in range(p):
                            old_s = sums[(i,j,k)]
                            new_s = old_s + delta * V[j*p+k][col] * W[k*n+i][col]
                            new_err = abs(new_s - 1.0)
                            if new_err > new_max_err:
                                new_max_err = new_err
                            sums[(i,j,k)] = new_s
                        
                        if new_max_err < max_err - 1e-12:
                            U[row][col] = new_val
                            max_err = new_max_err
                            improved = True
                            if max_err < 1e-9:
                                if _validate_half_integer(U, V, W, n, m, p):
                                    return {"U": U, "V": V, "W": W}
                                break
                        else:
                            # Revert
                            for k in range(p):
                                i2 = row // m
                                j2 = row % m
                                sums[(i2,j2,k)] -= delta * V[j2*p+k][col] * W[k*n+i2][col]
                    
                    elif mat_choice == 1:
                        row = rng.randint(0, m*p-1)
                        col = rng.randint(0, r-1)
                        old_val = V[row][col]
                        new_val = rng.choice(values)
                        if new_val == old_val:
                            continue
                        delta = new_val - old_val
                        
                        # V[row, col] affects sums where j*p+k == row
                        j = row // p
                        k = row % p
                        new_max_err = max_err
                        for i in range(n):
                            old_s = sums[(i,j,k)]
                            new_s = old_s + delta * U[i*m+j][col] * W[k*n+i][col]
                            new_err = abs(new_s - 1.0)
                            if new_err > new_max_err:
                                new_max_err = new_err
                            sums[(i,j,k)] = new_s
                        
                        if new_max_err < max_err - 1e-12:
                            V[row][col] = new_val
                            max_err = new_max_err
                            improved = True
                            if max_err < 1e-9:
                                if _validate_half_integer(U, V, W, n, m, p):
                                    return {"U": U, "V": V, "W": W}
                                break
                        else:
                            # Revert
                            for i in range(n):
                                sums[(i,j,k)] -= delta * U[i*m+j][col] * W[k*n+i][col]
                    
                    else:
                        row = rng.randint(0, p*n-1)
                        col = rng.randint(0, r-1)
                        old_val = W[row][col]
                        new_val = rng.choice(values)
                        if new_val == old_val:
                            continue
                        delta = new_val - old_val
                        
                        # W[row, col] affects sums where k*n+i == row
                        k = row // n
                        i = row % n
                        new_max_err = max_err
                        for j in range(m):
                            old_s = sums[(i,j,k)]
                            new_s = old_s + delta * U[i*m+j][col] * V[j*p+k][col]
                            new_err = abs(new_s - 1.0)
                            if new_err > new_max_err:
                                new_max_err = new_err
                            sums[(i,j,k)] = new_s
                        
                        if new_max_err < max_err - 1e-12:
                            W[row][col] = new_val
                            max_err = new_max_err
                            improved = True
                            if max_err < 1e-9:
                                if _validate_half_integer(U, V, W, n, m, p):
                                    return {"U": U, "V": V, "W": W}
                                break
                        else:
                            # Revert
                            for j in range(m):
                                sums[(i,j,k)] -= delta * U[i*m+j][col] * V[j*p+k][col]
            
            if max_err < 1e-9:
                if _validate_half_integer(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    return None

def _try_structured(r, n=3, m=3, p=5):
    """Try structured patterns for low-rank decompositions."""
    if _time_remaining() < 3.0:
        return None
    
    # Try: use rank-r with specific structure
    # For each rank, try to build U, V, W with many zeros
    rng = random.Random(12345)
    
    for attempt in range(100):
        if _time_remaining() < 3.0:
            break
        
        # Initialize with zeros
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # Assign non-zero entries randomly
        for rr in range(r):
            # Pick a few entries to be non-zero in each matrix
            for _ in range(rng.randint(1, 3)):
                idx = rng.randint(0, n*m-1)
                U[idx][rr] = rng.choice([-1, 0.5, 1, 1.5])
            for _ in range(rng.randint(1, 3)):
                idx = rng.randint(0, m*p-1)
                V[idx][rr] = rng.choice([-1, 0.5, 1, 1.5])
            for _ in range(rng.randint(1, 3)):
                idx = rng.randint(0, p*n-1)
                W[idx][rr] = rng.choice([-1, 0.5, 1, 1.5])
        
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = 3, 3, 5
    
    # Build schoolbook as fallback (rank 45)
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
    
    # Try ranks from 15 upward (lower bound is 15)
    for r in range(15, 46):
        if _time_remaining() < 8.0:
            break
        
        # Try structured patterns first
        sol = _try_structured(r, n, m, p)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
        
        # Try fast random/local search with multiple seeds
        found = False
        for s in [r, 100+r, 200+r, 300+r, 400+r]:
            if _time_remaining() < 8.0:
                break
            sol = _try_rank_fast(r, n, m, p, max_iters=50000, seed=s)
            if sol is not None:
                best_rank = r
                best_sol = sol
                found = True
                break
        
        if found:
            break
    
    return best_sol
