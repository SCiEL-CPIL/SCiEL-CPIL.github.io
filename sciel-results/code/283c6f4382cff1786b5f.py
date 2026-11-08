"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
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

def _validate(U, V, W, n, m, p):
    """Check if decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta(j==j) for all i,j,k"""
    rank = len(U[0])
    # Build matrices
    Um = np.array(U, dtype=float)  # (n*m, rank)
    Vm = np.array(V, dtype=float)  # (m*p, rank)
    Wm = np.array(W, dtype=float)  # (p*n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if True else 0.0  # C[i,k] = sum_j A[i,j]B[j,k], so each (i,j,k) contributes A[i,j]*B[j,k]
                # Actually the tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_reduction(n, m, p, max_rank, deadline):
    """Try to find a decomposition with rank < n*m*p using random search + local optimization."""
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    target_ranks = list(range(80, best_rank + 1))  # try from lower ranks up
    random.shuffle(target_ranks)
    target_ranks.sort()  # try lower ranks first
    
    for rank in target_ranks:
        if rank >= best_rank:
            break
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations for this rank
        max_iters = 500
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            # Random integer/half-integer initialization
            U = np.random.randint(-2, 3, size=(n*m, rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m*p, rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p*n, rank)) * 0.5
            
            # Local search: try to fix residuals
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Compute residuals for all (i,j,k)
                residual = 0.0
                worst_ijk = None
                worst_val = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = abs(s - 1.0)
                            if err > worst_val:
                                worst_val = err
                                worst_ijk = (i, j, k)
                
                if worst_val < 1e-9:
                    # Found valid decomposition!
                    best_rank = rank
                    best = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
                
                # Try to improve: pick a random column and adjust
                i, j, k = worst_ijk
                c = random.randint(0, rank - 1)
                
                # Try small adjustments
                for delta in [-0.5, 0.5]:
                    old_U = U[i*m+j, c]
                    U[i*m+j, c] += delta
                    # Check if this helps
                    s = 0.0
                    for cc in range(rank):
                        s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                    new_err = abs(s - 1.0)
                    if new_err < worst_val:
                        # Accept
                        pass
                    else:
                        U[i*m+j, c] = old_U
                
                # Also try adjusting V and W
                c = random.randint(0, rank - 1)
                for delta in [-0.5, 0.5]:
                    old_V = V[j*p+k, c]
                    V[j*p+k, c] += delta
                    s = 0.0
                    for cc in range(rank):
                        s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                    new_err = abs(s - 1.0)
                    if new_err >= worst_val:
                        V[j*p+k, c] = old_V
                
                c = random.randint(0, rank - 1)
                for delta in [-0.5, 0.5]:
                    old_W = W[k*n+i, c]
                    W[k*n+i, c] += delta
                    s = 0.0
                    for cc in range(rank):
                        s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                    new_err = abs(s - 1.0)
                    if new_err >= worst_val:
                        W[k*n+i, c] = old_W
                
                if worst_val < 1e-9:
                    break
        
        if best_rank == rank:
            break
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin for 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to reduce rank
    result, rank = _try_rank_reduction(n, m, p, best_rank, deadline)
    if rank < best_rank:
        best = result
        best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
