import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 8]

def _verify_decomposition(n, m, p, U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of M_{n,m,p}."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # For each (i, j, k), the tensor entry T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # should equal 1 if i,j,k correspond to a valid multiplication term, else 0.
    # Actually, the matrix multiplication tensor: T[i,j,k] = 1 if the output C[i,k] gets contribution
    # from A[i,j]*B[j,k], so T[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
    # Wait, let me re-read: the tensor for matrix multiplication M_{n,m,p} has entries
    # T[i,j,k] = 1 if (i,j,k) is a valid index triple, i.e., the standard multiplication tensor.
    # For multiplying (n x m) A by (m x p) B to get (n x p) C:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The decomposition: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for t in range(rank):
                    actual += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(actual - expected) > tol:
                    return False
    return True

def _strassen_28(n, m, p):
    """Return the rank-28 Strassen decomposition."""
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):
        k1 = 2 * q
        for r, prod in enumerate(strassen_terms):
            t = q * 7 + r
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
                
    return U, V, W

def _search_low_rank(n, m, p, max_rank=28, time_budget=100.0):
    """Try to find a lower-rank decomposition than 28."""
    start = time.time()
    
    # Try rank 24, 25, 26, 27
    for target_rank in range(24, max_rank):
        if time.time() - start > time_budget:
            break
        # Use random search with small integer/half-integer entries
        # This is a heuristic; we'll try a limited number of random initializations
        np.random.seed(42)
        found = False
        for trial in range(50):
            if time.time() - start > time_budget:
                break
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Initialize with small values
            for t in range(target_rank):
                U[:, t] = np.random.choice([-1, 0, 1], size=n*m) * np.random.choice([0.5, 1, 1.5, 2], size=n*m)
                V[:, t] = np.random.choice([-1, 0, 1], size=m*p) * np.random.choice([0.5, 1, 1.5, 2], size=m*p)
                W[:, t] = np.random.choice([-1, 0, 1], size=p*n) * np.random.choice([0.5, 1, 1.5, 2], size=p*n)
            
            # Quick check: compute the implied tensor and compare
            # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
            # We need T[i,j,k] = 1 for all i,j,k
            # This is a system of equations. Let's check if it's close.
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(target_rank))
                        errors.append(abs(val - 1.0))
            max_err = max(errors)
            if max_err < 1e-8:
                return U, V, W, target_rank
            # If we're getting close, try a local refinement
            if max_err < 1.0 and trial % 10 == 0:
                # Simple gradient-free local search
                for _ in range(100):
                    if time.time() - start > time_budget:
                        break
                    # Perturb one random entry
                    idx = np.random.randint(0, target_rank)
                    mat = np.random.choice(['U', 'V', 'W'])
                    row = np.random.randint(0, (n*m if mat=='U' else m*p if mat=='V' else p*n))
                    old_val = (U if mat=='U' else V if mat=='V' else W)[row, idx]
                    new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    (U if mat=='U' else V if mat=='V' else W)[row, idx] = new_val
                    
                    # Check error
                    max_err_new = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(target_rank))
                                max_err_new = max(max_err_new, abs(val - 1.0))
                    if max_err_new < max_err:
                        max_err = max_err_new
                        if max_err < 1e-8:
                            return U, V, W, target_rank
                    else:
                        (U if mat=='U' else V if mat=='V' else W)[row, idx] = old_val
    
    return None, None, None, None

def construct():
    start_time = time.time()
    deadline = 150.0  # 150 seconds wall-clock limit, use 140 as safety margin
    
    n, m, p = DIMENSIONS
    
    # First, get the known rank-28 solution
    U28, V28, W28 = _strassen_28(n, m, p)
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try to find a lower-rank solution
    time_budget = min(140.0 - (time.time() - start_time), 130.0)
    if time_budget > 10:
        U, V, W, rank = _search_low_rank(n, m, p, max_rank=28, time_budget=time_budget)
        if U is not None:
            best_U, best_V, best_W = U, V, W
            best_rank = rank
    
    # Safety check
    if time.time() - start_time > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
