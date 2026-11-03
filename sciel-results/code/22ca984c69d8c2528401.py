"""Matrix multiplication 2x3x6: rank-minimal exact tensor decomposition search.

Key structural insight: For fixed (j,k), the tensor pattern across i is identical.
T[i*m+j, j*p+k, k*n+i] = 1 for all i. This means W[k*n+i, r] must be the same
for all i when the term corresponds to a specific (j,k). This gives an upper
bound of rank m*p = 18.

We search for lower ranks using:
1. The structured m*p construction (rank 18)
2. Randomized search with the W-consistency constraint
3. Systematic structured search
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE_SECONDS = 135.0

def _target_tensor(n, m, p):
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _target_tensor(n, m, p)
    return np.allclose(T_recon, T_target, atol=tol)

def _structured_mp(n, m, p):
    """Construct rank m*p decomposition using the structural insight.
    One term per (j,k) pair."""
    rank = m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U[i * m + j, r] = 1
            V[j * p + k, r] = 1
            for i in range(n):
                W[k * n + i, r] = 1
    return U, V, W

def _try_random_rank(n, m, p, r, n_trials=1000, seed=None):
    """Try to find a rank-r decomposition with the W-consistency constraint.
    W[k*n+i, r_idx] is the same for all i (structural constraint).
    This reduces the search space significantly."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    nm, mp, pn = n * m, m * p, p * n
    # W has pn = p*n rows, but constrained: W[k*n+i, r_idx] same for all i
    # So effectively W is defined by p*n/rank groups... actually p groups of n identical rows
    # Let's define W_base of shape (p, r) and expand
    
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for trial in range(n_trials):
        # Random U: shape (nm, r)
        U = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(nm)])
        # Random V: shape (mp, r)
        V = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(mp)])
        # W_base: shape (p, r) - will be expanded to (pn, r)
        W_base = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(p)])
        W = np.zeros((pn, r))
        for k in range(p):
            for i in range(n):
                W[k * n + i] = W_base[k]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_structured_patterns(n, m, p, r):
    """Try various structured patterns for rank r."""
    nm, mp, pn = n * m, m * p, p * n
    
    # Try: assign each (j,k) to one of r groups, with possible sharing
    # For each (j,k), choose which rank index it uses
    # And for each rank index, the U, V, W patterns are determined
    
    # Simple approach: try all assignments of (j,k) pairs to r slots
    # This is (r)^(m*p) possibilities - too many for r > 1
    
    # Instead, try a few structured patterns:
    # Pattern 1: All (j,k) map to distinct slots (rank = m*p)
    # Pattern 2: Group by j (rank = m)
    # Pattern 3: Group by k (rank = p)
    # Pattern 4: Group by (j mod a, k mod b)
    
    patterns = []
    
    # Pattern: group by j
    if r >= m:
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        for j in range(m):
            r_idx = j
            for i in range(n):
                U[i * m + j, r_idx] = 1
            for k in range(p):
                V[j * p + k, r_idx] = 1
            for k in range(p):
                for i in range(n):
                    W[k * n + i, r_idx] = 1
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    # Pattern: group by k
    if r >= p:
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        for k in range(p):
            r_idx = k
            for j in range(m):
                for i in range(n):
                    U[i * m + j, r_idx] = 1
                V[j * p + k, r_idx] = 1
            for i in range(n):
                W[k * n + i, r_idx] = 1
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    # Pattern: group by (j, k) with some sharing - try random assignments
    rng = random.Random(42)
    for _ in range(100):
        # Assign each (j,k) to a random slot in [0, r)
        assign = {}
        for j in range(m):
            for k in range(p):
                assign[(j, k)] = rng.randint(0, r - 1)
        
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        for (j, k), r_idx in assign.items():
            for i in range(n):
                U[i * m + j, r_idx] += 1
            V[j * p + k, r_idx] += 1
            for i in range(n):
                W[k * n + i, r_idx] += 1
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def elapsed():
        return time.time() - start_time
    
    def time_up():
        return elapsed() > DEADLINE_SECONDS
    
    # Start with the structural rank m*p = 18 decomposition
    U_best, V_best, W_best = _structured_mp(n, m, p)
    best_rank = m * p
    best_result = {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n, m, p) = 6, but more realistically at least p = 6
    # Actually, the rank must be at least the max of the slice ranks
    # Slice rank fixing i: rank of matrix with entries T[i*m+j, j*p+k, k*n+i] over (j,k)
    # For fixed i, we have an m x p matrix of all 1s, rank 1
    # So slice rank lower bound is 1... but that's not tight
    
    # Better lower bound: the tensor has m*p = 18 "diagonal" entries in a sense
    # Let's just search from rank 6 upward
    
    # First try structured patterns for each rank
    for r in range(6, best_rank):
        if time_up():
            break
        
        result = _try_structured_patterns(n, m, p, r)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = r
            best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            break
    
    # If we still have rank > some threshold, do random search
    if best_rank > 6 and not time_up():
        for r in range(6, best_rank):
            if time_up():
                break
            
            # Do random search with the W-consistency constraint
            # Use more trials for lower ranks
            n_trials = 2000 if r <= 10 else 500
            result = _try_random_rank(n, m, p, r, n_trials=n_trials)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = r
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                break
    
    # If time remains, do more extensive random search at the current best rank
    # to see if we can find an even lower rank
    if not time_up() and best_rank > 6:
        # Try random search for rank best_rank - 1, best_rank - 2, etc.
        for r in range(max(6, best_rank - 5), best_rank):
            if time_up():
                break
            n_trials = 1000
            result = _try_random_rank(n, m, p, r, n_trials=n_trials, seed=r * 1000)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = r
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return best_result
