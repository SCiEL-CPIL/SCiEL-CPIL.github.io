"""Search-based exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [4, 4, 5]
_TIMEOUT = 115  # safety margin before 120s kill

def _time_left(deadline):
    return deadline - time.time()

def _tensor_values(U, V, W, n, m, p):
    """Compute the full tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    rank = len(U[0])
    T = np.zeros((n, m, p), dtype=float)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for c in range(rank):
        uc = Ua[:, c]
        vc = Va[:, c]
        wc = Wa[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    return T

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition exactly reproduces the matrix multiplication tensor."""
    T = _tensor_values(U, V, W, n, m, p)
    # The matrix multiplication tensor should have T[i,j,k] = delta[i,k] (1 if i==k, else 0)
    # Wait - let me reconsider. For matrix multiplication (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T[i,j,k] = 1 if this term contributes, which is always 1 for valid (i,j,k).
    # Actually the structure tensor of matrix multiplication has T[i,j,k] = 1 for all valid indices.
    # Let me verify: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For schoolbook: each (i,j,k) has exactly one c where all three are 1, so T[i,j,k] = 1.
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=tol)

def _rank(U):
    return len(U[0]) if U else 0

def _schoolbook(n, m, p):
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

def _greedy_build(n, m, p, deadline, max_rank=None, seed=42):
    """Greedy: try to build decomposition by adding terms one at a time."""
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    if max_rank is None:
        max_rank = n * m * p
    
    best_result = None
    best_rank = n * m * p
    
    for target_rank in range(max(10, max_rank - 30), max_rank + 1):
        if _time_left(deadline) < 2:
            break
        
        # Try multiple random attempts for this target rank
        for attempt in range(20):
            if _time_left(deadline) < 2:
                break
            
            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]
            W = [[0] * target_rank for _ in range(pn)]
            
            # Initialize with random sparse patterns
            for c in range(target_rank):
                # Randomly choose which entries are nonzero
                u_indices = set()
                v_indices = set()
                w_indices = set()
                
                # Each factor gets some random nonzero entries
                for _ in range(rng.randint(2, 5)):
                    u_indices.add(rng.randint(0, nm - 1))
                for _ in range(rng.randint(2, 5)):
                    v_indices.add(rng.randint(0, mp - 1))
                for _ in range(rng.randint(2, 5)):
                    w_indices.add(rng.randint(0, pn - 1))
                
                for idx in u_indices:
                    U[idx][c] = rng.choice([1, -1])
                for idx in v_indices:
                    V[idx][c] = rng.choice([1, -1])
                for idx in w_indices:
                    W[idx][c] = rng.choice([1, -1])
            
            # Verify
            if _is_valid(U, V, W, n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                return best_result
    
    return best_result

def _try_fixed_patterns(n, m, p, deadline):
    """Try known structured decompositions (e.g., based on Strassen-like ideas)."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try rank = n*m + m*p + p*n - n - p + 1 type bounds
    # For 4x4x5: lower bound considerations
    # Simple lower bound: max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # But actual lower bound is higher.
    
    best_result = None
    best_rank = n * m * p
    
    # Try various structured approaches
    for target_rank in range(30, min(60, n * m * p + 1)):
        if _time_left(deadline) < 2:
            break
        
        for seed in range(5):
            if _time_left(deadline) < 2:
                break
            
            rng = random.Random(seed + target_rank * 1000)
            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]
            W = [[0] * target_rank for _ in range(pn)]
            
            for c in range(target_rank):
                # Structured: each term covers a small set of indices
                ui = rng.sample(range(nm), rng.randint(1, 4))
                vi = rng.sample(range(mp), rng.randint(1, 4))
                wi = rng.sample(range(pn), rng.randint(1, 4))
                
                for idx in ui:
                    U[idx][c] = rng.choice([1, -1])
                for idx in vi:
                    V[idx][c] = rng.choice([1, -1])
                for idx in wi:
                    W[idx][c] = rng.choice([1, -1])
            
            if _is_valid(U, V, W, n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                return best_result
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + _TIMEOUT
    
    # Start with schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try greedy search
    result = _greedy_build(n, m, p, deadline, seed=0)
    if result is not None:
        r = _rank(result["U"])
        if r < best_rank:
            best_rank = r
            best_result = result
    
    # Try structured patterns
    if _time_left(deadline) > 5:
        result = _try_fixed_patterns(n, m, p, deadline)
        if result is not None:
            r = _rank(result["U"])
            if r < best_rank:
                best_rank = r
                best_result = result
    
    # Try more random attempts with different seeds
    if _time_left(deadline) > 5:
        for seed in [1, 2, 3, 4, 5, 10, 100, 42, 123, 999]:
            if _time_left(deadline) < 3:
                break
            result = _greedy_build(n, m, p, deadline, seed=seed)
            if result is not None:
                r = _rank(result["U"])
                if r < best_rank:
                    best_rank = r
                    best_result = result
    
    return best_result
# EVOLVE-BLOCK-END
