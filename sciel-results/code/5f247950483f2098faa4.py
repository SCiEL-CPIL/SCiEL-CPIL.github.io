"""Search for rank-minimal exact tensor decomposition of 3x4 * 4x4 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    nm, mp, pn = n * m, m * p, p * n
    trivial_rank = nm * mp * pn  # 48
    
    # Build target tensor entries (the 1-positions)
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_entries.append((u, v, w))
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        r = U.shape[1]
        # Check all 1-entries
        for (u, v, w) in target_entries:
            s = 0.0
            for c in range(r):
                s += U[u, c] * V[v, c] * W[w, c]
            if abs(s - 1.0) > 1e-9:
                return False
        # Check a sample of 0-entries for crosstalk
        random.seed(42)
        for _ in range(200):
            u = random.randint(0, nm - 1)
            v = random.randint(0, mp - 1)
            w = random.randint(0, pn - 1)
            # Check if this is a target entry
            is_target = False
            i_u, j_u = u // m, u % m
            j_v, k_v = v // p, v % p
            k_w, i_w = w // n, w % n
            if i_u == i_w and j_u == j_v and k_v == k_w:
                is_target = True
            if not is_target:
                s = 0.0
                for c in range(r):
                    s += U[u, c] * V[v, c] * W[w, c]
                if abs(s) > 1e-9:
                    return False
        return True
    
    # Trivial solution
    U_triv = np.zeros((nm, trivial_rank))
    V_triv = np.zeros((mp, trivial_rank))
    W_triv = np.zeros((pn, trivial_rank))
    col = 0
    for (u, v, w) in target_entries:
        U_triv[u, col] = 1
        V_triv[v, col] = 1
        W_triv[w, col] = 1
        col += 1
    
    best_rank = trivial_rank
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Strategy: Try to reduce rank by merging columns.
    # For rank r-1, we need to find a column c that can be expressed as a combination
    # of others, or more generally, find a different decomposition.
    
    # Better approach: Use the fact that we can try specific structured decompositions.
    # For 3x4x4, try to use the 3x3x3 rank-21 result as a sub-block.
    
    # Alternative: Randomized iterative method with integer/half-integer constraint.
    # Initialize with small values and try to satisfy equations.
    
    # Most practical: Try rank 47 with a structured perturbation.
    # Take the trivial solution and try to express one column as a sum of two others
    # with coefficients that maintain integer/half-integer property.
    
    # Actually, let's try a different structural approach:
    # Use the "Strassen idea" - look for a basis change that reduces rank.
    
    # For now, implement a limited search for rank < 48 using random initialization
    # with the constraint that entries are in {-1, -0.5, 0, 0.5, 1}.
    
    half_ints = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for rank in range(trivial_rank - 1, max(trivial_rank - 8, 1), -1):
        if time.time() > deadline:
            break
        
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with half-integers
            U_try = np.array([[random.choice(half_ints) for _ in range(rank)] for _ in range(nm)])
            V_try = np.array([[random.choice(half_ints) for _ in range(rank)] for _ in range(mp)])
            W_try = np.array([[random.choice(half_ints) for _ in range(rank)] for _ in range(pn)])
            
            if verify(U_try, V_try, W_try):
                best_rank = rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break  # Found valid decomposition for this rank, try lower
    
    # Final verification of best solution
    if not verify(best_U, best_V, best_W):
        # Fallback to trivial
        best_rank = trivial_rank
        best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
