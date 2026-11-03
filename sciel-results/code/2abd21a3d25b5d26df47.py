"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def construct():
    deadline = time.time() + 110
    
    # Generate shells
    shell4 = _generate_shell(4)
    shell2 = _generate_shell(2)
    
    n4 = len(shell4)
    n2 = len(shell2)
    
    # Precompute inner products for Shell 4
    # dots4 = shell4 @ shell4.T
    # Conflict if dots4[i,j] > 2.
    # Note: diagonal is 4.
    
    # To save memory and time, we can compute on the fly or use the matrix.
    # 5280 * 5280 * 8 bytes = ~223 MB. This might be too high for some environments, but usually OK.
    # Let's try to be memory efficient by using int8 or bool.
    # Inner products range from -4 to 4.
    # We only care if > 2.
    
    # Let's use the matrix for fast access in local search.
    dots4 = shell4 @ shell4.T
    # Conflict matrix: True if conflict (ip > 2)
    # We want independent set, so we want NO conflicts.
    # Conflict[i, j] = dots4[i, j] > 2
    # Note: i != j.
    
    # We will run multiple restarts of Greedy + Local Search.
    
    best_S4 = np.empty((0, D), dtype=np.int64)
    best_k4 = 0
    
    # Number of restarts
    # Each restart: O(N^2) for greedy? No, O(N * K) for vectorized greedy.
    # Local search: O(K * N * K) ?
    
    # Let's estimate time.
    # N=5280.
    # Greedy: 5280 iterations. Each dot product with current vector: O(N). Total O(N^2) ~ 27M ops. Fast in numpy.
    # Local search: For each u in S (K~500), find compatible v's.
    # Compatible with S\{u} means no conflict with any s in S where s != u.
    # We can precompute for each candidate v, the list of conflicts with S?
    
    # Let's implement a simpler Local Search:
    # 1. Get S via Greedy.
    # 2. Build a "compatibility mask" for S.
    # 3. Try to swap.
    
    # Given the complexity, let's start with **Multiple Randomized Greedy Restarts** first.
    # This is a proven way to beat single greedy.
    
    num_restarts = 50 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Shuffle
        idx = np.arange(n4)
        np.random.shuffle(idx)
        shell4_shuffled = shell4[idx]
        
        # Vectorized Greedy
        # We maintain a list of selected indices in the ORIGINAL array?
        # Or just work with the shuffled array and map back?
        # Working with shuffled array is easier for the greedy loop.
        
        selected_indices = []
        alive = np.ones(n4, dtype=bool)
        max_ip = np.full(n4, -np.inf)
        
        for i in range(n4):
            if not alive[i]:
                continue
            if max_ip[i] > 2:
                alive[i] = False
                continue
                
            selected_indices.append(i)
            alive[i] = False
            
            # Update max_ip
            # dots = shell4_shuffled @ shell4_shuffled[i]
            # This is O(N).
            # We can optimize by only updating alive indices?
            # max_ip[alive] = np.maximum(max_ip[alive], shell4_shuffled[alive] @ shell4_shuffled[i])
            
            dots = shell4_shuffled @ shell4_shuffled[i]
            max_ip = np.maximum(max_ip, dots)
            alive = max_ip <= 2
            
            if not np.any(alive):
                break
        
        S4_current = shell4_shuffled[selected_indices]
        k_current = len(S4_current)
        
        if k_current > best_k4:
            best_k4 = k_current
            best_S4 = S4_current.copy()
            
            # Optional: Local Search on best_S4?
            # Let's add a simple local search step here.
            # Try to improve best_S4.
            
            # Map best_S4 back to original indices? 
            # It's easier to just keep the vectors.
            # But for local search, we need to check against the whole pool.
            
            # Let's implement a "Swap" local search on the set of vectors.
            # 1. Identify conflicts in the full pool with respect to S4_current.
            # 2. For each u in S4_current, find v1, v2 not in S4_current such that:
            #    a. v1, v2 compatible with all s in S4_current \ {u}
            #    b. v1, v2 compatible with each other
            #    If found, replace u with v1, v2.
            
            # This is complex to vectorize efficiently in a short time.
            # Let's stick to multiple restarts first. If time permits, add local search.
            
    # Now select S2
    S4 = best_S4
    if len(S4) > 0:
        dots_24 = shell2 @ S4.T
        max_ip_24 = dots_24.max(axis=1)
        valid_2_mask = max_ip_24 <= 1
        shell2_filtered = shell2[valid_2_mask]
    else:
        shell2_filtered = shell2
    
    n2f = len(shell2_filtered)
    
    # Greedy for S2
    # Constraint: pairwise ip <= 0 (if S4 non-empty, M^2=4)
    # If S4 empty, M^2=2, constraint ip <= 1.
    # Assume S4 non-empty.
    
    if len(S4) == 0:
        max_ip_bound = 1
    else:
        max_ip_bound = 0
        
    # Run a few restarts for S2 as well? S2 is small (220), so greedy is fast.
    best_S2 = np.empty((0, D), dtype=np.int64)
    best_k2 = 0
    
    for r in range(20):
        idx2 = np.arange(n2f)
        np.random.shuffle(idx2)
        shell2_shuffled = shell2_filtered[idx2]
        
        selected2 = []
        alive2 = np.ones(n2f, dtype=bool)
        max_ip2 = np.full(n2f, -np.inf)
        
        for i in range(n2f):
            if not alive2[i]:
                continue
            if max_ip2[i] > max_ip_bound:
                alive2[i] = False
                continue
                
            selected2.append(i)
            alive2[i] = False
            
            dots = shell2_shuffled @ shell2_shuffled[i]
            max_ip2 = np.maximum(max_ip2, dots)
            alive2 = max_ip2 <= max_ip_bound
            
            if not np.any(alive2):
                break
        
        S2_current = shell2_shuffled[selected2]
        if len(S2_current) > best_k2:
            best_k2 = len(S2_current)
            best_S2 = S2_current.copy()
            
    S2 = best_S2
    
    if len(S4) > 0 and len(S2) > 0:
        result = np.vstack([S4, S2])
    elif len(S4) > 0:
        result = S4
    elif len(S2) > 0:
        result = S2
    else:
        # Fallback to seed-like
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        vectors.append(v)
        result = np.array(vectors, dtype=np.int64)
        
    return result

# EVOLVE-BLOCK-END
