import time
from math import gcd

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = 85.0  # Keep 5s margin for safety
    
    # Candidate pool: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    # x must be divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Ensure 255 is in candidates
    if 255 not in candidates:
        candidates.append(255)
        
    n = len(candidates)
    idx_map = {val: i for i, val in enumerate(candidates)}
    
    # Build adjacency list using bitmasks for efficiency
    adj_mask = [0] * n
    for i in range(n):
        vi = candidates[i]
        mask = 0
        for j in range(n):
            if i != j and gcd(vi, candidates[j]) > 1:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # Convert to sets for Bron-Kerbosch
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (adj_mask[i] >> j) & 1:
                adj[i].add(j)
    
    # Bron-Kerbosch algorithm with pivot to find maximum clique
    best_clique = []
    best_size = 0
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Pivot selection: choose u in P union X with maximum degree in P
        pivot = None
        max_degree = -1
        # Precompute P as a set for faster intersection
        P_set = P
        for u in P_set:
            deg = len(adj[u] & P_set)
            if deg > max_degree:
                max_degree = deg
                pivot = u
        for u in X:
            deg = len(adj[u] & P_set)
            if deg > max_degree:
                max_degree = deg
                pivot = u
        
        if pivot is not None:
            # Recurse on P \ N(pivot)
            candidates_for_recursion = P_set - adj[pivot]
            for v in list(candidates_for_recursion):
                bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
                P_set.remove(v)
                X.add(v)
                if time.time() > deadline:
                    return
        else:
            for v in list(P_set):
                bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
                P_set.remove(v)
                X.add(v)
                if time.time() > deadline:
                    return

    # Start with empty R, all candidates in P, empty X
    # Since 255 is a universal vertex, it will be part of the max clique
    # We can fix 255 in the clique to reduce search space
    idx_255 = idx_map[255]
    
    # Initialize with a greedy clique containing 255
    greedy_clique = {idx_255}
    remaining = set(range(n)) - {idx_255}
    while remaining:
        # Pick vertex with max degree in remaining that is adjacent to all in greedy_clique
        best_v = None
        best_deg = -1
        for v in remaining:
            if all((adj_mask[v] >> u) & 1 for u in greedy_clique):
                deg = len(adj[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        if best_v is None:
            break
        greedy_clique.add(best_v)
        remaining.discard(best_v)
    
    if len(greedy_clique) > best_size:
        best_size = len(greedy_clique)
        best_clique = list(greedy_clique)
    
    # Run Bron-Kerbosch
    P = set(range(n))
    X = set()
    R = set()
    
    bron_kerbosch(R, P, X)
    
    # If best_clique is empty (should not happen), fallback
    if not best_clique:
        # Fallback to multiples of 3
        result_vals = [i for i in candidates if i % 3 == 0]
        if 255 not in result_vals:
            result_vals.append(255)
        return result_vals
    
    result_vals = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in the result (it should be, but just in case)
    if 255 not in result_vals:
        result_vals.append(255)
        
    # Final verification
    valid = True
    for i in range(len(result_vals)):
        for j in range(i+1, len(result_vals)):
            if gcd(result_vals[i], result_vals[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to known good set
        result_vals = [i for i in candidates if i % 3 == 0]
        if 255 not in result_vals:
            result_vals.append(255)

    return result_vals
