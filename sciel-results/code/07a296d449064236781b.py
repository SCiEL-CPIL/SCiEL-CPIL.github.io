import math
import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 255
    target = 255
    
    # 1. Generate candidate pool: numbers in [1, 255] that share a factor with 255
    # Factors of 255 are 3, 5, 17
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, target) > 1:
            candidates.append(i)
            
    # Ensure 255 is in the list (it should be, gcd(255, 255)=255)
    if target not in candidates:
        candidates.append(target)
        
    # Map values to indices for graph representation
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    m = len(candidates)
    
    # 2. Build adjacency list for the "non-coprime" graph
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Bron-Kerbosch with pivot to find maximum clique
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
            
        # Pivot selection: choose u in P union X that maximizes |N(u) intersect P|
        P_union_X = P.union(X)
        if not P_union_X:
            return
            
        max_count = -1
        pivot = None
        for u in P_union_X:
            count = len(P.intersection(adj[u]))
            if count > max_count:
                max_count = count
                pivot = u
                
        # Recurse on P \ N(pivot)
        # We iterate over candidates in P that are NOT neighbors of pivot
        # If P is empty, we already returned.
        # N(pivot) is adj[pivot]
        candidates_to_recurse = P.difference(adj[pivot])
        
        for v in list(candidates_to_recurse):
            bron_kerbosch(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
            P.remove(v)
            X.add(v)
            
            if time.time() > deadline:
                return

    # Initialize
    R = set()
    P = set(range(m))
    X = set()
    
    bron_kerbosch(R, P, X)
    
    # If no clique found (should not happen), fallback to multiples of 3
    if not best_clique:
        return [i for i in candidates if i % 3 == 0]
        
    # Convert indices back to values
    result_set = set(candidates[i] for i in best_clique)
    
    # 4. Local search: try to add elements from the full range [1,255] not in result
    # that are compatible with all current members
    def is_compatible_with_all(val, current_set):
        for v in current_set:
            if math.gcd(val, v) == 1:
                return False
        return True
    
    # Try adding any number from 1 to 255 not already in result
    for val in range(1, 256):
        if val not in result_set and is_compatible_with_all(val, result_set):
            result_set.add(val)
    
    # 5. Local search: try single swaps (remove one, add one or more)
    improved = True
    while improved:
        improved = False
        current_size = len(result_set)
        
        # Try removing each element and see if we can add more
        for remove_val in list(result_set):
            if remove_val == target:
                continue  # Don't remove 255
            trial_set = result_set - {remove_val}
            # Try to add all compatible elements
            for val in range(1, 256):
                if val not in trial_set and is_compatible_with_all(val, trial_set):
                    trial_set.add(val)
            if len(trial_set) > current_size:
                result_set = trial_set
                improved = True
                break
        if time.time() > deadline:
            break
    
    # Ensure 255 is in the result
    if target not in result_set:
        result_set.add(target)
    
    return list(result_set)

# EVOLVE-BLOCK-END
