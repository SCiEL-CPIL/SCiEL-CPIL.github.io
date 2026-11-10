import math
import time

def construct():
    N = 255
    deadline = time.time() + 55.0
    
    # 1. Generate candidates
    # Numbers in [1, 255] divisible by 3, 5, or 17
    candidates = [i for i in range(1, N + 1) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n_cand = len(candidates)
    
    # 2. Build adjacency matrix
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Initial heuristic: Multiples of 3
    # This is a valid clique of size 85
    init_clique_indices = [i for i, c in enumerate(candidates) if c % 3 == 0]
    best_clique = list(init_clique_indices)
    
    # 4. Tomita's Max Clique Algorithm
    # To improve performance, we can order vertices by degree
    # However, the adjacency sets are already built. 
    # Reordering indices requires remapping adj.
    # For N=127, it's manageable.
    
    # Calculate degrees
    degrees = [len(adj[i]) for i in range(n_cand)]
    # Order indices by descending degree
    ordered_indices = sorted(range(n_cand), key=lambda x: degrees[x], reverse=True)
    
    # Remap adjacency to new order
    # new_adj[i] will be set of new indices
    # old_idx -> new_idx mapping
    old_to_new = {old: new for new, old in enumerate(ordered_indices)}
    new_adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in adj[i]:
            new_adj[old_to_new[i]].add(old_to_new[j])
            
    # Update candidates order for final result mapping
    # We need to map back from new indices to original values
    # new_idx -> old_idx -> value
    # ordered_indices[new_idx] = old_idx
    # candidates[old_idx] = value
    
    # Heuristic initial clique in new indexing
    # We need to find the indices of the multiples of 3 in the new order
    init_clique_new = []
    for i in range(n_cand):
        old_idx = ordered_indices[i]
        if candidates[old_idx] % 3 == 0:
            init_clique_new.append(i)
            
    best_clique = init_clique_new
    
    def tomita(P, X, clique):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P:
            if len(clique) > len(best_clique):
                best_clique = list(clique)
            return
            
        if len(clique) + len(P) <= len(best_clique):
            return
            
        # Pivot selection: u in P with max |P \cap N(u)|
        # Since P is small in deep recursion, this is fast
        u = max(P, key=lambda x: len(P & new_adj[x]))
        
        # Branch on P \ N(u)
        for v in list(P - new_adj[u]):
            tomita(P & new_adj[v], X | {v}, clique + [v])
            P.remove(v)
            X.add(v)
            
    tomita(set(range(n_cand)), set(), [])
    
    # Map back to original values
    result = []
    for idx in best_clique:
        old_idx = ordered_indices[idx]
        result.append(candidates[old_idx])
        
    # Ensure 255 is in result
    if 255 not in result:
        result.append(255)
        
    return result
