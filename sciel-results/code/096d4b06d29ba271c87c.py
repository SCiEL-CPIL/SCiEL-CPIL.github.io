"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    es = set()
    for e in P['edges']:
        es.add(tuple(e))
        es.add(tuple(reversed(e)))
    
    # Adjacency as sets
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    # Precompute is_adj matrix for fast lookup
    is_adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            is_adj[i][j] = True
    
    deadline = time.time() + 55  # safety margin before 90s kill
    
    # Order vertices by degree descending (more constrained first)
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    
    def can_add_to_clique(part, v):
        """Check if v can be added to part as a clique"""
        for u in part:
            if not is_adj[u][v]:
                return False
        return True
    
    def can_add_to_independent(part, v):
        """Check if v can be added to part as an independent set"""
        for u in part:
            if is_adj[u][v]:
                return False
        return True
    
    best_parts = None
    best_count = n + 1  # worst case: n parts of size 1
    
    # Try to find a partition with k parts for k = 1, 2, ...
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        if k >= best_count:
            break
        
        # For k parts, each can be type 0 (clique) or 1 (independent)
        # Generate all type combinations
        type_combos = list(product([0, 1], repeat=k))
        
        # Sort type combinations by a heuristic: prefer more cliques if graph is dense, more independent if sparse
        # For now, just try all
        found = False
        for types in type_combos:
            if time.time() > deadline:
                break
            
            parts = [[] for _ in range(k)]
            success = False
            
            def backtrack(idx):
                nonlocal success
                if time.time() > deadline:
                    return False
                if idx == len(order):
                    success = True
                    return True
                v = order[idx]
                
                for part_idx in range(k):
                    t = types[part_idx]
                    if t == 0:  # clique
                        if can_add_to_clique(parts[part_idx], v):
                            parts[part_idx].append(v)
                            if backtrack(idx + 1):
                                return True
                            parts[part_idx].pop()
                    else:  # independent
                        if can_add_to_independent(parts[part_idx], v):
                            parts[part_idx].append(v)
                            if backtrack(idx + 1):
                                return True
                            parts[part_idx].pop()
                return False
            
            if backtrack(0):
                # Collect non-empty parts
                result = []
                for p in parts:
                    if p:
                        result.append(list(p))
                if result and len(result) < best_count:
                    best_parts = result
                    best_count = len(result)
                found = True
                break
        
        # If we found a solution with k parts, we're done (k is minimal)
        if found:
            break
    
    if best_parts is None:
        # Fallback: each vertex its own part
        best_parts = [[v] for v in range(n)]
    
    # Ensure all parts are non-empty and return as list of lists
    final = []
    for p in best_parts:
        if p:
            final.append(list(p))
    
    return final

# EVOLVE-BLOCK-END
