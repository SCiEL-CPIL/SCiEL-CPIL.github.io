"""Rewritten program for cochromatic partition of a pinned random graph."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    
    # Helper: check if a set of vertices forms a clique
    def is_clique(vertices):
        vlist = list(vertices)
        for i in range(len(vlist)):
            for j in range(i + 1, len(vlist)):
                if vlist[j] not in adj[vlist[i]]:
                    return False
        return True
    
    # Helper: check if a set of vertices forms an independent set
    def is_independent(vertices):
        vlist = list(vertices)
        for i in range(len(vlist)):
            for j in range(i + 1, len(vlist)):
                if vlist[j] in adj[vlist[i]]:
                    return False
        return True
    
    # Precompute all maximal cliques and maximal independent sets
    # For n=14, we can enumerate all subsets of size up to n
    all_cliques = []
    all_independent = []
    
    # Enumerate all subsets
    for mask in range(1, 1 << n):
        vertices = set()
        for v in range(n):
            if mask & (1 << v):
                vertices.add(v)
        
        if is_clique(vertices):
            all_cliques.append(vertices)
        if is_independent(vertices):
            all_independent.append(vertices)
    
    # Find maximal cliques (not contained in any other clique)
    maximal_cliques = []
    for i, c1 in enumerate(all_cliques):
        is_maximal = True
        for j, c2 in enumerate(all_cliques):
            if i != j and c1.issubset(c2):
                is_maximal = False
                break
        if is_maximal:
            maximal_cliques.append(c1)
    
    # Find maximal independent sets
    maximal_independent = []
    for i, c1 in enumerate(all_independent):
        is_maximal = True
        for j, c2 in enumerate(all_independent):
            if i != j and c1.issubset(c2):
                is_maximal = False
                break
        if is_maximal:
            maximal_independent.append(c1)
    
    # Combine all candidate parts (maximal cliques and maximal independent sets)
    # Also include single vertices as fallback
    candidate_parts = list(maximal_cliques) + list(maximal_independent)
    for v in range(n):
        candidate_parts.append({v})
    
    # Remove duplicates
    unique_parts = []
    seen = set()
    for part in candidate_parts:
        key = frozenset(part)
        if key not in seen:
            seen.add(key)
            unique_parts.append(part)
    
    # Sort by size descending (prefer larger parts)
    unique_parts.sort(key=len, reverse=True)
    
    # Use backtracking search to find minimum cochromatic partition
    best_solution = None
    best_count = n + 1  # Upper bound
    
    # State: set of unassigned vertices, list of parts used so far
    def backtrack(unassigned, current_parts, depth):
        nonlocal best_solution, best_count
        
        # Deadline check
        if time.time() > deadline:
            return
        
        # Pruning: if current parts already exceed best, stop
        if depth >= best_count:
            return
        
        if not unassigned:
            # Found a valid partition
            if depth < best_count:
                best_count = depth
                best_solution = [list(p) for p in current_parts]
            return
        
        # Try each candidate part
        for part in unique_parts:
            # Check if part is subset of unassigned
            if part.issubset(unassigned):
                # Check if part is clique or independent (already guaranteed by construction)
                new_unassigned = unassigned - part
                new_parts = current_parts + [part]
                backtrack(new_unassigned, new_parts, depth + 1)
                
                # Deadline check
                if time.time() > deadline:
                    return
    
    # Initial call
    all_vertices = set(range(n))
    backtrack(all_vertices, [], 0)
    
    if best_solution is not None:
        # Convert to list of lists with Python ints
        result = []
        for part in best_solution:
            result.append([int(v) for v in part])
        return result
    else:
        # Fallback: greedy assignment
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts

# EVOLVE-BLOCK-END
