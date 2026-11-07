"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time
import sys

sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 75.0  # safety margin before 90s kill
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute edge membership for quick checks
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if vertices[j] not in adj[vertices[i]]:
                    return False
        return True
    
    def is_independent(vertices):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if vertices[j] in adj[vertices[i]]:
                    return False
        return True
    
    # Use backtracking to find minimum number of parts
    # Each part must be a clique or independent set
    # We'll try to assign vertices to existing parts or create new parts
    
    best_parts = None
    best_count = n + 1
    
    # Order vertices by degree (descending) for better pruning
    vertices = list(range(n))
    vertices.sort(key=lambda v: len(adj[v]), reverse=True)
    
    # We will use a more efficient backtracking approach
    # State: list of parts, where each part is a list of vertices
    # We track the type of each part (clique or independent)
    
    def backtrack(idx, parts, part_types):
        nonlocal best_parts, best_count
        if time.time() > deadline:
            return
        
        if idx == len(vertices):
            count = len(parts)
            if count < best_count:
                best_count = count
                best_parts = [list(p) for p in parts]
            return
        
        v = vertices[idx]
        
        # Try adding to existing parts
        for i, part in enumerate(parts):
            # Check if we can add v to this part
            if part_types[i] == 'clique':
                if all(v in adj[u] for u in part):
                    part.append(v)
                    backtrack(idx + 1, parts, part_types)
                    part.pop()
            else:  # independent
                if all(v not in adj[u] for u in part):
                    part.append(v)
                    backtrack(idx + 1, parts, part_types)
                    part.pop()
        
        # Try creating a new part (only if it doesn't exceed best_count)
        if len(parts) < best_count - 1:
            # New part as clique
            parts.append([v])
            part_types.append('clique')
            backtrack(idx + 1, parts, part_types)
            parts.pop()
            part_types.pop()
            
            # New part as independent
            parts.append([v])
            part_types.append('independent')
            backtrack(idx + 1, parts, part_types)
            parts.pop()
            part_types.pop()
    
    backtrack(0, [], [])
    
    # If backtracking didn't find anything or timed out, use greedy fallback
    if best_parts is None:
        parts = []
        for v in vertices:
            placed = False
            for part in parts:
                if is_clique(part + [v]):
                    part.append(v)
                    placed = True
                    break
                if is_independent(part + [v]):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        best_parts = parts
    
    # Ensure all parts are nonempty lists of Python ints
    result = []
    for part in best_parts:
        if part:
            result.append([int(v) for v in part])
    
    return result
# EVOLVE-BLOCK-END
