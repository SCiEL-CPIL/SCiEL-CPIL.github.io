"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from collections import deque
import sys

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    def find_all_simple_paths(src, dst, forbidden, max_len=10):
        """Find all simple paths from src to dst avoiding forbidden vertices, up to max_len."""
        paths = []
        # forbidden should not contain src or dst
        forbidden = set(forbidden)
        
        def dfs(curr, path, visited):
            if len(path) > max_len:
                return
            if curr == dst:
                paths.append(list(path))
                return
            for nb in adj[curr]:
                if nb in visited or nb in forbidden:
                    continue
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
                
        visited = {src}
        dfs(src, [src], visited)
        
        # Sort by length
        paths.sort(key=lambda p: len(p))
        return paths

    def verify(branches):
        """
        Verify if the given branches admit pairwise internally disjoint paths.
        Returns list of paths if valid, else None.
        """
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        pairs = list(combinations(range(k), 2))
        
        # Heuristic: Sort pairs by the number of available paths (shortest first)
        # We will dynamically order them in the recursive step if needed, 
        # but for now, let's just use a static order based on initial shortest path length.
        
        def get_initial_cost(i, j):
            b_i = branches[i]
            b_j = branches[j]
            # Forbidden are all other branches
            forb = branch_set - {b_i, b_j}
            paths = find_all_simple_paths(b_i, b_j, forb, max_len=5)
            if not paths:
                return float('inf')
            return len(paths[0])
            
        pair_costs = [(get_initial_cost(i, j), i, j) for i, j in pairs]
        pair_costs.sort()
        
        # If any pair has inf cost, it's impossible
        if any(c == float('inf') for c, _, _ in pair_costs):
            return None
            
        ordered_pairs = [(i, j) for _, i, j in pair_costs]
        
        used_interior = set()
        found_paths = {}
        
        def backtrack(idx):
            if idx == len(ordered_pairs):
                return True
            
            i, j = ordered_pairs[idx]
            b_i = branches[i]
            b_j = branches[j]
            
            # Forbidden: used interiors + all other branches
            forbidden = used_interior | (branch_set - {b_i, b_j})
            
            # Find candidate paths
            candidates = find_all_simple_paths(b_i, b_j, forbidden, max_len=10)
            
            if not candidates:
                return False
            
            # Try each candidate path
            for path in candidates:
                interiors = path[1:-1]
                # Check if interiors are disjoint (they are simple paths, so unique)
                # And not in used_interior (checked in forbidden)
                
                # Add to used
                for v in interiors:
                    used_interior.add(v)
                found_paths[(i, j)] = path
                
                if backtrack(idx + 1):
                    return True
                
                # Backtrack
                for v in interiors:
                    used_interior.remove(v)
                del found_paths[(i, j)]
                
            return False
            
        if backtrack(0):
            # Convert to output format
            result_paths = []
            for (i, j), path in found_paths.items():
                result_paths.append(list(path))
            return result_paths
        else:
            return None

    # Strategy: Greedily add high-degree vertices
    degrees = [len(adj[v]) for v in range(n)]
    sorted_vertices = sorted(range(n), key=lambda v: -degrees[v])
    
    best_branches = []
    best_paths = None
    
    for v in sorted_vertices:
        trial_set = best_branches + [v]
        if len(trial_set) < 2:
            continue
            
        paths = verify(trial_set)
        if paths is not None:
            best_branches = trial_set
            best_paths = paths
        else:
            # Optional: try skipping this vertex? 
            # The greedy approach is simple. If adding v fails, we skip it.
            # This might miss a solution where v is included but a previous vertex is not.
            # Given N=24, this is a reasonable heuristic.
            pass

    if best_paths is not None:
        return {'branches': list(best_branches), 'paths': best_paths}
    
    # Fallback: Find any edge
    if edges:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
