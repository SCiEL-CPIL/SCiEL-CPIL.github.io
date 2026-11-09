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
    
    def bfs_shortest_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src, dst)."""
        if src == dst:
            return [src]
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            curr, path = queue.popleft()
            for nb in adj[curr]:
                if nb == dst:
                    return path + [dst]
                if nb in forbidden or nb in visited:
                    continue
                visited.add(nb)
                queue.append((nb, path + [nb]))
        return None
    
    def find_paths_with_backtrack(branches, pair_order):
        """
        Try to find pairwise internally disjoint paths using backtracking.
        pair_order: list of (i,j) pairs in the order to assign
        Returns dict mapping (i,j) to path, or None if not possible.
        """
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        used_interior = set()
        paths = {}
        
        def backtrack(idx):
            if idx == len(pair_order):
                return True
            
            i, j = pair_order[idx]
            src, dst = branches[i], branches[j]
            
            # Find all shortest paths (or just one) from src to dst avoiding used_interior
            # and not going through other branches
            found_path = None
            visited = {src}
            queue = deque([(src, [src])])
            while queue:
                curr, path = queue.popleft()
                for nb in adj[curr]:
                    if nb == dst:
                        candidate_path = path + [dst]
                        # Check no interior vertex is a branch or already used
                        valid = True
                        for v in candidate_path[1:-1]:
                            if v in branch_set or v in used_interior:
                                valid = False
                                break
                        if valid:
                            found_path = candidate_path
                            break
                    if nb in visited or nb in used_interior or nb in branch_set:
                        continue
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
                if found_path:
                    break
            
            if found_path is None:
                return False
            
            # Use this path
            interior = set(found_path[1:-1])
            used_interior.update(interior)
            paths[(i, j)] = found_path
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            used_interior.difference_update(interior)
            del paths[(i, j)]
            return False
        
        if backtrack(0):
            return paths
        return None
    
    def try_subset(branches):
        """Try to find valid paths for the given subset of branches."""
        k = len(branches)
        if k < 2:
            return None
        
        pairs = list(combinations(range(k), 2))
        
        # Try multiple orderings
        # Order 1: by shortest path length
        def get_pair_length(i, j):
            sp = bfs_shortest_path(branches[i], branches[j], set())
            if sp is None:
                return float('inf')
            return len(sp)
        
        # Try ordering by shortest path
        pair_lengths = [(get_pair_length(i, j), i, j) for i, j in pairs]
        pair_lengths.sort()
        order1 = [(i, j) for _, i, j in pair_lengths]
        
        # Try ordering by sum of degrees (lower first - more constrained)
        def degree_sum(i, j):
            return len(adj[branches[i]]) + len(adj[branches[j]])
        order2 = sorted(pairs, key=lambda p: degree_sum(p[0], p[1]))
        
        # Try reverse of order1
        order3 = list(reversed(order1))
        
        for order in [order1, order2, order3]:
            paths = find_paths_with_backtrack(branches, order)
            if paths is not None:
                result_paths = []
                for (i, j), path in paths.items():
                    result_paths.append(list(path))
                return {'branches': list(branches), 'paths': result_paths}
        
        return None
    
    # Strategy: Greedy construction with multiple attempts
    degrees = [len(adj[v]) for v in range(n)]
    sorted_vertices = sorted(range(n), key=lambda v: -degrees[v])
    
    best_result = None
    best_k = 0
    
    # Try multiple starting points and orderings
    for start_idx in range(min(5, n)):
        current_set = []
        # Start with vertex at start_idx
        start_v = sorted_vertices[start_idx]
        current_set.append(start_v)
        
        for v in sorted_vertices:
            if v == start_v:
                continue
            trial_set = current_set + [v]
            if len(trial_set) >= 2:
                result = try_subset(trial_set)
                if result is not None:
                    current_set = trial_set
                    if len(current_set) > best_k:
                        best_k = len(current_set)
                        best_result = result
    
    if best_result is not None:
        return best_result
    
    # Fallback: find any edge
    if edges:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
