import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find maximum clique using Bron-Kerbosch
    best_clique = []
    
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        pivot_candidates = P_set | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
        candidates = [v for v in P_set if v not in adj[pivot]]
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P_set = P_set - {v}
            X = X | {v}
    
    bron_kerbosch(set(), set(range(n)), set())
    
    # Try to find a larger clique subdivision by using non-clique vertices as branches
    # with internally disjoint paths
    # Strategy: try all subsets of vertices as potential branches, check if we can
    # find internally disjoint paths between all pairs
    
    def find_bfs_path(src, dst, forbidden):
        """Find a simple path from src to dst avoiding forbidden vertices."""
        if src == dst:
            return None
        if dst in forbidden:
            return None
        queue = deque()
        queue.append((src, [src]))
        visited = {src}
        while queue:
            node, path = queue.popleft()
            for nb in adj[node]:
                if nb == dst:
                    return path + [dst]
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None
    
    def can_find_disjoint_paths(branches):
        """Check if we can find internally disjoint paths for all pairs of branches."""
        branch_set = set(branches)
        paths = {}
        used_interior = set()
        
        # Sort pairs to process in a deterministic order
        pairs = list(combinations(branches, 2))
        
        # Try to find paths greedily
        for u, v in pairs:
            forbidden = branch_set - {u, v} | used_interior
            path = find_bfs_path(u, v, forbidden)
            if path is None:
                return None
            # Add interior vertices to used set
            for vertex in path[1:-1]:
                used_interior.add(vertex)
            paths[(u, v)] = path
        
        return paths
    
    # First try the maximum clique
    if len(best_clique) >= 2:
        branches = sorted(best_clique)
        paths = can_find_disjoint_paths(branches)
        if paths is not None:
            result_paths = []
            for u, v in combinations(branches, 2):
                result_paths.append([int(x) for x in paths[(u, v)]])
            return {'branches': [int(x) for x in branches], 'paths': result_paths}
    
    # Try larger sets by adding vertices that can be connected with disjoint paths
    # Start with the clique and try to add more vertices
    best_result = None
    best_count = 0
    
    # Try all subsets of size up to min(n, 8)
    # Use a greedy approach: start with clique, try adding vertices one by one
    current_branches = list(best_clique) if len(best_clique) >= 2 else [0, 1]
    
    # Try to expand the branch set
    for _ in range(3):  # Try a few rounds of expansion
        if time.time() > deadline:
            break
        expanded = False
        for candidate in range(n):
            if candidate in current_branches:
                continue
            if time.time() > deadline:
                break
            new_branches = current_branches + [candidate]
            paths = can_find_disjoint_paths(new_branches)
            if paths is not None:
                current_branches = new_branches
                expanded = True
                break
    
    if len(current_branches) > best_count:
        best_count = len(current_branches)
        paths = can_find_disjoint_paths(current_branches)
        if paths is not None:
            result_paths = []
            for u, v in combinations(current_branches, 2):
                result_paths.append([int(x) for x in paths[(u, v)]])
            best_result = {'branches': [int(x) for x in sorted(current_branches)], 'paths': result_paths}
    
    # Also try random subsets to find better solutions
    import random
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline:
            break
        size = random.randint(3, min(n, 7))
        subset = random.sample(range(n), size)
        paths = can_find_disjoint_paths(subset)
        if paths is not None and len(subset) > best_count:
            best_count = len(subset)
            result_paths = []
            for u, v in combinations(subset, 2):
                result_paths.append([int(x) for x in paths[(u, v)]])
            best_result = {'branches': [int(x) for x in sorted(subset)], 'paths': result_paths}
    
    if best_result is not None:
        return best_result
    
    # Fallback: return an edge
    for u, v in P['edges']:
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    return {'branches': [0, 1], 'paths': [[0, 1]]}
