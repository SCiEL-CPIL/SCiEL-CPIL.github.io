import time
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 55
    n = P['n']
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
        
    def is_adjacent(u, v):
        return (min(u,v), max(u,v)) in edge_set

    # Candidate vertices: Top 18 by degree
    degrees = {v: len(adj[v]) for v in range(n)}
    candidates = sorted(range(n), key=lambda v: -degrees[v])[:18]

    def find_short_paths(u, v, forbidden_interior, max_len=8, max_paths=20):
        """Find all simple paths from u to v of length <= max_len avoiding forbidden_interior."""
        paths = []
        if u in forbidden_interior or v in forbidden_interior:
            return paths
        if u == v:
            return paths
        
        # Sort neighbors by degree? Or just iterate
        def dfs(current, target, path, visited):
            nonlocal paths
            if time.time() > deadline:
                return
            if current == target:
                if len(path) - 1 <= max_len:
                    paths.append(list(path))
                return
            if len(path) - 1 >= max_len:
                return
            # Heuristic: try neighbors that are closer to target? 
            # Just iterate
            for nb in adj[current]:
                if nb not in visited and nb not in forbidden_interior:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, target, path, visited)
                    path.pop()
                    visited.remove(nb)
                    
        dfs(u, v, [u], {u})
        
        # Sort by length and take top max_paths
        paths.sort(key=lambda x: len(x))
        return paths[:max_paths]

    def solve_path_packing(branches):
        """
        Try to find vertex-disjoint paths for all pairs in branches.
        Returns dict of paths or None.
        """
        pairs = list(combinations(branches, 2))
        if not pairs:
            return {}
        
        used_interiors = set()
        paths_dict = {}
        
        def backtrack(unassigned_pairs):
            if time.time() > deadline:
                return None
            if not unassigned_pairs:
                return dict(paths_dict)
            
            # MRV: Pick pair with fewest valid paths
            min_count = float('inf')
            best_pair = None
            best_paths = None
            
            for p in unassigned_pairs:
                u, v = p
                forb_interior = used_interiors | (set(branches) - {u, v})
                ps = find_short_paths(u, v, forb_interior, max_len=8, max_paths=20)
                if len(ps) < min_count:
                    min_count = len(ps)
                    best_pair = p
                    best_paths = ps
                if min_count == 0:
                    return None
            
            if best_pair is None:
                return None
                
            u, v = best_pair
            for path in best_paths:
                interior = set(path[1:-1])
                
                # Update state
                new_used = used_interiors | interior
                paths_dict[(u, v)] = path
                paths_dict[(v, u)] = path
                
                res = backtrack([p for p in unassigned_pairs if p != best_pair])
                if res is not None:
                    return res
                    
                # Backtrack
                del paths_dict[(u, v)]
                del paths_dict[(v, u)]
                
            return None
            
        return backtrack(pairs)

    # Search for largest K
    best_branches = None
    best_paths = None
    
    # Try K from high to low
    for k in range(n, 1, -1):
        if time.time() > deadline:
            break
            
        if k > len(candidates):
            candidates = sorted(range(n), key=lambda v: -degrees[v])[:k+2]
            
        found = False
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            branches = list(combo)
            res = solve_path_packing(branches)
            if res is not None:
                best_branches = branches
                best_paths = res
                found = True
                break
        
        if found:
            break

    if best_branches is None:
        # Fallback
        best_branches = [0, 1]
        best_paths = {(0, 1): [0, 1]}

    # Build output
    output_paths = []
    for b1, b2 in combinations(best_branches, 2):
        key = (min(b1, b2), max(b1, b2))
        path = best_paths.get(key)
        if path is None:
            path = find_short_paths(b1, b2, set(best_branches) - {b1, b2}, max_len=10, max_paths=1)[0]
            if not path:
                path = [b1, b2]
        
        output_paths.append(list(path))
        
    return {
        'branches': list(best_branches),
        'paths': output_paths
    }
