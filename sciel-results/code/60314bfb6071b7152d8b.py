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
    
    def find_shortest_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden interior vertices."""
        if src == dst:
            return None
        if src in forbidden or dst in forbidden:
            return None
        if is_adjacent(src, dst):
            return [src, dst]
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            if cur == dst:
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[cur]:
                if nb not in visited and nb not in forbidden:
                    visited.add(nb)
                    parent[nb] = cur
                    queue.append(nb)
        return None
    
    def get_interior(path):
        return set(path[1:-1]) if len(path) > 2 else set()
    
    def evaluate_solution(branches, paths_dict):
        if len(branches) < 2:
            return False
        used_interiors = set()
        for b1, b2 in combinations(branches, 2):
            key = frozenset((b1, b2))
            if key not in paths_dict:
                return False
            path = paths_dict[key]
            if path[0] not in (b1, b2) or path[-1] not in (b1, b2):
                return False
            interior = get_interior(path)
            if interior & used_interiors:
                return False
            if interior & set(branches):
                return False
            used_interiors.update(interior)
        return True
    
    best_branches = None
    best_paths = None
    best_size = 0
    
    # Strategy 1: Greedy set extension with multiple starting pairs
    # Order by degree
    degrees = {v: len(adj[v]) for v in range(n)}
    vertices_by_degree = sorted(range(n), key=lambda v: -degrees[v])
    
    def greedy_extend(start_branches, start_paths, start_forbidden, order):
        """Extend a partial solution greedily."""
        branches = list(start_branches)
        paths_dict = dict(start_paths)
        forbidden = set(start_forbidden)
        
        for v in order:
            if time.time() > deadline:
                break
            if v in branches:
                continue
            
            # Try to add v
            new_paths = dict(paths_dict)
            new_forbidden = set(forbidden)
            success = True
            
            for b in branches:
                path = find_shortest_path(v, b, new_forbidden)
                if path is None:
                    success = False
                    break
                interior = get_interior(path)
                if interior & new_forbidden:
                    success = False
                    break
                new_paths[frozenset((v, b))] = path
                new_forbidden.update(interior)
            
            if success:
                branches.append(v)
                paths_dict = new_paths
                forbidden = new_forbidden
        
        return branches, paths_dict, forbidden
    
    # Try all pairs as starting points
    for i in range(len(vertices_by_degree)):
        if time.time() > deadline:
            break
        v1 = vertices_by_degree[i]
        for j in range(i+1, len(vertices_by_degree)):
            if time.time() > deadline:
                break
            v2 = vertices_by_degree[j]
            
            path12 = find_shortest_path(v1, v2, set())
            if path12 is None:
                continue
            
            initial_paths = {frozenset((v1, v2)): path12}
            initial_forbidden = get_interior(path12)
            
            branches, paths_dict, _ = greedy_extend([v1, v2], initial_paths, initial_forbidden, vertices_by_degree)
            
            if len(branches) > best_size:
                if evaluate_solution(branches, paths_dict):
                    best_size = len(branches)
                    best_branches = list(branches)
                    best_paths = {tuple(sorted(k)): list(v) for k, v in paths_dict.items()}
    
    # Strategy 2: Random restarts with different orderings
    import random
    random.seed(42)
    
    for _ in range(500):
        if time.time() > deadline:
            break
        
        # Random starting pair
        v1, v2 = random.sample(range(n), 2)
        path12 = find_shortest_path(v1, v2, set())
        if path12 is None:
            continue
        
        initial_paths = {frozenset((v1, v2)): path12}
        initial_forbidden = get_interior(path12)
        
        # Random order for extension
        remaining = [v for v in range(n) if v not in (v1, v2)]
        random.shuffle(remaining)
        
        branches, paths_dict, _ = greedy_extend([v1, v2], initial_paths, initial_forbidden, remaining)
        
        if len(branches) > best_size:
            if evaluate_solution(branches, paths_dict):
                best_size = len(branches)
                best_branches = list(branches)
                best_paths = {tuple(sorted(k)): list(v) for k, v in paths_dict.items()}
    
    # Strategy 3: Backtracking search for larger cliques
    # Try to find a set of k vertices that admits a clique subdivision
    # Use a smarter approach: try to add vertices in an order that maximizes flexibility
    
    def backtrack_search(current_branches, current_paths, current_forbidden, remaining_candidates, depth=0):
        """Recursively try to add more branches."""
        nonlocal best_branches, best_paths, best_size
        
        if time.time() > deadline:
            return
        
        if len(current_branches) > best_size:
            if evaluate_solution(current_branches, current_paths):
                best_size = len(current_branches)
                best_branches = list(current_branches)
                best_paths = {tuple(sorted(k)): list(v) for k, v in current_paths.items()}
        
        if not remaining_candidates:
            return
        
        # Try adding each candidate
        for idx, v in enumerate(remaining_candidates):
            if time.time() > deadline:
                return
            
            new_paths = dict(current_paths)
            new_forbidden = set(current_forbidden)
            success = True
            
            for b in current_branches:
                path = find_shortest_path(v, b, new_forbidden)
                if path is None:
                    success = False
                    break
                interior = get_interior(path)
                if interior & new_forbidden:
                    success = False
                    break
                new_paths[frozenset((v, b))] = path
                new_forbidden.update(interior)
            
            if success:
                new_remaining = [x for x in remaining_candidates if x != v and x not in current_branches]
                backtrack_search(current_branches + [v], new_paths, new_forbidden, new_remaining, depth + 1)
    
    # Try backtracking from a few good starting points
    if best_size < 8:
        for start_pair in [(vertices_by_degree[0], vertices_by_degree[1]), 
                          (vertices_by_degree[0], vertices_by_degree[2]),
                          (vertices_by_degree[1], vertices_by_degree[2])]:
            if time.time() > deadline:
                break
            v1, v2 = start_pair
            path12 = find_shortest_path(v1, v2, set())
            if path12 is None:
                continue
            
            initial_paths = {frozenset((v1, v2)): path12}
            initial_forbidden = get_interior(path12)
            remaining = [v for v in vertices_by_degree if v not in (v1, v2)]
            backtrack_search([v1, v2], initial_paths, initial_forbidden, remaining[:10])
    
    # Final fallback
    if best_branches is None:
        best_branches = [0, 1]
        best_paths = {(0, 1): [0, 1]}
    
    # Build output
    output_paths = []
    for b1, b2 in combinations(best_branches, 2):
        key = (min(b1, b2), max(b1, b2))
        path = best_paths.get(key)
        if path is None:
            path = [b1, b2]
        output_paths.append(list(path))
    
    return {
        'branches': list(best_branches),
        'paths': output_paths
    }
