import time
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 55  # safety margin
    
    n = P['n']
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
        # Direct edge
        if is_adjacent(src, dst):
            return [src, dst]
        # BFS
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            if cur == dst:
                # Reconstruct path
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
    
    def find_path(src, dst, forbidden, max_len=5):
        """Find a path from src to dst avoiding forbidden, with length <= max_len."""
        if src == dst:
            return None
        if src in forbidden or dst in forbidden:
            return None
        if is_adjacent(src, dst):
            return [src, dst]
        # BFS with depth limit
        visited = {src}
        parent = {src: None}
        queue = deque([(src, 0)])
        while queue:
            cur, depth = queue.popleft()
            if depth >= max_len - 1:
                continue
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
                    queue.append((nb, depth + 1))
        return None
    
    def get_interior(path):
        """Get interior vertices of a path (excluding endpoints)."""
        return set(path[1:-1])
    
    def try_add_branch(branches, paths_dict, new_branch, forbidden_interiors):
        """Try to add new_branch to the set. Returns (success, updated_paths, updated_forbidden)."""
        new_paths = dict(paths_dict)
        new_forbidden = set(forbidden_interiors)
        
        for b in branches:
            # Find path between new_branch and b
            path = find_path(new_branch, b, new_forbidden, max_len=6)
            if path is None:
                return False, None, None
            interior = get_interior(path)
            if interior & new_forbidden:
                return False, None, None
            new_paths[frozenset((new_branch, b))] = path
            new_forbidden.update(interior)
        
        return True, new_paths, new_forbidden
    
    def evaluate_solution(branches, paths_dict):
        """Verify the solution is valid."""
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
    
    # Strategy: Try all pairs as starting point, then greedily add more branches
    # Order vertices by degree (higher degree first) for better path finding
    degrees = {v: len(adj[v]) for v in range(n)}
    vertices_by_degree = sorted(range(n), key=lambda v: -degrees[v])
    
    # Try different starting pairs
    for i in range(len(vertices_by_degree)):
        if time.time() > deadline:
            break
        v1 = vertices_by_degree[i]
        for j in range(i+1, len(vertices_by_degree)):
            if time.time() > deadline:
                break
            v2 = vertices_by_degree[j]
            
            # Start with pair (v1, v2)
            initial_forbidden = set()
            initial_paths = {}
            
            path12 = find_path(v1, v2, initial_forbidden, max_len=6)
            if path12 is None:
                continue
            
            initial_paths[frozenset((v1, v2))] = path12
            initial_forbidden.update(get_interior(path12))
            
            branches = [v1, v2]
            paths_dict = dict(initial_paths)
            forbidden = set(initial_forbidden)
            
            # Try to add more branches greedily
            for v3 in vertices_by_degree:
                if v3 in branches:
                    continue
                if time.time() > deadline:
                    break
                
                success, new_paths, new_forbidden = try_add_branch(
                    branches, paths_dict, v3, forbidden
                )
                if success:
                    branches.append(v3)
                    paths_dict = new_paths
                    forbidden = new_forbidden
            
            # Check and update best
            if len(branches) > best_size:
                if evaluate_solution(branches, paths_dict):
                    best_size = len(branches)
                    best_branches = list(branches)
                    best_paths = {tuple(sorted(k)): list(v) for k, v in paths_dict.items()}
    
    # Try randomized approach for improvement
    import random
    random.seed(42)
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Random starting pair
        v1, v2 = random.sample(range(n), 2)
        
        initial_forbidden = set()
        initial_paths = {}
        
        path12 = find_path(v1, v2, initial_forbidden, max_len=6)
        if path12 is None:
            continue
        
        initial_paths[frozenset((v1, v2))] = path12
        initial_forbidden.update(get_interior(path12))
        
        branches = [v1, v2]
        paths_dict = dict(initial_paths)
        forbidden = set(initial_forbidden)
        
        # Random order for adding branches
        remaining = [v for v in range(n) if v not in branches]
        random.shuffle(remaining)
        
        for v3 in remaining:
            if time.time() > deadline:
                break
            success, new_paths, new_forbidden = try_add_branch(
                branches, paths_dict, v3, forbidden
            )
            if success:
                branches.append(v3)
                paths_dict = new_paths
                forbidden = new_forbidden
        
        if len(branches) > best_size:
            if evaluate_solution(branches, paths_dict):
                best_size = len(branches)
                best_branches = list(branches)
                best_paths = {tuple(sorted(k)): list(v) for k, v in paths_dict.items()}
    
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
            # Should not happen if evaluate_solution passed
            path = [b1, b2]
        output_paths.append(list(path))
    
    return {
        'branches': list(best_branches),
        'paths': output_paths
    }
