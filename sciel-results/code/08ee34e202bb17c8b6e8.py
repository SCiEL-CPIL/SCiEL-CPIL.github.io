"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    _t0 = _time.time()
    _deadline = _t0 + 85.0  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def find_path(src, dst, forbidden):
        """BFS to find shortest path from src to dst avoiding forbidden vertices."""
        if src == dst:
            return [src]
        if src in forbidden or dst in forbidden:
            return None
        from collections import deque
        q = deque()
        q.append((src, [src]))
        visited = {src}
        while q:
            node, path = q.popleft()
            if node == dst:
                return path
            for nb in adj[node]:
                if nb not in visited and nb not in forbidden:
                    visited.add(nb)
                    q.append((nb, path + [nb]))
        return None
    
    def check_branch_set(branches):
        """Check if we can find internally vertex-disjoint paths for all pairs in branches."""
        bset = set(branches)
        paths = {}
        forbidden = set()
        # Process pairs in order; for each pair, find a path avoiding all branch vertices and previously used interior vertices
        pair_list = list(combinations(sorted(branches), 2))
        for a, b in pair_list:
            # Forbidden: all branch vertices except a and b, plus all previously used interior vertices
            forb = (bset - {a, b}) | forbidden
            p = find_path(a, b, forb)
            if p is None:
                return None
            # Add interior vertices to forbidden
            for v in p[1:-1]:
                forbidden.add(v)
            paths[(a, b)] = p
        return paths
    
    def try_grow(initial_branches):
        """Try to grow a branch set greedily."""
        branches = list(initial_branches)
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if v in branches:
                    continue
                new_branches = branches + [v]
                paths = check_branch_set(new_branches)
                if paths is not None:
                    branches = new_branches
                    changed = True
                    if _time.time() > _deadline:
                        break
            if _time.time() > _deadline:
                break
        return branches
    
    # Try multiple starting points
    best_result = None
    best_size = 0
    
    # Start with pairs of high-degree vertices
    deg = [len(adj[i]) for i in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: -deg[x])
    
    # Try all pairs of top vertices as seeds
    for i in range(min(8, len(sorted_verts))):
        for j in range(i+1, min(8, len(sorted_verts))):
            if _time.time() > _deadline:
                break
            seed = [sorted_verts[i], sorted_verts[j]]
            grown = try_grow(seed)
            if len(grown) > best_size:
                paths = check_branch_set(grown)
                if paths is not None:
                    best_size = len(grown)
                    best_result = (grown, paths)
        if _time.time() > _deadline:
            break
    
    # Also try single vertex starts
    for v in sorted_verts[:6]:
        if _time.time() > _deadline:
            break
        grown = try_grow([v])
        if len(grown) > best_size:
            paths = check_branch_set(grown)
            if paths is not None:
                best_size = len(grown)
                best_result = (grown, paths)
    
    if best_result is None:
        # Fallback: just an edge
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    branches, paths = best_result
    branch_list = [int(b) for b in branches]
    path_list = []
    for (a, b) in combinations(sorted(branch_list), 2):
        p = paths.get((a, b)) or paths.get((b, a))
        if p is None:
            # Should not happen
            p = [a, b]
        path_list.append([int(x) for x in p])
    
    return {'branches': branch_list, 'paths': path_list}

# EVOLVE-BLOCK-END
