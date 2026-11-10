"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = _time.time() + 85
    
    def find_shortest_path(src, dst, forbidden):
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return list(reversed(path))
            for v in adj[u]:
                if v not in visited and (v == dst or v not in forbidden):
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
        return None
    
    def check_branch_set(branches):
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        used_interiors = set()
        paths = {}
        
        def pair_key(p):
            path = find_shortest_path(p[0], p[1], bset - {p[0], p[1]})
            if path is None:
                return (float('inf'), p)
            return (len(path), p)
        
        sorted_pairs = sorted(pairs, key=pair_key)
        
        for u, v in sorted_pairs:
            forbidden = (bset - {u, v}) | used_interiors
            path = find_shortest_path(u, v, forbidden)
            if path is None:
                return None
            interior = set(path[1:-1])
            if interior & bset:
                return None
            if interior & used_interiors:
                return None
            used_interiors |= interior
            paths[(u, v)] = path
        
        return paths
    
    # Maximum clique via branch-and-bound with greedy coloring (Tomita's algorithm)
    def max_clique():
        best = []
        
        def expand(clique, candidates):
            nonlocal best
            if _time.time() > deadline:
                return
            if not candidates:
                if len(clique) > len(best):
                    best = list(clique)
                return
            
            # Greedy coloring for upper bound
            colors = {}
            for v in candidates:
                used_colors = set()
                for u in candidates:
                    if u in adj[v] and u in colors:
                        used_colors.add(colors[u])
                c = 0
                while c in used_colors:
                    c += 1
                colors[v] = c
            
            max_color = max(colors.values()) + 1
            if len(clique) + max_color <= len(best):
                return
            
            # Process vertices in descending color order
            ordered = sorted(candidates, key=lambda v: -colors[v])
            candidates_set = set(candidates)
            
            for v in ordered:
                if v not in candidates_set:
                    continue
                new_clique = clique + [v]
                new_candidates = [u for u in candidates_set if u in adj[v] and u != v]
                expand(new_clique, new_candidates)
                candidates_set.discard(v)
                # Update bound: max color among remaining candidates adjacent to clique
                remaining_max = 0
                for u in candidates_set:
                    if all(u in adj[c] for c in clique):
                        remaining_max = max(remaining_max, colors.get(u, 0) + 1)
                if len(clique) + remaining_max <= len(best):
                    break
        
        expand([], list(range(n)))
        return best
    
    # Find maximum clique
    clique = max_clique()
    
    best_branches = None
    best_paths = None
    
    # Verify the clique as branches
    if len(clique) >= 2:
        paths = check_branch_set(clique)
        if paths is not None:
            best_branches = list(clique)
            best_paths = paths
    
    # Try to grow by adding non-clique vertices
    if best_branches:
        current = list(best_branches)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            for candidate in range(n):
                if candidate in current:
                    continue
                new_set = current + [candidate]
                paths = check_branch_set(new_set)
                if paths is not None:
                    current = new_set
                    best_branches = current
                    best_paths = paths
                    improved = True
                    break
    
    # Try top-degree vertex combinations
    deg = [len(adj[i]) for i in range(n)]
    top = sorted(range(n), key=lambda x: -deg[x])[:10]
    for size in range(min(8, len(top)), 1, -1):
        if _time.time() > deadline:
            break
        found = False
        for combo in combinations(top, size):
            if _time.time() > deadline:
                break
            paths = check_branch_set(list(combo))
            if paths is not None and (best_branches is None or size > len(best_branches)):
                best_branches = list(combo)
                best_paths = paths
                found = True
                break
        if found:
            break
    
    # Random search for larger sets
    random.seed(99)
    for _ in range(500):
        if _time.time() > deadline:
            break
        size = random.randint(4, 12)
        combo = random.sample(range(n), size)
        paths = check_branch_set(combo)
        if paths is not None and (best_branches is None or size > len(best_branches)):
            best_branches = combo
            best_paths = paths
    
    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}
    
    branches = [int(b) for b in best_branches]
    paths_list = []
    for (u, v) in combinations(branches, 2):
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path is None:
            path = [u, v]
        paths_list.append([int(x) for x in path])
    
    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
