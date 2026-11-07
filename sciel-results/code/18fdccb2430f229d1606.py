"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    for i in range(n):
        adj_list[i].sort(key=lambda x: -degrees[x])
    
    # Precompute common neighbors for length-3 path options
    common = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            common[u][v] = adj[u] & adj[v]
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    def find_bfs_path(src, dst, forbidden, max_len=7):
        """Find a simple path from src to dst avoiding forbidden vertices."""
        if src == dst:
            return None
        q = deque()
        q.append((src, (src,)))
        visited = {src}
        while q:
            curr, path = q.popleft()
            if len(path) >= max_len:
                continue
            if curr == dst:
                return list(path)
            for nb in adj_list[curr]:
                if nb not in visited and (nb not in forbidden or nb == dst):
                    visited.add(nb)
                    q.append((nb, path + (nb,)))
        return None
    
    def check_subdivision(branches):
        """Check if branches form a valid clique subdivision. Returns paths or None."""
        k = len(branches)
        branch_set = frozenset(branches)
        
        all_pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                u, v = branches[i], branches[j]
                all_pairs.append((u, v))
        
        # Pigeonhole: non-adjacent pairs need at least 1 interior vertex each
        non_adj_count = sum(1 for u, v in all_pairs if v not in adj[u])
        if non_adj_count > n - k:
            return None
        
        # Pre-compute path options for each pair, sorted by length (shorter first)
        pair_options = {}
        for u, v in all_pairs:
            options = []
            if v in adj[u]:
                options.append([u, v])
            # Length-3 paths: u-w-v
            for w in common[u][v]:
                if w not in branch_set and w != u and w != v:
                    options.append([u, w, v])
            pair_options[(u, v)] = options
        
        # If any pair has no options, fail early
        for u, v in all_pairs:
            if not pair_options[(u, v)]:
                return None
        
        # Sort pairs by number of options (most constrained first)
        sorted_pairs = sorted(all_pairs, key=lambda p: len(pair_options[p]))
        
        # Backtracking
        used_interiors = set()
        path_map = {}
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(sorted_pairs):
                return True
            
            u, v = sorted_pairs[idx]
            
            for option in pair_options[(u, v)]:
                interior = set(option[1:-1])
                if interior.isdisjoint(used_interiors) and interior.isdisjoint(branch_set):
                    path_map[(u, v)] = option
                    used_interiors.update(interior)
                    if backtrack(idx + 1):
                        return True
                    used_interiors.difference_update(interior)
                    del path_map[(u, v)]
            
            # If no short option worked, try BFS for a longer path
            forbidden = used_interiors | (set(branch_set) - {u, v})
            path = find_bfs_path(u, v, forbidden, max_len=6)
            if path is not None:
                interior = set(path[1:-1])
                if interior.isdisjoint(branch_set) and interior.isdisjoint(used_interiors):
                    path_map[(u, v)] = path
                    used_interiors.update(interior)
                    if backtrack(idx + 1):
                        return True
                    used_interiors.difference_update(interior)
                    del path_map[(u, v)]
            
            return False
        
        if backtrack(0):
            paths = []
            for i in range(k):
                for j in range(i + 1, k):
                    paths.append(list(path_map[(branches[i], branches[j])]))
            return paths
        
        return None
    
    # Find max clique using Bron-Kerbosch with pivot
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = list(P_set - adj[pivot])
        else:
            candidates = list(P_set)
        for v in candidates:
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    clique_size = len(best_clique)
    
    # Try k from min(12, n) down to clique_size
    for k in range(min(12, n), clique_size - 1, -1):
        if time.time() > deadline:
            break
        
        # Start with high-degree vertices
        sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
        pool_size = min(n, max(k + 4, n - 4))
        candidates = sorted_vertices[:pool_size]
        
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            
            branches = list(combo)
            paths = check_subdivision(branches)
            if paths is not None:
                result_branches = [int(b) for b in branches]
                result_paths = [list(p) for p in paths]
                return {'branches': result_branches, 'paths': result_paths}
    
    # Fallback: max clique
    if len(best_clique) >= 2:
        branches = sorted(best_clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([int(branches[i]), int(branches[j])])
        return {'branches': list(branches), 'paths': paths}
    
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
