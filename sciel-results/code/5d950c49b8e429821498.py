"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 85.0

def _build_adj():
    n = P['n']
    edges = P['edges']
    edge_set = set()
    adj = [set() for _ in range(n)]
    for e in edges:
        a, b = e
        edge_set.add((a, b))
        edge_set.add((b, a))
        adj[a].add(b)
        adj[b].add(a)
    adj_list = [sorted(adj[i]) for i in range(n)]
    return n, edges, edge_set, adj, adj_list

def _bfs_path(s, t, blocked, adj_list):
    """BFS shortest path from s to t avoiding blocked vertices (except s, t)."""
    if s == t:
        return [s]
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == t:
            break
        for nb in adj_list[cur]:
            if nb in prev:
                continue
            if nb in blocked and nb != s and nb != t:
                continue
            prev[nb] = cur
            q.append(nb)
    if t not in prev:
        return None
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def _bfs_all_shortest_paths(s, t, blocked, adj_list, max_paths=20):
    """Find multiple shortest paths from s to t avoiding blocked vertices."""
    if s == t:
        return [[s]]
    
    # First BFS to find distance
    dist = {s: 0}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == t:
            break
        for nb in adj_list[cur]:
            if nb in dist:
                continue
            if nb in blocked and nb != s and nb != t:
                continue
            dist[nb] = dist[cur] + 1
            q.append(nb)
    
    if t not in dist:
        return []
    
    target_dist = dist[t]
    paths = []
    
    # DFS to find all shortest paths (with limit)
    def dfs(cur, path_so_far):
        if len(paths) >= max_paths:
            return
        if cur == t:
            if len(path_so_far) == target_dist + 1:
                paths.append(list(path_so_far))
            return
        if len(path_so_far) > target_dist:
            return
        for nb in adj_list[cur]:
            if nb in blocked and nb != s and nb != t:
                continue
            if nb in dist and dist[nb] == len(path_so_far):
                path_so_far.append(nb)
                dfs(nb, path_so_far)
                path_so_far.pop()
    
    dfs(s, [s])
    return paths

def _try_solve_with_paths(branches, path_options, adj_list, edge_set, t0):
    """
    Given branch set and precomputed path options for each pair,
    use backtracking to find internally disjoint paths.
    """
    if time.time() - t0 > DEADLINE - 3.0:
        return None
    
    bs = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    
    # Order pairs by number of available paths (fewer first = more constrained)
    pair_order = []
    for p in pairs:
        u, v = p
        key = (u, v) if u < v else (v, u)
        opts = path_options.get(key, [])
        if not opts:
            return None
        pair_order.append((len(opts), p))
    pair_order.sort()
    ordered_pairs = [p for (_, p) in pair_order]
    
    used = set()
    paths_found = []
    
    def backtrack(idx):
        if time.time() - t0 > DEADLINE - 3.0:
            return False
        if idx == len(ordered_pairs):
            return True
        u, v = ordered_pairs[idx]
        key = (u, v) if u < v else (v, u)
        opts = path_options[key]
        
        # Try each path option
        for path in opts:
            interior = set(path[1:-1])
            if interior & used:
                continue
            used.update(interior)
            paths_found.append(path)
            if backtrack(idx + 1):
                return True
            paths_found.pop()
            used.difference_update(interior)
        return False
    
    if backtrack(0):
        return paths_found
    return None

def _max_clique(vertices, edge_set):
    """Find maximum clique using Bron-Kerbosch with pivot."""
    best = []
    
    def bronk(R, P, X):
        nonlocal best
        if not P and not X:
            if len(R) > len(best):
                best = list(R)
            return
        # Choose pivot
        if P:
            pivot_candidates = P | X
            pivot = max(pivot_candidates, key=lambda v: len(P & edge_set_neighbors(v)))
        else:
            pivot = None
        
        P_minus = set(P)
        if pivot is not None:
            P_minus -= edge_set_neighbors(pivot)
        
        for v in list(P_minus):
            new_R = R | {v}
            new_P = P & edge_set_neighbors(v)
            new_X = X & edge_set_neighbors(v)
            bronk(new_R, new_P, new_X)
            P.discard(v)
            X.add(v)
    
    # Precompute neighbors for edge_set
    def edge_set_neighbors(v):
        return {u for (a, b) in edge_set if a == v for u in [b] if (a,b) in edge_set} | \
               {u for (a, b) in edge_set if b == v for u in [a] if (a,b) in edge_set}
    
    # Simpler: build adjacency
    adj_dict = {}
    for (a, b) in edge_set:
        if a not in adj_dict:
            adj_dict[a] = set()
        if b not in adj_dict:
            adj_dict[b] = set()
        adj_dict[a].add(b)
        adj_dict[b].add(a)
    
    def neighbors(v):
        return adj_dict.get(v, set())
    
    def bronk2(R, P, X):
        nonlocal best
        if not P and not X:
            if len(R) > len(best):
                best = list(R)
            return
        if P:
            pivot_candidates = P | X
            pivot = max(pivot_candidates, key=lambda v: len(P & neighbors(v)))
        else:
            pivot = None
        
        P_minus = set(P)
        if pivot is not None:
            P_minus -= neighbors(pivot)
        
        for v in list(P_minus):
            new_R = R | {v}
            new_P = P & neighbors(v)
            new_X = X & neighbors(v)
            bronk2(new_R, new_P, new_X)
            P.discard(v)
            X.add(v)
    
    bronk2(set(), set(vertices), set())
    return best

def construct():
    t0 = time.time()
    n, edges, edge_set, adj, adj_list = _build_adj()
    vertices = list(range(n))
    
    def time_up():
        return time.time() - t0 > DEADLINE - 3.0
    
    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Find maximum clique
    clique = _max_clique(vertices, edge_set)
    
    # Precompute path options for all pairs in the clique
    path_options = {}
    for pair in combinations(sorted(clique), 2):
        u, v = pair
        blocked = set(clique) - {u, v}
        paths = _bfs_all_shortest_paths(u, v, blocked, adj_list, max_paths=30)
        if paths:
            path_options[(u, v) if u < v else (v, u)] = paths
    
    # Try to solve for the full clique
    best = None
    best_size = 0
    
    if path_options:
        paths = _try_solve_with_paths(clique, path_options, adj_list, edge_set, t0)
        if paths is not None:
            best = {'branches': [int(x) for x in sorted(clique)], 
                    'paths': [[int(x) for x in pp] for pp in paths]}
            best_size = len(clique)
    
    # If full clique doesn't work, try subsets
    if best_size < len(clique):
        # Try all subsets of size len(clique)-1
        for subset in combinations(sorted(clique), len(clique) - 1):
            if time_up():
                break
            if len(subset) <= best_size:
                continue
            sub_set = set(subset)
            sub_path_options = {}
            feasible = True
            for pair in combinations(sorted(subset), 2):
                u, v = pair
                blocked = sub_set - {u, v}
                paths = _bfs_all_shortest_paths(u, v, blocked, adj_list, max_paths=20)
                if not paths:
                    feasible = False
                    break
                sub_path_options[(u, v) if u < v else (v, u)] = paths
            if not feasible:
                continue
            paths = _try_solve_with_paths(list(subset), sub_path_options, adj_list, edge_set, t0)
            if paths is not None:
                best = {'branches': [int(x) for x in sorted(subset)], 
                        'paths': [[int(x) for x in pp] for pp in paths]}
                best_size = len(subset)
                break
    
    # Greedy expansion from best solution
    if best_size >= 2 and not time_up():
        base = sorted(best['branches'])
        improved = True
        max_rounds = 20
        for _ in range(max_rounds):
            if not improved or time_up():
                break
            improved = False
            current = sorted(base)
            non_branch = [x for x in vertices if x not in set(current)]
            non_branch.sort(key=lambda x: len(adj[x]), reverse=True)
            for vb in non_branch:
                if time_up():
                    break
                new_bs = current + [vb]
                new_set = set(new_bs)
                
                # Check if vb can connect to all existing branches
                sub_path_options = {}
                feasible = True
                for pair in combinations(sorted(new_bs), 2):
                    u, v = pair
                    blocked = new_set - {u, v}
                    paths = _bfs_all_shortest_paths(u, v, blocked, adj_list, max_paths=15)
                    if not paths:
                        feasible = False
                        break
                    sub_path_options[(u, v) if u < v else (v, u)] = paths
                if not feasible:
                    continue
                
                paths = _try_solve_with_paths(new_bs, sub_path_options, adj_list, edge_set, t0)
                if paths is not None:
                    base = sorted(new_bs)
                    best = {'branches': [int(x) for x in base], 
                            'paths': [[int(x) for x in pp] for pp in paths]}
                    best_size = len(base)
                    improved = True
                    break
    
    # Try random restarts with different clique subsets
    if not time_up() and best_size < 10:
        random.seed(42)
        for trial in range(100):
            if time_up():
                break
            # Try random subset of vertices
            k = random.randint(3, min(8, n))
            subset = random.sample(vertices, k)
            sub_set = set(subset)
            
            sub_path_options = {}
            feasible = True
            for pair in combinations(sorted(subset), 2):
                u, v = pair
                blocked = sub_set - {u, v}
                paths = _bfs_all_shortest_paths(u, v, blocked, adj_list, max_paths=10)
                if not paths:
                    feasible = False
                    break
                sub_path_options[(u, v) if u < v else (v, u)] = paths
            if not feasible:
                continue
            
            paths = _try_solve_with_paths(subset, sub_path_options, adj_list, edge_set, t0)
            if paths is not None and len(subset) > best_size:
                best = {'branches': [int(x) for x in sorted(subset)], 
                        'paths': [[int(x) for x in pp] for pp in paths]}
                best_size = len(subset)
                
                # Try to expand
                current = sorted(subset)
                for _ in range(10):
                    if time_up():
                        break
                    non_branch = [x for x in vertices if x not in set(current)]
                    non_branch.sort(key=lambda x: len(adj[x]), reverse=True)
                    added = False
                    for vb in non_branch:
                        if time_up():
                            break
                        new_bs = current + [vb]
                        new_set = set(new_bs)
                        sub_path_options2 = {}
                        feasible2 = True
                        for pair in combinations(sorted(new_bs), 2):
                            u, v = pair
                            blocked = new_set - {u, v}
                            paths2 = _bfs_all_shortest_paths(u, v, blocked, adj_list, max_paths=10)
                            if not paths2:
                                feasible2 = False
                                break
                            sub_path_options2[(u, v) if u < v else (v, u)] = paths2
                        if not feasible2:
                            continue
                        paths3 = _try_solve_with_paths(new_bs, sub_path_options2, adj_list, edge_set, t0)
                        if paths3 is not None:
                            current = sorted(new_bs)
                            if len(current) > best_size:
                                best = {'branches': [int(x) for x in current], 
                                        'paths': [[int(x) for x in pp] for pp in paths3]}
                                best_size = len(current)
                            added = True
                            break
                    if not added:
                        break
    
    if best is None:
        return fallback()
    
    return best
# EVOLVE-BLOCK-END
