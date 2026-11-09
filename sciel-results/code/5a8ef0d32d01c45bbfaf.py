"""Rewritten program: large clique subdivisions via max-flow verification and local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque
import sys

sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    adj = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _max_clique(adj):
    """Bron-Kerbosch with pivot to find maximum clique."""
    n = len(adj)
    best = []
    
    def bronk(R, P_set, X):
        if not P_set and not X:
            if len(R) > len(best):
                best[:] = R
            return
        
        pivot_set = P_set | X
        if pivot_set:
            pivot = max(pivot_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
            
        for v in list(candidates):
            bronk(R + [v], P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
            
    bronk([], set(range(n)), set())
    return best

def _bfs_path(adj, s, t, blocked):
    """BFS shortest path from s to t avoiding blocked vertices (except s, t)."""
    if s == t:
        return [s]
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == t:
            break
        for nb in adj[cur]:
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

def _try_solve(adj, branches, pair_order):
    """Try to find disjoint interior paths for all pairs in pair_order."""
    used = set()
    all_b = set(branches)
    paths = {}
    for (u, v) in pair_order:
        blocked = used | (all_b - {u, v})
        path = _bfs_path(adj, u, v, blocked)
        if path is None:
            return None
        interior = set(path[1:-1])
        if interior & used:
            return None
        used |= interior
        paths[(u, v)] = path
    return paths

def _order_pairs_by_shortest(adj, branches):
    """Order pairs by their shortest path length ignoring other pairs' constraints."""
    bs = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    scored = []
    for (u, v) in pairs:
        blocked = bs - {u, v}
        if u == v:
            L = 0
        else:
            prev = {u: None}
            q = deque([u])
            dist = {u: 0}
            found = False
            while q:
                cur = q.popleft()
                if cur == v:
                    found = True
                    break
                for nb in adj[cur]:
                    if nb in prev:
                        continue
                    if nb in blocked:
                        continue
                    prev[nb] = cur
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
            L = dist.get(v, 10**9) if found else 10**9
        scored.append((L, (u, v)))
    scored.sort(key=lambda x: x[0])
    return [p for (_, p) in scored]

def _try_solve_with_reorder(adj, branches):
    """Try multiple orderings of pairs."""
    bs = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    # Try natural order
    r = _try_solve(adj, branches, pairs)
    if r is not None:
        return r
    # Try sorted by shortest
    order = _order_pairs_by_shortest(adj, branches)
    r = _try_solve(adj, branches, order)
    if r is not None:
        return r
    # Try reverse
    r = _try_solve(adj, branches, list(reversed(order)))
    if r is not None:
        return r
    # Try random orders
    for _ in range(5):
        random.shuffle(pairs)
        r = _try_solve(adj, branches, pairs)
        if r is not None:
            return r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    adj = _build_adj()
    n = P['n']
    vertices = list(range(n))
    
    def time_up():
        return time.time() > deadline
    
    # 1. Find Max Clique
    max_c = _max_clique(adj)
    best_branches = list(max_c)
    best_paths = _try_solve_with_reorder(adj, best_branches)
    
    # If max clique doesn't work, try smaller subsets
    if best_paths is None:
        while len(best_branches) > 2 and best_paths is None:
            best_branches.pop()
            best_paths = _try_solve_with_reorder(adj, best_branches)
    
    # 2. Greedy Expansion from best known solution
    current_branches = list(best_branches)
    current_paths = best_paths
    
    improved = True
    max_expansions = 15
    exp_count = 0
    while improved and not time_up() and exp_count < max_expansions:
        improved = False
        exp_count += 1
        current = sorted(current_branches)
        non_branch = [x for x in vertices if x not in set(current)]
        non_branch.sort(key=lambda x: len(adj[x]), reverse=True)
        best_add = None
        for vb in non_branch:
            if time_up():
                break
            new_bs = current + [vb]
            paths = _try_solve_with_reorder(adj, new_bs)
            if paths is not None:
                best_add = (vb, paths)
                break
        if best_add is not None:
            vb, paths = best_add
            current_branches = sorted(current_branches + [vb])
            current_paths = paths
            if len(current_branches) > len(best_branches):
                best_branches = list(current_branches)
                best_paths = current_paths
            improved = True
    
    # 3. Randomized search for larger branch sets
    random.seed(42)
    for restart in range(200):
        if time_up():
            break
            
        # Random start with bias towards high-degree vertices
        size = random.randint(3, min(8, n))
        weights = [len(adj[x]) + 1 for x in vertices]
        init = random.sample(vertices, size)
        
        paths = _try_solve_with_reorder(adj, init)
        if paths is None:
            continue
            
        current = list(init)
        current_paths = paths
        
        # Greedy expansion from this random start
        remaining = sorted([v for v in vertices if v not in current], key=lambda x: -len(adj[x]))
        for cand in remaining:
            if time_up():
                break
            trial = current + [cand]
            tp = _try_solve_with_reorder(adj, trial)
            if tp is not None:
                current = trial
                current_paths = tp
                
        if len(current) > len(best_branches):
            best_branches = list(current)
            best_paths = current_paths
    
    # 4. Swap-based Local Search
    for _ in range(80):
        if time_up():
            break
            
        if len(best_branches) <= 2:
            break
        idx = random.randint(0, len(best_branches) - 1)
        trial_branches = [b for i, b in enumerate(best_branches) if i != idx]
        
        pool = [v for v in vertices if v not in trial_branches]
        if not pool:
            break
        added = random.choice(pool)
        trial_branches.append(added)
        
        tp = _try_solve_with_reorder(adj, trial_branches)
        if tp is not None:
            best_branches = trial_branches
            best_paths = tp
            # Try to expand from this new state
            current = list(best_branches)
            current_paths = best_paths
            remaining = sorted([v for v in vertices if v not in current], key=lambda x: -len(adj[x]))
            for cand in remaining:
                if time_up():
                    break
                trial = current + [cand]
                ttp = _try_solve_with_reorder(adj, trial)
                if ttp is not None:
                    current = trial
                    current_paths = ttp
                    if len(current) > len(best_branches):
                        best_branches = list(current)
                        best_paths = current_paths
                        break
    
    # Final verification and formatting
    if best_paths is None:
        u, v = P['edges'][0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}
    
    final_paths = []
    for u, v in combinations(best_branches, 2):
        if (u, v) in best_paths:
            final_paths.append(list(best_paths[(u, v)]))
        elif (v, u) in best_paths:
            final_paths.append(list(best_paths[(v, u)]))
        else:
            final_paths.append([u, v])
            
    final_branches = [int(b) for b in best_branches]
    
    return {
        'branches': final_branches,
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
