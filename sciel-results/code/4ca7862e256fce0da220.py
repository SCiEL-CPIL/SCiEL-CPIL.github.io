"""Rewritten program: large clique subdivisions via backtracking with disjoint path constraints."""
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

def _find_path_bfs(adj, src, dst, blocked, deadline):
    """Find a simple path from src to dst avoiding blocked vertices."""
    if src == dst:
        return [src]
    if src in blocked or dst in blocked:
        return None
    
    visited = {src: None}
    q = deque([src])
    
    while q:
        curr = q.popleft()
        if time.time() > deadline:
            return None
        if curr == dst:
            # Reconstruct path
            path = []
            node = dst
            while node is not None:
                path.append(node)
                node = visited[node]
            path.reverse()
            return path
        
        for neighbor in adj[curr]:
            if neighbor in blocked:
                continue
            if neighbor not in visited:
                visited[neighbor] = curr
                q.append(neighbor)
    
    return None

def _try_assign_paths(adj, branches, deadline):
    """
    Try to find pairwise internally disjoint paths for all pairs in branches.
    Uses backtracking with randomized ordering.
    Returns dict of paths if successful, else None.
    """
    if time.time() > deadline:
        return None
    
    k = len(branches)
    if k < 2:
        return None
    
    pairs = list(combinations(branches, 2))
    branch_set = set(branches)
    
    # Sort pairs by a heuristic: prefer pairs that are "easy" (direct edge or short path)
    # But also randomize to avoid getting stuck in bad orderings
    def pair_key(p):
        u, v = p
        if v in adj[u]:
            return (0, 0, u, v)
        # Estimate distance via BFS depth (approximate with degree)
        return (1, len(adj[u]) + len(adj[v]), u, v)
    
    # Randomize slightly to break symmetry
    random.shuffle(pairs)
    pairs.sort(key=pair_key)
    
    used_interiors = set()
    paths = {}
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == len(pairs):
            return True
        
        u, v = pairs[idx]
        blocked = branch_set - {u, v}
        blocked = blocked | used_interiors
        
        # Try to find a path
        path = _find_path_bfs(adj, u, v, blocked, deadline)
        if path is None:
            return False
        
        # Mark interiors
        for mid in path[1:-1]:
            used_interiors.add(mid)
        
        paths[(u, v)] = path
        
        if backtrack(idx + 1):
            return True
        
        # Undo
        for mid in path[1:-1]:
            used_interiors.discard(mid)
        del paths[(u, v)]
        
        return False
    
    # Try a few random orderings
    for attempt in range(10):
        if time.time() > deadline:
            break
        random.shuffle(pairs)
        pairs.sort(key=pair_key)
        used_interiors.clear()
        paths.clear()
        if backtrack(0):
            return paths
    
    return None

def _max_clique(adj):
    """Bron-Kerbosch with pivot to find maximum clique."""
    n = len(adj)
    best = []
    
    def bronk(R, P_set, X):
        if not P_set and not X:
            if len(R) > len(best):
                best[:] = R
            return
        
        # Pivot
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
    return list(best)

def _greedy_expand(adj, current_branches, deadline):
    """
    Greedily try to add vertices to current_branches.
    Returns the expanded set and paths if successful.
    """
    current = list(current_branches)
    paths = _try_assign_paths(adj, current, deadline)
    if paths is None:
        return current, None
    
    n = P['n']
    # Candidates sorted by degree (high degree first)
    candidates = sorted([v for v in range(n) if v not in current], key=lambda x: -len(adj[x]))
    
    for cand in candidates:
        if time.time() > deadline:
            break
        trial = current + [cand]
        tp = _try_assign_paths(adj, trial, deadline)
        if tp is not None:
            current = trial
            paths = tp
            
    return current, paths

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    adj = _build_adj()
    n = P['n']
    
    # 1. Find Max Clique as initial seed
    max_c = _max_clique(adj)
    best_branches = list(max_c)
    best_paths = _try_assign_paths(adj, best_branches, deadline)
    
    # If max clique doesn't work, try smaller subsets
    if best_paths is None:
        while len(best_branches) > 2 and best_paths is None:
            best_branches.pop()
            best_paths = _try_assign_paths(adj, best_branches, deadline)
    
    # 2. Greedy Expansion from max clique
    if best_paths is not None:
        expanded, exp_paths = _greedy_expand(adj, best_branches, deadline)
        if len(expanded) > len(best_branches):
            best_branches = expanded
            best_paths = exp_paths
    
    # 3. Randomized Restarts with Backtracking
    random.seed(123)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Random start: pick a random subset of vertices
        size = random.randint(3, min(10, n))
        init = random.sample(range(n), size)
        
        # Check if init is valid
        ip = _try_assign_paths(adj, init, deadline)
        if ip is None:
            continue
        
        current = list(init)
        current_paths = ip
        
        # Greedy expansion
        remaining = sorted([v for v in range(n) if v not in current], key=lambda x: -len(adj[x]))
        for cand in remaining:
            if time.time() > deadline:
                break
            trial = current + [cand]
            tp = _try_assign_paths(adj, trial, deadline)
            if tp is not None:
                current = trial
                current_paths = tp
        
        if len(current) > len(best_branches):
            best_branches = list(current)
            best_paths = current_paths
            
    # 4. Swap-based Local Search
    for _ in range(200):
        if time.time() > deadline:
            break
        
        if len(best_branches) <= 2:
            break
            
        # Pick a random vertex to remove
        idx = random.randint(0, len(best_branches) - 1)
        removed = best_branches[idx]
        trial_branches = [b for i, b in enumerate(best_branches) if i != idx]
        
        # Pick a random vertex to add
        pool = [v for v in range(n) if v not in trial_branches]
        if not pool:
            break
        added = random.choice(pool)
        trial_branches.append(added)
        
        # Check validity
        tp = _try_assign_paths(adj, trial_branches, deadline)
        if tp is not None:
            best_branches = trial_branches
            best_paths = tp
            
            # Try to expand from this new state
            current = list(best_branches)
            current_paths = best_paths
            remaining = sorted([v for v in range(n) if v not in current], key=lambda x: -len(adj[x]))
            for cand in remaining:
                if time.time() > deadline:
                    break
                trial = current + [cand]
                ttp = _try_assign_paths(adj, trial, deadline)
                if ttp is not None:
                    current = trial
                    current_paths = ttp
                    if len(current) > len(best_branches):
                        best_branches = list(current)
                        best_paths = current_paths
                        break

    # Final verification and formatting
    if best_paths is None:
        # Fallback
        u, v = P['edges'][0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}
        
    # Ensure types are correct
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
