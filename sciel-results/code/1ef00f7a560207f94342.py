# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from collections import deque
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)
adj_list = [sorted(adj[i]) for i in range(N)]

DEADLINE = time.time() + 80

def time_left():
    return DEADLINE - time.time()

def greedy_coloring(candidates, adj_set):
    """Greedy coloring of candidates, return (num_colors, color_of)"""
    colors = {}
    for v in candidates:
        used = set()
        for u in candidates:
            if u != v and u in colors and u in adj_set[v]:
                used.add(colors[u])
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    if not colors:
        return 0, colors
    return max(colors.values()) + 1, colors

def max_clique_branch_bound(adj_set, deadline):
    """Find maximum clique using branch and bound with greedy coloring."""
    n = len(adj_set)
    best_clique = []
    
    def expand(candidates, current):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not candidates:
            if len(current) > len(best_clique):
                best_clique = list(current)
            return
        
        num_colors, color_map = greedy_coloring(candidates, adj_set)
        if len(current) + num_colors <= len(best_clique):
            return
        
        # Process candidates in reverse color order
        for i in range(len(candidates) - 1, -1, -1):
            if time.time() > deadline:
                return
            v = candidates[i]
            if len(current) + i + 1 <= len(best_clique):
                break
            new_candidates = [u for u in candidates[:i] if u in adj_set[v]]
            new_current = current + [v]
            expand(new_candidates, new_current)
            candidates = candidates[:i]
            # Rebound after removing v
            num_colors, _ = greedy_coloring(candidates, adj_set)
            if len(current) + num_colors <= len(best_clique):
                return
    
    expand(list(range(n)), [])
    return best_clique

def find_disjoint_paths(branches):
    """
    Given branch vertices, find one simple path for each unordered pair
    with pairwise disjoint interiors avoiding all branches.
    Returns list of paths or None.
    """
    n_b = len(branches)
    if n_b < 2:
        return None
    branch_set = set(branches)
    pairs = list(combinations(range(n_b), 2))
    
    # Sort pairs: direct edges first (shortest paths)
    def pair_key(p):
        u, v = branches[p[0]], branches[p[1]]
        return 0 if v in adj[u] else 1
    pairs.sort(key=pair_key)
    
    used_interior = set()
    paths = {}
    
    for (i, j) in pairs:
        s, t = branches[i], branches[j]
        parent = {}
        visited = {s}
        q = deque([s])
        found = False
        while q:
            u = q.popleft()
            if u == t:
                found = True
                break
            for w in adj_list[u]:
                if w in visited:
                    continue
                if w in branch_set and w != t:
                    continue
                if w in used_interior:
                    continue
                visited.add(w)
                parent[w] = u
                q.append(w)
        if not found:
            return None
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        for v in path[1:-1]:
            used_interior.add(v)
        paths[(i, j)] = path
    
    result = []
    for (i, j) in combinations(range(n_b), 2):
        result.append(paths[(i, j)])
    return result

def try_branch_set(branches):
    return find_disjoint_paths(branches)

def local_search_extend(branches, deadline):
    """Try to add one more vertex to the branch set."""
    n_b = len(branches)
    for candidate in range(N):
        if candidate in branches:
            continue
        if time_left() < 2:
            break
        trial = list(branches) + [candidate]
        paths = try_branch_set(trial)
        if paths is not None:
            return trial, paths
    return None, None

def local_search_swap(branches, deadline):
    """Try to swap a branch vertex with another to enable further extension."""
    n_b = len(branches)
    for i in range(n_b):
        if time_left() < 2:
            break
        for candidate in range(N):
            if candidate in branches:
                continue
            trial = list(branches)
            trial[i] = candidate
            paths = try_branch_set(trial)
            if paths is not None:
                # Try to extend
                extended, ext_paths = local_search_extend(trial, deadline)
                if extended is not None and len(extended) > n_b:
                    return extended, ext_paths
                elif paths is not None:
                    # Keep the swap if it allows same size but different structure
                    pass
    return None, None

def search_from_seed(seed_branches, deadline):
    """Starting from a seed, try to build up the largest routable branch set."""
    branches = list(seed_branches)
    paths = try_branch_set(branches)
    if paths is None:
        return None, None
    
    improved = True
    while improved and time_left() > 3:
        improved = False
        # Try adding vertices
        for candidate in range(N):
            if candidate in branches:
                continue
            if time_left() < 2:
                break
            trial = list(branches) + [candidate]
            trial_paths = try_branch_set(trial)
            if trial_paths is not None:
                branches = trial
                paths = trial_paths
                improved = True
                break
        # Try swaps if no addition worked
        if not improved and time_left() > 3:
            for i in range(len(branches)):
                if time_left() < 2:
                    break
                for candidate in range(N):
                    if candidate in branches:
                        continue
                    trial = list(branches)
                    trial[i] = candidate
                    trial_paths = try_branch_set(trial)
                    if trial_paths is not None:
                        # Check if this swap allows further extension
                        extended, ext_paths = local_search_extend(trial, deadline)
                        if extended is not None and len(extended) > len(branches):
                            branches = extended
                            paths = ext_paths
                            improved = True
                            break
                if improved:
                    break
    return branches, paths

def construct():
    global DEADLINE
    DEADLINE = time.time() + 80
    
    best_branches = None
    best_paths = None
    best_count = 1
    
    # Fallback
    u0, v0 = P['edges'][0]
    fallback_paths = try_branch_set([u0, v0])
    if fallback_paths is not None:
        best_branches = [u0, v0]
        best_paths = fallback_paths
        best_count = 2
    
    # Strategy 1: Find maximum clique, then find largest routable subset
    if time_left() > 10:
        clique = max_clique_branch_bound(adj, DEADLINE - 5)
        if clique and len(clique) >= 2:
            # Try the full clique
            paths = try_branch_set(clique)
            if paths is not None:
                best_branches = list(clique)
                best_paths = paths
                best_count = len(clique)
            else:
                # Try subsets of the clique in decreasing size
                for k in range(len(clique) - 1, 1, -1):
                    if time_left() < 5:
                        break
                    found = False
                    for subset in combinations(clique, k):
                        if time_left() < 3:
                            break
                        subset_list = list(subset)
                        paths = try_branch_set(subset_list)
                        if paths is not None:
                            if len(subset_list) > best_count:
                                best_branches = subset_list
                                best_paths = paths
                                best_count = len(subset_list)
                            found = True
                            break
                    if found:
                        break
    
    # Strategy 2: Greedy from multiple seeds
    if time_left() > 10:
        # Seed from high-degree vertices
        deg = [len(adj[i]) for i in range(N)]
        order = sorted(range(N), key=lambda x: -deg[x])
        
        # Try building from top-degree pairs
        tried_seeds = set()
        for i in range(min(10, N)):
            for j in range(i+1, min(15, N)):
                if time_left() < 5:
                    break
                seed = [order[i], order[j]]
                key = tuple(sorted(seed))
                if key in tried_seeds:
                    continue
                tried_seeds.add(key)
                b, p = search_from_seed(seed, DEADLINE)
                if b is not None and len(b) > best_count:
                    best_count = len(b)
                    best_branches = b
                    best_paths = p
    
    # Strategy 3: Local search from best found
    if best_branches is not None and time_left() > 5:
        extended, ext_paths = local_search_extend(best_branches, DEADLINE)
        if extended is not None and len(extended) > best_count:
            best_count = len(extended)
            best_branches = extended
            best_paths = ext_paths
        
        swapped, swap_paths = local_search_swap(best_branches, DEADLINE)
        if swapped is not None and len(swapped) > best_count:
            best_count = len(swapped)
            best_branches = swapped
            best_paths = swap_paths
    
    # Final validation and cleanup
    if best_branches is None or len(best_branches) < 2:
        best_branches = [u0, v0]
        best_paths = fallback_paths
    
    # Ensure types are correct
    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(v) for v in p] for p in best_paths]
    
    # Validate
    branch_set = set(branches_out)
    used_interior = set()
    for p in paths_out:
        if len(p) < 2:
            return {'branches': [u0, v0], 'paths': [[u0, v0]]}
        if p[0] not in branch_set or p[-1] not in branch_set:
            return {'branches': [u0, v0], 'paths': [[u0, v0]]}
        if p[0] == p[-1]:
            return {'branches': [u0, v0], 'paths': [[u0, v0]]}
        for v in p[1:-1]:
            if v in branch_set:
                return {'branches': [u0, v0], 'paths': [[u0, v0]]}
            if v in used_interior:
                return {'branches': [u0, v0], 'paths': [[u0, v0]]}
            used_interior.add(v)
        for a, b in zip(p, p[1:]):
            if b not in adj[a]:
                return {'branches': [u0, v0], 'paths': [[u0, v0]]}
    
    # Check all pairs covered
    n_b = len(branches_out)
    expected_pairs = n_b * (n_b - 1) // 2
    if len(paths_out) != expected_pairs:
        return {'branches': [u0, v0], 'paths': [[u0, v0]]}
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
