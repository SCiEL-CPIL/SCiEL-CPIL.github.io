"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_path(adj, src, dst, blocked):
    """Find shortest path from src to dst avoiding blocked vertices (interior only)."""
    if src == dst:
        return [src, dst]
    parent = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb == dst:
                # Reconstruct path
                path = [nb]
                node = cur
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            if nb not in parent and nb not in blocked:
                parent[nb] = cur
                q.append(nb)
    return None

def _find_disjoint_paths(adj, branches, n, deadline):
    """
    Given a set of branch vertices, find vertex-disjoint paths between all pairs.
    Path interiors must be disjoint and avoid all branch vertices.
    Returns dict of (u,v) -> path, or None if impossible.
    """
    branch_set = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    paths = {}
    blocked = set()  # interior vertices used
    
    # Try to find paths greedily, shortest first
    # Order pairs by BFS distance to prioritize "easy" pairs
    pair_order = []
    for (u, v) in pairs:
        p = _bfs_path(adj, u, v, blocked)
        if p is None:
            return None
        pair_order.append((len(p), u, v))
    pair_order.sort()
    
    for _, u, v in pair_order:
        if time.time() > deadline:
            return None
        # Find path avoiding currently blocked vertices
        best_path = None
        best_len = float('inf')
        # Try BFS
        p = _bfs_path(adj, u, v, blocked)
        if p is not None:
            best_path = p
            best_len = len(p)
        
        if best_path is None:
            return None
        
        # Mark interior vertices as blocked
        for mid in best_path[1:-1]:
            blocked.add(mid)
        paths[(min(u,v), max(u,v))] = best_path
    
    return paths

def _try_branch_set(adj, branches, n, deadline):
    """Try to find disjoint paths for given branch set. Returns result or None."""
    if len(branches) < 2:
        return None
    paths = _find_disjoint_paths(adj, branches, n, deadline)
    if paths is None:
        return None
    # Build result
    branch_list = sorted(branches)
    path_list = []
    for (u, v) in combinations(branch_list, 2):
        path_list.append(list(paths[(u, v)]))
    return {'branches': branch_list, 'paths': path_list}

def _tomita_cliques(adj, n, deadline):
    """Find all maximal cliques using Tomita's algorithm."""
    cliques = []
    def expand(R, P):
        if time.time() > deadline:
            return
        if not P:
            cliques.append(list(R))
            return
        # Color sort for efficiency
        # Simple version: just iterate
        for v in list(P):
            P.remove(v)
            expand(R | {v}, P & adj[v])
    
    expand(set(), set(range(n)))
    return cliques

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = P['n']
    adj = _build_adj(n, P['edges'])
    
    best_result = None
    best_score = 0
    
    # Strategy 1: Find all cliques (up to size ~8) and try them
    # Use Tomita with early termination
    cliques_found = []
    
    def _expand_cliques(R, P, Q):
        if time.time() > deadline:
            return
        if not P:
            if len(R) >= 2:
                cliques_found.append(frozenset(R))
            return
        # Greedy coloring to bound
        # Simple: iterate
        for v in list(P):
            P.discard(v)
            _expand_cliques(R | {v}, P & adj[v], Q)
    
    _expand_cliques(set(), set(range(n)), set(range(n)))
    
    # Sort cliques by size descending
    cliques_found = sorted(set(cliques_found), key=len, reverse=True)
    
    # Try large cliques first
    for clique in cliques_found:
        if time.time() > deadline:
            break
        if len(clique) <= best_score:
            continue
        result = _try_branch_set(adj, list(clique), n, deadline)
        if result is not None:
            score = len(result['branches'])
            if score > best_score:
                best_score = score
                best_result = result
    
    # Strategy 2: Greedy expansion - start with a clique, try to add vertices
    # For each maximal clique, try adding non-clique vertices
    if best_result is not None:
        current_branches = set(best_result['branches'])
    else:
        # Fallback: use first edge
        u, v = P['edges'][0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
        best_score = 2
        current_branches = {u, v}
    
    # Try to expand by adding one vertex at a time
    expanded = True
    while expanded:
        expanded = False
        if time.time() > deadline:
            break
        candidates = [v for v in range(n) if v not in current_branches]
        # Sort by degree descending
        candidates.sort(key=lambda v: len(adj[v]), reverse=True)
        for cand in candidates:
            if time.time() > deadline:
                break
            new_branches = list(current_branches) + [cand]
            result = _try_branch_set(adj, new_branches, n, deadline)
            if result is not None:
                score = len(result['branches'])
                if score > best_score:
                    best_score = score
                    best_result = result
                    current_branches = set(result['branches'])
                    expanded = True
                    break
    
    # Strategy 3: Randomized search - random subsets of vertices
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        # Try random sets of size 4-7
        for size in [4, 5, 6, 7]:
            if size <= best_score:
                continue
            subset = random.sample(range(n), size)
            result = _try_branch_set(adj, subset, n, deadline)
            if result is not None:
                score = len(result['branches'])
                if score > best_score:
                    best_score = score
                    best_result = result
    
    # Ensure valid types
    if best_result is None:
        u, v = P['edges'][0]
        best_result = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    else:
        best_result['branches'] = [int(b) for b in best_result['branches']]
        best_result['paths'] = [[int(x) for x in p] for p in best_result['paths']]
    
    return best_result

# EVOLVE-BLOCK-END
