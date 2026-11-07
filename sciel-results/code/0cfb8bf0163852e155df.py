<<<<<<< SEARCH
"""Rewritten program: explicit search over vertex sets for max edge-disjoint Hamiltonian cycles."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(vertices, adj, used_edges, timeout_per_try=None, start=None):
    """Find a Hamiltonian cycle on vertex set using only unused edges. Fast DFS with Warnsdorff + edge-pruning."""
    k = len(vertices)
    if k < 3:
        return None
    vset = set(vertices)
    if start is None:
        start = min(vset)
    if start not in vset:
        return None
    # quick degree feasibility
    for v in vertices:
        deg = 0
        for nb in adj[v]:
            if nb in vset and (min(v, nb), max(v, nb)) not in used_edges:
                deg += 1
        if deg < 2:
            return None

    path = [start]
    visited = set([start])
    deadline_local = time.time() + timeout_per_try if timeout_per_try else None
    deadline_check_interval = 100
    call_count = [0]

    sys.setrecursionlimit(10000)

    # Precompute available edges within vset for faster lookup
    avail = set()
    for u in vset:
        for v in adj[u]:
            if v in vset:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    avail.add(e)

    def dfs(cur):
        call_count[0] += 1
        if deadline_local and call_count[0] % deadline_check_interval == 0 and time.time() > deadline_local:
            return None
        if len(path) == k:
            last = path[-1]
            e_close = (min(last, start), max(last, start))
            if e_close in avail:
                return list(path)
            return None
        remaining = k - len(path)
        # candidates: unvisited vertices in vset adjacent to cur via unused edge
        cands = []
        for nxt in adj[cur]:
            if nxt not in visited and nxt in vset:
                e = (min(cur, nxt), max(cur, nxt))
                if e not in avail:
                    continue
                # Warnsdorff: prefer nxt with fewer onward options
                if remaining == 1:
                    # must be able to close to start
                    e_close = (min(nxt, start), max(nxt, start))
                    if e_close not in avail:
                        continue
                    onward = 0
                else:
                    onward = 0
                    for nn in adj[nxt]:
                        if nn not in visited and nn in vset and nn != cur:
                            if (min(nxt, nn), max(nxt, nn)) in avail:
                                onward += 1
                    # Additional pruning: if onward == 0 and remaining > 1, dead end
                    if onward == 0:
                        continue
                cands.append((onward, nxt))
        if not cands:
            return None
        cands.sort()
        for _, nxt in cands:
            e = (min(cur, nxt), max(cur, nxt))
            visited.add(nxt)
            path.append(nxt)
            res = dfs(nxt)
            if res is not None:
                return res
            path.pop()
            visited.discard(nxt)
        return None

    return dfs(start)
=======
"""Rewritten program: explicit search over vertex sets for max edge-disjoint Hamiltonian cycles."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bipartite_matching(n_left, left_nodes, right_nodes, adj_lr):
    """Kuhn's algorithm for bipartite matching. adj_lr maps left node -> list of right nodes."""
    match_r = {}
    def try_augment(u, visited):
        for v in adj_lr.get(u, []):
            if v not in visited:
                visited.add(v)
                if v not in match_r or try_augment(match_r[v], visited):
                    match_r[v] = u
                    return True
        return False
    for u in left_nodes:
        try_augment(u, set())
    return match_r

def _find_2factor(vertices, adj, used_edges):
    """Find a 2-regular spanning subgraph (2-factor) on vertices using bipartite matching.
    Returns list of cycles (each a list of vertices) or None if no 2-factor exists."""
    k = len(vertices)
    if k < 3:
        return None
    vset = set(vertices)
    
    # Check each vertex has degree >= 2 in available edges
    for v in vertices:
        deg = 0
        for nb in adj[v]:
            if nb in vset and (min(v, nb), max(v, nb)) not in used_edges:
                deg += 1
        if deg < 2:
            return None
    
    # Build bipartite graph: left = vertices (as "out"), right = vertices (as "in")
    # Edge u->v exists if (u,v) is an available undirected edge
    # We need to match each left vertex to exactly one right vertex and vice versa
    # such that no vertex is matched to itself (to avoid 2-cycles in the directed sense,
    # but actually we want a 2-regular undirected subgraph)
    
    # Actually, for a 2-factor: each vertex has exactly 2 incident edges in the subgraph.
    # This is equivalent to finding a perfect matching in the line graph, but that's complex.
    # Alternative: use the fact that a 2-factor in an undirected graph can be found by
    # reducing to a perfect matching problem via a standard reduction.
    
    # Simpler approach: just try to find it via greedy + repair, or use the fact that
    # for small graphs we can use DFS with memoization.
    
    # For now, let's use a different strategy: find a 2-factor by iteratively
    # selecting edges. This is NP-hard in general, but for our small graphs it should work.
    
    # Actually, let's skip this for now and focus on improving DFS.
    return None

def _find_ham_cycle(vertices, adj, used_edges, timeout_per_try=None, start=None):
    """Find a Hamiltonian cycle on vertex set using only unused edges. Fast DFS with Warnsdorff + degree-pruning."""
    k = len(vertices)
    if k < 3:
        return None
    vset = set(vertices)
    if start is None:
        start = min(vset)
    if start not in vset:
        return None
    # quick degree feasibility
    for v in vertices:
        deg = 0
        for nb in adj[v]:
            if nb in vset and (min(v, nb), max(v, nb)) not in used_edges:
                deg += 1
        if deg < 2:
            return None

    path = [start]
    visited = set([start])
    deadline_local = time.time() + timeout_per_try if timeout_per_try else None
    deadline_check_interval = 50
    call_count = [0]

    sys.setrecursionlimit(10000)

    # Precompute available edges within vset for faster lookup
    avail = set()
    for u in vset:
        for v in adj[u]:
            if v in vset:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    avail.add(e)
    
    # Precompute adjacency in available graph
    avail_adj = defaultdict(set)
    for u, v in avail:
        avail_adj[u].add(v)
        avail_adj[v].add(u)

    def dfs(cur):
        call_count[0] += 1
        if deadline_local and call_count[0] % deadline_check_interval == 0 and time.time() > deadline_local:
            return None
        if len(path) == k:
            last = path[-1]
            e_close = (min(last, start), max(last, start))
            if e_close in avail:
                return list(path)
            return None
        remaining = k - len(path)
        # candidates: unvisited vertices in vset adjacent to cur via unused edge
        cands = []
        for nxt in avail_adj[cur]:
            if nxt not in visited and nxt in vset:
                # Warnsdorff: prefer nxt with fewer onward options
                if remaining == 1:
                    # must be able to close to start
                    e_close = (min(nxt, start), max(nxt, start))
                    if e_close not in avail:
                        continue
                    onward = 0
                else:
                    onward = 0
                    for nn in avail_adj[nxt]:
                        if nn not in visited and nn in vset and nn != cur:
                            onward += 1
                    # Additional pruning: if onward == 0 and remaining > 1, dead end
                    if onward == 0:
                        continue
                    # Degree pruning: check that all unvisited vertices still have degree >= 2
                    # in the remaining graph (excluding cur and nxt which are being used)
                    # This is expensive, so only do it for small remaining sets
                    if remaining <= 6:
                        feasible = True
                        for unv in vset:
                            if unv in visited or unv == nxt or unv == cur:
                                continue
                            # Count available neighbors of unv that are unvisited and not cur/nxt
                            cnt = 0
                            for nn in avail_adj[unv]:
                                if nn not in visited and nn != cur and nn != nxt and nn in vset:
                                    cnt += 1
                            if cnt < 2:
                                feasible = False
                                break
                        if not feasible:
                            continue
                cands.append((onward, nxt))
        if not cands:
            return None
        cands.sort()
        for _, nxt in cands:
            e = (min(cur, nxt), max(cur, nxt))
            visited.add(nxt)
            path.append(nxt)
            res = dfs(nxt)
            if res is not None:
                return res
            path.pop()
            visited.discard(nxt)
        return None

    return dfs(start)
>>>>>>> REPLACE

