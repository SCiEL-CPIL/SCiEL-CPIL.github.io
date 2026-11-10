"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _rnd
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_DEADLINE = _time.monotonic() + 82.0


def _time_left():
    return _time.monotonic() < _DEADLINE


def _max_flow(n_nodes, adj_list, cap, s, t):
    """Dinic's algorithm. Returns (flow_value, parent) where parent maps node->prev node on flow paths."""
    parent = {}
    total = 0
    # work on a mutable copy
    cap_local = [row[:] for row in cap]
    while True:
        # BFS levels
        level = [-1] * n_nodes
        q = deque([s])
        level[s] = 0
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if level[v] < 0 and cap_local[u][v] > 0:
                    level[v] = level[u] + 1
                    q.append(v)
        if level[t] < 0:
            break
        # DFS blocking flow with iterative stack
        def dfs(u, f):
            if u == t:
                return f
            for i in range(len(adj_list[u])):
                v = adj_list[u][i]
                if level[v] == level[u] + 1 and cap_local[u][v] > 0:
                    ret = dfs(v, min(f, cap_local[u][v]))
                    if ret > 0:
                        cap_local[u][v] -= ret
                        cap_local[v][u] += ret
                        parent[v] = u
                        return ret
            return 0
        while True:
            pushed = dfs(s, 10**9)
            if pushed == 0:
                break
            total += pushed
    return total, parent


def _solve_with_flow(branches, adj, n):
    """Given a branch set (clique), build flow network and try to realize paths.
    Returns list of paths or None."""
    Sset = set(branches)
    pairs = list(combinations(branches, 2))
    m = len(pairs)
    if m == 0:
        return None

    # Node indexing:
    # 0: super source, 1: super sink
    # 2 + i : branch i (i in 0..k-1)
    # 2 + k + j : non-branch vertex j
    k = len(branches)
    bidx = {b: i for i, b in enumerate(branches)}
    nb_verts = [v for v in range(n) if v not in Sset]
    nidx = {v: 2 + k + j for j, v in enumerate(nb_verts)}

    # total nodes
    N = 2 + k + len(nb_verts)
    INF = 10**9

    # Build adjacency list (undirected in graph sense, but flow is directed with reverse edges)
    # We'll use a dict-of-dict for capacity: cap[u][v]
    cap = [[0] * N for _ in range(N)]
    adj_list = [set() for _ in range(N)]

    def add_edge(u, v, c):
        if cap[u][v] == 0 and cap[v][u] == 0:
            adj_list[u].add(v)
            adj_list[v].add(u)
        cap[u][v] += c

    # Source to branches
    for i, b in enumerate(branches):
        add_edge(0, 2 + i, INF)
    # Branches to sink
    for i, b in enumerate(branches):
        add_edge(2 + i, 1, INF)
    # Branch to non-branch neighbors
    for i, b in enumerate(branches):
        nb_node = 2 + i
        for nb in adj[b]:
            if nb in Sset:
                continue
            nn = nidx[nb]
            add_edge(nb_node, nn, INF)
            add_edge(nn, nb_node, INF)
    # Non-branch to non-branch neighbors
    for nb in nb_verts:
        nn = nidx[nb]
        for nb2 in adj[nb]:
            if nb2 in Sset:
                continue
            nn2 = nidx[nb2]
            add_edge(nn, nn2, 1)
            add_edge(nn2, nn, 1)

    s = 0
    t = 1
    flow_val, parent = _max_flow(N, [list(s) for s in adj_list], cap, s, t)
    if flow_val < m:
        return None

    # Reconstruct paths from parent pointers
    # Each unit of flow from source goes to a branch node, through non-branch nodes, to a branch node, to sink.
    # We need to trace each flow path.
    # Since capacities are 1 on non-branch edges, each path is a simple path.
    # We'll trace by following parent from sink backwards? No, parent maps v->u meaning flow went u->v.
    # Let's trace forward: from source, find which branch nodes received flow, then follow.
    # Actually, let's trace each path by finding all nodes with parent set and grouping.
    # Better: for each branch node (2+i), find the path from source to that branch (inflow), then from that branch to sink (outflow).
    # But flow can enter a branch from multiple non-branch nodes. Since branch has INF capacity, multiple flows can pass.
    # Let's trace each unit of flow separately by following the residual graph? 
    # Alternative: collect all edges with flow > 0, then decompose.
    
    # Simpler: for each branch pair, we know a path exists. Let's find paths by DFS in the flow graph.
    # Build flow graph: edge u->v has flow if cap_original - cap_current > 0.
    # But we mutated cap. Let's rebuild: we stored original caps implicitly.
    # Actually, let's just trace from each branch node to find its paired branch.
    
    # For each branch node bn = 2+i, trace back to source to find where it came from (non-branch node),
    # then trace forward from that non-branch node to find which branch it goes to.
    
    paths = []
    # For each branch, find one incoming flow path from source
    for i, b in enumerate(branches):
        bn = 2 + i
        # Find a non-branch node that sent flow to bn
        prev_nb = None
        for nb in nb_verts:
            nn = nidx[nb]
            # Check if there was flow from nn to bn
            # We need to check if cap[nn][bn] was reduced. We don't have original caps easily.
            # Let's use parent: if parent[bn] == nn, then flow came from nn to bn.
            if parent.get(bn) == nn:
                prev_nb = nb
                break
        if prev_nb is None:
            # Maybe direct? No, source->branch is INF, but flow must come through non-branch for path.
            # Actually source->branch->...->branch->sink. The flow into branch bn comes from a non-branch node.
            # Let's search more carefully.
            for nb in nb_verts:
                nn = nidx[nb]
                # Check if parent[bn] could be nn
                if parent.get(bn) == nn:
                    prev_nb = nb
                    break
        if prev_nb is None:
            return None  # shouldn't happen if flow == m
        
        # Now trace from prev_nb forward to find the destination branch
        cur = prev_nb
        dest_branch = None
        visited = set()
        while True:
            if cur in Sset:
                dest_branch = cur
                break
            if cur in visited:
                return None
            visited.add(cur)
            # Find next node: a node v such that parent[v] == nidx[cur] or parent[nidx[cur]] ... 
            # Actually, we need to follow the flow direction. parent[v] = u means flow u->v.
            # So from cur, find v where parent[v] == nidx[cur]
            next_node = None
            for v in range(N):
                if parent.get(v) == nidx[cur]:
                    next_node = v
                    break
            if next_node is None:
                return None
            if next_node < 2 + k:
                # It's a branch node
                dest_branch = branches[next_node - 2]
                break
            else:
                # It's a non-branch node
                cur = nb_verts[next_node - (2 + k)]
        
        if dest_branch is None or dest_branch == b:
            return None
        
        # Reconstruct full path: b -> ... -> prev_nb -> ... -> dest_branch
        # Backward from prev_nb to b:
        back_path = [prev_nb]
        cur = prev_nb
        while cur != b:
            # Find u where parent[nidx[cur]] == nidx[u] or parent[2+i] ...
            # Actually, parent maps node index to node index.
            # From cur (non-branch), find u such that parent[nidx[cur]] == nidx[u] (u non-branch) or parent[nidx[cur]] == 2+j (u branch)
            nn = nidx[cur]
            prev_idx = parent.get(nn)
            if prev_idx is None:
                return None
            if prev_idx < 2 + k:
                # prev is branch
                back_path.append(branches[prev_idx - 2])
                if branches[prev_idx - 2] != b:
                    return None
                break
            else:
                prev_v = nb_verts[prev_idx - (2 + k)]
                back_path.append(prev_v)
                cur = prev_v
        
        # Forward from prev_nb to dest_branch:
        fwd_path = [prev_nb]
        cur = prev_nb
        while cur != dest_branch:
            nn = nidx[cur]
            next_idx = None
            for v in range(N):
                if parent.get(v) == nn:
                    next_idx = v
                    break
            if next_idx is None:
                return None
            if next_idx < 2 + k:
                fwd_path.append(branches[next_idx - 2])
                if branches[next_idx - 2] != dest_branch:
                    return None
                break
            else:
                next_v = nb_verts[next_idx - (2 + k)]
                fwd_path.append(next_v)
                cur = next_v
        
        full_path = [b] + back_path[:0:-1] + fwd_path[1:]
        # Ensure no duplicates in interior
        interior = full_path[1:-1]
        if len(interior) != len(set(interior)):
            return None
        paths.append(full_path)
    
    # Verify all pairs covered
    path_pairs = set()
    for p in paths:
        a, b = p[0], p[-1]
        path_pairs.add(frozenset((a, b)))
    if len(path_pairs) != m:
        return None
    
    return paths


def construct():
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for a, b in edges:
        a = int(a)
        b = int(b)
        adj[a].add(b)
        adj[b].add(a)

    # Fallback
    u0, v0 = int(edges[0][0]), int(edges[0][1])
    best_branches = [u0, v0]
    best_paths = [[u0, v0]]
    best_k = 2

    def is_clique(S):
        for x, y in combinations(S, 2):
            if y not in adj[x]:
                return False
        return True

    # Simulated annealing over branch sets
    _rnd.seed(42)
    cur = None
    cur_k = 0
    best_sa = None
    best_sa_k = 0

    # Initialize with a random clique of size 2
    cur = [u0, v0]
    res = _solve_with_flow(cur, adj, n)
    if res is not None:
        best_sa = cur
        best_sa_k = 2
        best_branches = cur[:]
        best_paths = res
        best_k = 2

    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    max_iter = 20000

    for it in range(max_iter):
        if not _time_left():
            break
        if T < T_min:
            break
        
        # Propose a move: add, remove, or swap a vertex
        move_type = _rnd.random()
        new_cur = cur[:]
        
        if move_type < 0.4 and len(new_cur) < n:
            # Add a vertex
            candidates = [v for v in range(n) if v not in set(new_cur)]
            if not candidates:
                continue
            v = _rnd.choice(candidates)
            new_cur.append(v)
        elif move_type < 0.7 and len(new_cur) > 2:
            # Remove a vertex
            v = _rnd.choice(new_cur)
            new_cur.remove(v)
        else:
            # Swap
            if len(new_cur) == 0:
                continue
            out_v = _rnd.choice(new_cur)
            candidates = [v for v in range(n) if v not in set(new_cur)]
            if not candidates:
                continue
            in_v = _rnd.choice(candidates)
            new_cur.remove(out_v)
            new_cur.append(in_v)
        
        if len(new_cur) < 2:
            continue
        if not is_clique(new_cur):
            # Reject non-clique
            T *= cooling
            continue
        
        # Evaluate
        res_new = _solve_with_flow(new_cur, adj, n)
        if res_new is not None:
            new_k = len(new_cur)
            if new_k > best_sa_k:
                best_sa = new_cur[:]
                best_sa_k = new_k
                best_branches = new_cur[:]
                best_paths = res_new
                best_k = new_k
            # Accept if better or with probability
            if new_k > cur_k:
                cur = new_cur
                cur_k = new_k
            else:
                # Same size or worse; accept with prob exp((new_k - cur_k)/T) but since k is int, only if equal
                if new_k == cur_k:
                    if _rnd.random() < 0.5:
                        cur = new_cur
                # else reject
        else:
            # Not realizable; reject
            pass
        
        T *= cooling

    # Final: try to improve best_sa by greedy addition
    if best_sa is not None and _time_left():
        cur_best = best_sa[:]
        cur_set = set(cur_best)
        for v in range(n):
            if v in cur_set:
                continue
            if not _time_left():
                break
            ok = True
            for x in cur_best:
                if v not in adj[x]:
                    ok = False
                    break
            if not ok:
                continue
            trial = cur_best + [v]
            res = _solve_with_flow(trial, adj, n)
            if res is not None:
                cur_best = trial
                cur_set.add(v)
                best_branches = cur_best[:]
                best_paths = res
                best_k = len(cur_best)

    # Final type conversion
    best_branches = [int(x) for x in best_branches]
    best_paths = [[int(x) for x in p] for p in best_paths]

    if len(best_branches) < 2:
        best_branches = [u0, v0]
        best_paths = [[u0, v0]]

    return {'branches': best_branches, 'paths': best_paths}
# EVOLVE-BLOCK-END
