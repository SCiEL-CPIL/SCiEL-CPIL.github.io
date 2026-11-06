import time
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']
m = len(edge_order)

# Arrival index (0-based) for each unordered edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    arrival[(min(u, v), max(u, v))] = idx

def edge_arrival(u, v):
    if u > v:
        u, v = v, u
    return arrival[(u, v)]

def cycle_score(cycle):
    s = 0
    L = len(cycle)
    for i in range(L):
        a = cycle[i]
        b = cycle[(i + 1) % L]
        ar = edge_arrival(a, b) + 1
        if ar > s:
            s = ar
    return s

def is_hamiltonian(cycle):
    if len(cycle) != n:
        return False
    if len(set(cycle)) != n:
        return False
    for v in range(n):
        if v not in set(cycle):
            return False
    return True

def find_hc_prefix(K, deadline, start=0):
    """Find Hamilton cycle using only edges with arrival <= K (0-based).
       Returns list of vertices or None."""
    # Build adjacency with only edges arrival <= K
    adj = [set() for _ in range(n)]
    for idx in range(K + 1):
        u, v = edge_order[idx]
        adj[u].add(v)
        adj[v].add(u)
    # Degree check
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    # Precompute neighbor list sorted by arrival (heuristic: try low-arrival first)
    adj_list = []
    for v in range(n):
        nb = list(adj[v])
        nb.sort(key=lambda x: edge_arrival(v, x))
        adj_list.append(nb)
    
    visited = [False] * n
    path = [start]
    visited[start] = True
    found = [None]
    
    # Connectivity pruning helper: ensure unvisited set (plus current) stays connected
    # and that start is reachable from unvisited for closing.
    # We'll do a lightweight articulation/bridge check on the subgraph induced by
    # unvisited vertices plus current vertex.
    
    def connected_component_ok(cur):
        # Check that the set of unvisited vertices (plus cur) induces a connected
        # component containing cur, and that start has a path into it via unvisited+cur
        # (start will connect to the last vertex).
        # Simple BFS over allowed vertices: cur + unvisited
        allowed = [False] * n
        for v in range(n):
            if not visited[v]:
                allowed[v] = True
        allowed[cur] = True
        # BFS from cur
        seen = [False] * n
        dq = deque([cur])
        seen[cur] = True
        cnt = 0
        while dq:
            u = dq.popleft()
            cnt += 1
            for w in adj_list[u]:
                if allowed[w] and not seen[w]:
                    seen[w] = True
                    dq.append(w)
        # Every allowed vertex must be reached
        for v in range(n):
            if allowed[v] and not seen[v]:
                return False
        return True
    
    # Also need: for closing, the last unvisited vertex must connect to start.
    # We'll handle at depth n-1.
    
    def backtrack():
        if time.time() > deadline:
            return False
        if len(path) == n:
            last = path[-1]
            if edge_arrival(last, start) <= K:
                found[0] = path[:]
                return True
            return False
        cur = path[-1]
        remaining_unvisited = n - len(path)
        # Try neighbors
        for nb in adj_list[cur]:
            if visited[nb]:
                continue
            # If this is the last vertex we add (remaining_unvisited == 1),
            # nb must connect back to start
            if remaining_unvisited == 1:
                if edge_arrival(nb, start) > K:
                    continue
            visited[nb] = True
            path.append(nb)
            # Connectivity pruning
            if len(path) < n and not connected_component_ok(nb):
                path.pop()
                visited[nb] = False
                continue
            if backtrack():
                return True
            path.pop()
            visited[nb] = False
        return False
    
    if backtrack():
        return found[0]
    return None

def greedy_two_regular(deadline):
    """Greedy: add edges in arrival order, keep max degree 2, avoid premature small cycles.
       Returns a single cycle if formed, else None."""
    degree = [0] * n
    adj = [set() for _ in range(n)]
    parent = [-1] * n  # union-find
    size = [1] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]
        return True
    
    for idx in range(m):
        if time.time() > deadline:
            break
        u, v = edge_order[idx]
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        # Adding edge (u,v): if u,v in same component and component size < n, it forms a small cycle -> skip
        ru, rv = find(u), find(v)
        if ru == rv:
            if size[ru] < n:
                continue
            # if size == n, adding closes the big cycle
        degree[u] += 1
        degree[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        union(u, v)
        if all(d == 2 for d in degree):
            # Check single component
            if find(0) == find(1) and find(0) == find(n-1):
                # all same?
                ok = True
                root = find(0)
                for v in range(n):
                    if find(v) != root:
                        ok = False
                        break
                if ok:
                    # Extract cycle
                    cycle = [0]
                    prev = -1
                    cur = 0
                    for _ in range(n - 1):
                        nxts = list(adj[cur])
                        nxt = nxts[0] if nxts[0] != prev else nxts[1]
                        prev, cur = cur, nxt
                        cycle.append(cur)
                    return cycle
            return None
    return None

def two_opt_improve(cycle, K, deadline):
    """Try to reduce max arrival by 2-opt style swaps. Returns improved cycle or original."""
    cur = list(cycle)
    cur_score = cycle_score(cur)
    best = cur
    best_score = cur_score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Represent cycle edges
        def edges_of(c):
            e = set()
            L = len(c)
            for i in range(L):
                a, b = c[i], c[(i+1) % L]
                e.add((min(a,b), max(a,b)))
            return e
        cur_edges = edges_of(cur)
        L = len(cur)
        # Try removing two non-adjacent edges and reinserting to form a single cycle
        # Standard 2-opt on the cycle: pick two edges, reverse the segment between them.
        # We iterate over pairs of edge positions.
        found_better = False
        for i in range(L):
            if found_better or time.time() > deadline:
                break
            a1 = cur[i]
            b1 = cur[(i+1) % L]
            for j in range(L):
                if found_better or time.time() > deadline:
                    break
                if j <= (i+1) % L and not (j == (i-1) % L):
                    # skip adjacent / same
                    if j == i or j == (i+1) % L or j == (i-1) % L:
                        continue
                a2 = cur[j]
                b2 = cur[(j+1) % L]
                # New edges would be (a1,a2) & (b1,b2) or (a1,b2) & (b1,a2)
                # We need to ensure these new edges exist in the allowed set (arrival <= K) 
                # and produce a single cycle.
                for opt in ((a1, a2, b1, b2), (a1, b2, b1, a2)):
                    na1, nb1, na2, nb2 = opt
                    if na1 == nb1 or na2 == nb2:
                        continue
                    e1 = (min(na1, nb1), max(na1, nb1))
                    e2 = (min(na2, nb2), max(na2, nb2))
                    if edge_arrival(na1, nb1) > K or edge_arrival(na2, nb2) > K:
                        continue
                    # Build new cycle: the 2-opt reversal
                    # Remove edges (a1,b1) and (a2,b2); the two path segments.
                    # Reconstruct: from a1 go to a2, traverse to b1 in one direction,
                    # then b1 to b2, then b2 to a2... this is the standard reversal.
                    # Easier: build new edge set and check it's a single cycle.
                    new_edges = (cur_edges - {(min(a1,b1),max(a1,b1)), (min(a2,b2),max(a2,b2))})
                    new_edges.add(e1)
                    new_edges.add(e2)
                    # Check each vertex degree 2
                    deg = [0]*n
                    for (x,y) in new_edges:
                        deg[x] += 1
                        deg[y] += 1
                    if any(d != 2 for d in deg):
                        continue
                    # Check single cycle: BFS
                    adj2 = [set() for _ in range(n)]
                    for (x,y) in new_edges:
                        adj2[x].add(y)
                        adj2[y].add(x)
                    seen = [False]*n
                    dq = deque([0])
                    seen[0] = True
                    cnt = 0
                    while dq:
                        u = dq.popleft()
                        cnt += 1
                        for w in adj2[u]:
                            if not seen[w]:
                                seen[w] = True
                                dq.append(w)
                    if cnt != n:
                        continue
                    # Extract cycle
                    cyc = [0]
                    prev = -1
                    cur_v = 0
                    for _ in range(n-1):
                        nxts = list(adj2[cur_v])
                        nxt = nxts[0] if nxts[0] != prev else nxts[1]
                        prev, cur_v = cur_v, nxt
                        cyc.append(cur_v)
                    sc = cycle_score(cyc)
                    if sc < best_score:
                        best_score = sc
                        best = cyc
                        cur = cyc
                        cur_edges = edges_of(cur)
                        found_better = True
                        improved = True
                        break
                if found_better:
                    break
        # loop
    return best, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin under 90s
    
    default_cycle = list(range(n))
    best_cycle = default_cycle
    best_score = cycle_score(default_cycle)
    
    # Greedy 2-regular first (fast, gives a concrete upper bound)
    g = greedy_two_regular(deadline - 5.0)
    if g is not None:
        sc = cycle_score(g)
        if sc < best_score:
            best_score = sc
            best_cycle = g
    
    # Now try to find a Hamilton cycle in prefixes of increasing K.
    # Use binary search since feasibility is monotone in K.
    lo = 23  # minimum possible K
    hi = best_score - 1  # we want to find something with score < best_score, so K+1 < best_score => K < best_score-1
    found_at = None
    
    # Quick linear scan from low K to find the first feasible K (narrow the range)
    for K in range(lo, min(hi + 1, lo + 8)):
        if time.time() > deadline - 15.0:
            break
        if K + 1 >= best_score:
            break
        hc = find_hc_prefix(K, deadline - 10.0)
        if hc is not None:
            sc = cycle_score(hc)
            if sc < best_score:
                best_score = sc
                best_cycle = hc
                found_at = K
            break
    
    # Binary search for the minimum feasible K
    if found_at is None:
        lo2 = lo
        hi2 = hi
        while lo2 <= hi2 and time.time() < deadline - 10.0:
            mid = (lo2 + hi2) // 2
            if mid + 1 >= best_score:
                hi2 = mid - 1
                continue
            hc = find_hc_prefix(mid, deadline - 8.0)
            if hc is not None:
                sc = cycle_score(hc)
                if sc < best_score:
                    best_score = sc
                    best_cycle = hc
                    found_at = mid
                hi2 = mid - 1
            else:
                lo2 = mid + 1
    
    # If we found a cycle, try to improve with 2-opt at the current threshold
    if time.time() < deadline - 5.0:
        Kpolish = best_score - 1
        imp, imps = two_opt_improve(best_cycle, Kpolish, deadline - 2.0)
        if imps < best_score:
            best_score = imps
            best_cycle = imp
    
    return best_cycle
# EVOLVE-BLOCK-END
