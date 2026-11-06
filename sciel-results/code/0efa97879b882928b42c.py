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
    
    # Fast connectivity check using bitmasks for the unvisited set
    unvisited_mask = ((1 << n) - 1) ^ (1 << start)
    
    def connected_component_ok(cur):
        # BFS from cur over unvisited + cur
        seen = 1 << cur
        dq = deque([cur])
        while dq:
            u = dq.popleft()
            for w in adj_list[u]:
                bit = 1 << w
                if (unvisited_mask & bit) and not (seen & bit):
                    seen |= bit
                    dq.append(w)
        # All unvisited must be reached
        return (unvisited_mask | (1 << cur)) == seen
    
    def backtrack():
        nonlocal unvisited_mask
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
        # Warnsdorff: sort unvisited neighbors by fewest onward moves
        candidates = []
        for nb in adj_list[cur]:
            if visited[nb]:
                continue
            if remaining_unvisited == 1:
                if edge_arrival(nb, start) > K:
                    continue
            # Count onward moves for nb among unvisited
            onward = 0
            for w in adj_list[nb]:
                if not visited[w]:
                    onward += 1
            if remaining_unvisited == 1:
                # Must be able to close to start
                if onward == 0 and edge_arrival(nb, start) > K:
                    continue
            candidates.append((onward, nb))
        candidates.sort()
        for _, nb in candidates:
            visited[nb] = True
            path.append(nb)
            unvisited_mask &= ~(1 << nb)
            # Connectivity pruning
            if len(path) < n and not connected_component_ok(nb):
                path.pop()
                visited[nb] = False
                unvisited_mask |= (1 << nb)
                continue
            if backtrack():
                return True
            path.pop()
            visited[nb] = False
            unvisited_mask |= (1 << nb)
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
        L = len(cur)
        # Precompute current edges as a set
        cur_edges = set()
        for i in range(L):
            a, b = cur[i], cur[(i+1) % L]
            cur_edges.add((min(a,b), max(a,b)))
        # Build adjacency for current cycle
        cur_adj = [set() for _ in range(n)]
        for (x,y) in cur_edges:
            cur_adj[x].add(y)
            cur_adj[y].add(x)
        found_better = False
        for i in range(L):
            if found_better or time.time() > deadline:
                break
            a1 = cur[i]
            b1 = cur[(i+1) % L]
            e1_old = (min(a1,b1), max(a1,b1))
            for j in range(i+2, L):
                if found_better or time.time() > deadline:
                    break
                if j == 0 and i == L-1:
                    continue
                if j == (i+1) % L or j == (i-1) % L:
                    continue
                a2 = cur[j]
                b2 = cur[(j+1) % L]
                e2_old = (min(a2,b2), max(a2,b2))
                # 2-opt: reverse segment between i and j
                # New edges: (a1, a2) and (b1, b2)
                # Check arrivals
                if edge_arrival(a1, a2) > K or edge_arrival(b1, b2) > K:
                    continue
                # Build new cycle by reversing
                new_cur = cur[:i+1] + cur[j+1:i+1:-1] + cur[j+1:]
                # Wait, that's not right. Let me think again.
                # Standard 2-opt: remove edges (a1,b1) and (a2,b2), add (a1,a2) and (b1,b2)
                # The new cycle: from a1, go to a2, then follow old cycle from a2 to b2 (reversed), then b1, then old cycle from b1 to a1
                # Actually simpler: just build the new edge set and extract
                new_edges = cur_edges - {e1_old, e2_old}
                ne1 = (min(a1,a2), max(a1,a2))
                ne2 = (min(b1,b2), max(b1,b2))
                new_edges.add(ne1)
                new_edges.add(ne2)
                # Quick check: all degree 2
                deg = [0]*n
                for (x,y) in new_edges:
                    deg[x] += 1
                    deg[y] += 1
                if any(d != 2 for d in deg):
                    continue
                # Check single cycle
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
                prev_v = -1
                cur_v = 0
                for _ in range(n-1):
                    nxts = list(adj2[cur_v])
                    nxt = nxts[0] if nxts[0] != prev_v else nxts[1]
                    prev_v, cur_v = cur_v, nxt
                    cyc.append(cur_v)
                sc = cycle_score(cyc)
                if sc < best_score:
                    best_score = sc
                    best = cyc
                    cur = cyc
                    cur_edges = new_edges
                    cur_adj = adj2
                    found_better = True
                    improved = True
                    break
            if found_better:
                break
    return best, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 87.0  # safety margin under 90s
    
    default_cycle = list(range(n))
    best_cycle = default_cycle
    best_score = cycle_score(default_cycle)
    
    # Greedy 2-regular first (fast, gives a concrete upper bound)
    g = greedy_two_regular(t0 + 3.0)
    if g is not None:
        sc = cycle_score(g)
        if sc < best_score:
            best_score = sc
            best_cycle = g
    
    # Now try to find a Hamilton cycle in prefixes of increasing K.
    start_K = 23
    found_at = None
    for K in range(start_K, m):
        if time.time() > deadline - 5.0:
            break
        if K + 1 >= best_score:
            break
        # Give more time as K increases (more likely to succeed)
        per_k_deadline = min(deadline - 5.0, time.time() + 2.0)
        hc = find_hc_prefix(K, per_k_deadline)
        if hc is not None:
            sc = cycle_score(hc)
            if sc < best_score:
                best_score = sc
                best_cycle = hc
                found_at = K
                # Try to improve with 2-opt at this K
                if time.time() < deadline - 3.0:
                    imp, imps = two_opt_improve(best_cycle, K, deadline - 3.0)
                    if imps < best_score:
                        best_score = imps
                        best_cycle = imp
                break
    # If we have a decent cycle, do a final 2-opt polish at the current best_score-1 threshold
    if time.time() < deadline - 2.0:
        Kpolish = best_score - 1
        imp, imps = two_opt_improve(best_cycle, Kpolish, deadline - 1.0)
        if imps < best_score:
            best_score = imps
            best_cycle = imp
    return best_cycle
# EVOLVE-BLOCK-END
