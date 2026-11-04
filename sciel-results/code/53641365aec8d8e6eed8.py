import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # --- Fast girth verification via non-backtracking BFS ---
    def _check_girth(adj, n, g_limit):
        """Check that no cycle shorter than g_limit exists.
        Uses BFS from each vertex up to depth (g_limit-1)//2.
        If two neighbors at depth d-1 share a child, or we find a cross-edge
        within the tree, we detect a short cycle."""
        max_depth = (g_limit - 1) // 2
        for start in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a non-tree edge; compute cycle length
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g_limit:
                            return False
        return True

    # --- Lazebnik-Ustimenko-Woldar graph (seed baseline) ---
    def _luw_graph(k, g):
        q = k
        # For girth g, we need d such that the graph has girth at least d+5 (odd d)
        # or d+4 (even d). The LUW construction gives girth >= d+5 for odd d.
        # For g=16, we want girth >= 16, so d+5 >= 16 => d >= 11 (odd).
        # But the key insight: fixing d=7 for g=16 gives a smaller graph.
        # Actually, the LUW graph CD(d,q) has girth at least d+5 for odd d.
        # So for g=16, d=11 gives girth >= 16. d=9 gives girth >= 14 (too small).
        # Wait - the finding says d=7 gives ~4k vertices. Let me reconsider.
        # The LUW graph has girth at least d+5 for odd d. So d=11 -> girth >= 16.
        # But maybe the actual girth is larger than the lower bound, so d=7 (girth >= 12)
        # might still have girth >= 16 in practice? No, that's not guaranteed.
        # Let me use d=11 for correctness, but also try d=7 and verify.
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # --- LPS Graph Construction on PSL(2, q) ---
    def _lps_graph(q):
        """Construct a 3-regular graph from PSL(2, q) using standard generators."""
        if q < 5:
            return None, None
        # Generate all 2x2 matrices with det=1 mod q, identifying M ~ -M
        mat_to_idx = {}
        mats = []
        idx = 0
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # Need ad - bc = 1 mod q
                    # d = (1 + bc) * a^{-1} mod q, requires a != 0
                    if a == 0:
                        # Then -bc = 1 mod q, need b != 0
                        if b == 0:
                            continue
                        try:
                            b_inv = pow(b, -1, q)
                        except ValueError:
                            continue
                        d = (-1 * b_inv) % q
                        m = (a, b, c, d)
                        key = frozenset([m, ((q-a)%q, (q-b)%q, (q-c)%q, (q-d)%q)])
                        if key not in mat_to_idx:
                            mat_to_idx[key] = idx
                            mats.append(m)
                            idx += 1
                    else:
                        try:
                            a_inv = pow(a, -1, q)
                        except ValueError:
                            continue
                        d = ((1 + b * c) * a_inv) % q
                        m = (a, b, c, d)
                        key = frozenset([m, ((q-a)%q, (q-b)%q, (q-c)%q, (q-d)%q)])
                        if key not in mat_to_idx:
                            mat_to_idx[key] = idx
                            mats.append(m)
                            idx += 1

        n = len(mats)
        
        # Generators for PSL(2,q):
        # S = [[0, -1], [1, 0]], T = [[1, 1], [0, 1]]
        # S^2 = -I ~ I in PSL, so S has order 2 in PSL
        # T has order q
        # Use S and T, T^{-1} as generators for 3-regular
        S = (0, q-1, 1, 0)  # [[0,-1],[1,0]]
        T = (1, 1, 0, 1)    # [[1,1],[0,1]]
        T_inv = (1, q-1, 0, 1)  # [[1,-1],[0,1]]
        
        gens = [S, T, T_inv]
        
        def mat_mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            return ((a1*a2 + b1*c2) % q, (a1*b2 + b1*d2) % q,
                    (c1*a2 + d1*c2) % q, (c1*b2 + d1*d2) % q)
        
        adj = [[] for _ in range(n)]
        edges = []
        edge_set = set()
        
        for i, m in enumerate(mats):
            for g in gens:
                nm = mat_mul(m, g)
                key = frozenset([nm, ((q-nm[0])%q, (q-nm[1])%q, (q-nm[2])%q, (q-nm[3])%q)])
                if key in mat_to_idx:
                    j = mat_to_idx[key]
                    if i != j:
                        if (i, j) not in edge_set and (j, i) not in edge_set:
                            adj[i].append(j)
                            adj[j].append(i)
                            edge_set.add((min(i,j), max(i,j)))
                            edges.append((min(i,j), max(i,j)))
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None, None
        
        return n, edges

    # --- Voltage graph / covering construction ---
    def _voltage_graph(base_n, voltages, group_order):
        """Construct a covering graph using voltage assignment.
        base_n: number of vertices in base
        voltages: dict mapping (u,v) -> voltage in Z/group_order
        group_order: size of the cyclic group
        """
        n = base_n * group_order
        adj = [[] for _ in range(n)]
        edges = []
        edge_set = set()
        
        for u in range(base_n):
            for v, vol in voltages.items():
                if u not in v:
                    continue
                # Edge from (u, 0) to (v, vol) and from (u, 1) to (v, vol+1) etc.
                for offset in range(group_order):
                    u_full = u * group_order + offset
                    v_full = v * group_order + (offset + vol) % group_order
                    if u_full != v_full:
                        if (u_full, v_full) not in edge_set and (v_full, u_full) not in edge_set:
                            adj[u_full].append(v_full)
                            adj[v_full].append(u_full)
                            edge_set.add((min(u_full, v_full), max(u_full, v_full)))
                            edges.append((min(u_full, v_full), max(u_full, v_full)))
        
        for i in range(n):
            if len(adj[i]) != 3:
                return None, None
        return n, edges

    # --- Main logic ---
    if k == 3 and g == 16:
        best_n = float('inf')
        best_edges = None
        
        # Strategy 1: Try LPS graphs for various primes
        # PSL(2, q) has q(q^2-1)/2 elements
        # q=7: 7*48/2 = 168
        # q=11: 11*120/2 = 660
        # q=13: 13*168/2 = 1092
        # q=17: 17*288/2 = 2448
        # q=19: 19*360/2 = 3420
        for q in [7, 11, 13, 17, 19, 23]:
            if time.time() > deadline - 30:
                break
            n_lps, edges_lps = _lps_graph(q)
            if n_lps is not None and n_lps < best_n:
                adj = [[] for _ in range(n_lps)]
                for u, v in edges_lps:
                    adj[u].append(v)
                    adj[v].append(u)
                if _check_girth(adj, n_lps, g):
                    best_n = n_lps
                    best_edges = edges_lps
                    break  # Found a valid one, use it
        
        # Strategy 2: LUW graph with d=11 (guaranteed girth >= 16)
        if time.time() < deadline - 20:
            n_luw, edges_luw = _luw_graph(k, g)
            if n_luw < best_n:
                adj = [[] for _ in range(n_luw)]
                for u, v in edges_luw:
                    adj[u].append(v)
                    adj[v].append(u)
                if _check_girth(adj, n_luw, g):
                    best_n = n_luw
                    best_edges = edges_luw
        
        # Strategy 3: Try smaller LUW with d=7 and verify girth
        # CD(7, 3) might have girth >= 16 even though the bound says >= 12
        if time.time() < deadline - 15:
            q = k
            d = 7
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
            else:
                add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)

            start = (0, (None,) + (0,) * d)
            index, order, edges = {start: 0}, [start], []
            for vertex in order:
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            
            n_small = len(order)
            if n_small < best_n:
                adj = [[] for _ in range(n_small)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                if _check_girth(adj, n_small, g):
                    best_n = n_small
                    best_edges = edges
        
        # Strategy 4: Try circulant graphs C(n, {a, b, n/2}) for even n
        if time.time() < deadline - 10 and best_n > 512:
            moore_bound = 512
            for n in range(moore_bound, min(best_n, 1200) + 1, 2):
                if time.time() > deadline - 10:
                    break
                if n % 2 != 0:
                    continue
                a_candidates = list(range(2, n // 4 + 1))
                tried = set()
                for a in a_candidates:
                    if time.time() > deadline - 10:
                        break
                    na = (n - a) % n
                    if na == 0 or na == n // 2 or na == a:
                        continue
                    conn = frozenset([a, na, n // 2])
                    if conn in tried:
                        continue
                    tried.add(conn)
                    
                    adj = [[] for _ in range(n)]
                    edges = []
                    edge_set = set()
                    valid = True
                    for i in range(n):
                        for c in conn:
                            j = (i + c) % n
                            if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
                                adj[i].append(j)
                                adj[j].append(i)
                                edge_set.add((min(i, j), max(i, j)))
                                edges.append((min(i, j), max(i, j)))
                    for i in range(n):
                        if len(adj[i]) != 3:
                            valid = False
                            break
                    if not valid:
                        continue
                    if _check_girth(adj, n, g):
                        best_n = n
                        best_edges = edges
                        break
        
        # Return best found
        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback: LUW
        n_luw, edges_luw = _luw_graph(k, g)
        return n_luw, edges_luw
    
    # General fallback
    n_luw, edges_luw = _luw_graph(k, g)
    return n_luw, edges_luw
