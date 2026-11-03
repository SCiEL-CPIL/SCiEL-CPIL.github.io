# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    q = k

    # Precompute finite field tables for speed
    if q == 4:
        add_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return mul_t[a][b]
    else:
        add_t = [[(a+b) % q for b in range(q)] for a in range(q)]
        mul_t = [[(a*b) % q for b in range(q)] for a in range(q)]
        def f_add(a, b): return add_t[a][b]
        def f_sub(a, b): return add_t[a][q-1-b if b != 0 else 0]  # a - b mod q
        def f_mul(a, b): return mul_t[a][b]

    def build_luw(d, rules):
        """Build LUW graph with given depth d and rules. Returns (n, edges)."""
        start_key = (0, (0,) * (d + 1))
        label = {start_key: 0}
        queue = [start_key]
        edge_list = []

        for vkey in queue:
            side, coords = vkey
            vidx = label[vkey]
            for first_val in range(q):
                ncoords = [0] * (d + 1)
                ncoords[1] = first_val
                for c in range(2, d + 1):
                    a, b = rules[c]
                    if side == 0:
                        ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                    else:
                        ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
                nkey = (1 - side, tuple(ncoords))
                if nkey not in label:
                    label[nkey] = len(queue)
                    queue.append(nkey)
                if side == 0:
                    edge_list.append((vidx, label[nkey]))

        return len(queue), edge_list

    def check_girth_fast(n, edges, min_girth):
        """Check girth using non-backtracking BFS with early termination."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Check each vertex for short cycles via non-backtracking walks
        for start in range(n):
            # State: (prev, curr) -> distance from start
            dist = {}
            q_bfs = deque()
            for nb in adj[start]:
                q_bfs.append((start, nb, 1))
                dist[(start, nb)] = 1

            while q_bfs:
                u, v, d_cur = q_bfs.popleft()
                if d_cur >= min_girth - 1:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue  # non-backtracking
                    if w == start:
                        if d_cur + 1 < min_girth:
                            return False
                        else:
                            continue  # cycle of exactly min_girth is OK
                    state = (v, w)
                    if state not in dist:
                        dist[state] = d_cur + 1
                        q_bfs.append((v, w, d_cur + 1))
        return True

    def moore_bound(k, g):
        if k <= 2:
            return g if g % 2 == 0 else 2 * g - 1
        if g % 2 == 0:
            r = g // 2
            return 2 * k * sum((k - 1) ** i for i in range(r))
        else:
            r = (g - 1) // 2
            return 1 + k * sum((k - 1) ** i for i in range(r + 1))

    mb = moore_bound(k, g)

    best_result = None
    best_n = float('inf')

    # Strategy: try different d values and rule sets
    # For LUW, girth is at least d+4 (even d) or d+5 (odd d)
    # We need girth >= g, so try d values that should give sufficient girth
    
    # Generate candidate (d, rules) combinations
    candidates = []
    
    # d=3: girth >= 7 (odd) - may not be enough for g=9
    # d=4: girth >= 8 (even) - may not be enough for g=9
    # d=5: girth >= 10 (odd) - should be enough for g=9
    # d=6: girth >= 10 (even) - should be enough for g=9
    
    for d in [3, 4, 5, 6, 7]:
        if d < 3:
            continue
        # Base rules for first few positions
        base_rules = {}
        if d >= 2:
            base_rules[2] = (1, 1)
        if d >= 3:
            base_rules[3] = (2, 1)
        if d >= 4:
            base_rules[4] = (1, 2)
        
        # Generate rule variations for positions 4..d
        if d >= 4:
            # Try different rule for position 4
            r4_options = [(1, 2), (2, 1), (1, 1), (2, 2), (3, 1), (1, 3), (2, 3)]
        else:
            r4_options = [None]
        
        for r4 in r4_options:
            if time.time() > deadline - 10:
                break
            rules = base_rules.copy()
            if d >= 4 and r4 is not None:
                rules[4] = r4
            
            # For d >= 5, try different rule for position 5
            if d >= 5:
                r5_options = [(1, 3), (3, 1), (2, 2), (1, 4), (4, 1), (2, 3), (3, 2)]
                for r5 in r5_options:
                    if time.time() > deadline - 5:
                        break
                    rules_test = rules.copy()
                    rules_test[5] = r5
                    
                    # For d >= 6, try different rule for position 6
                    if d >= 6:
                        r6_options = [(2, 4), (4, 2), (1, 5), (5, 1), (3, 5), (5, 3)]
                        for r6 in r6_options:
                            if time.time() > deadline - 3:
                                break
                            rules_test6 = rules_test.copy()
                            rules_test6[6] = r6
                            candidates.append((d, rules_test6))
                    else:
                        candidates.append((d, rules_test))
            else:
                candidates.append((d, rules))

    # Sort candidates by expected graph size (smaller d first, then by rule diversity)
    # Prefer smaller d values as they produce smaller graphs
    candidates.sort(key=lambda x: x[0])

    for d, rules in candidates:
        if time.time() > deadline - 5:
            break
        
        n, edges = build_luw(d, rules)
        
        # Quick check: if n is already too large, skip girth check
        if n >= best_n:
            continue
        
        if check_girth_fast(n, edges, g):
            best_n = n
            best_result = (n, edges)
            if n == mb:
                return best_result

    # If no valid result found from candidates, try a fallback with larger d
    if best_result is None:
        d = 7
        rules = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3), 6: (2, 4), 7: (1, 5)}
        n, edges = build_luw(d, rules)
        return n, edges

    return best_result
# EVOLVE-BLOCK-END
