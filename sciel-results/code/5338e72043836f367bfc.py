# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 150s limit, leave 40s margin for safety
    
    q = k
    
    def f_add(a, b): return (a + b) % q
    def f_sub(a, b): return (a - b) % q
    def f_mul(a, b): return (a * b) % q

    def build_luw(d, rules):
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
        """Fast girth check using non-backtracking BFS with early termination."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = min_girth - 1
        
        for start in range(n):
            # visited[(curr, prev)] = distance
            visited = set()
            queue = deque()
            for neighbor in adj[start]:
                state = (neighbor, start)
                if state not in visited:
                    visited.add(state)
                    queue.append((state, 1))
            
            while queue:
                (curr, prev), dist = queue.popleft()
                if dist >= max_depth:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if nxt == start:
                        if dist + 1 < min_girth:
                            return False
                        continue
                    new_state = (nxt, curr)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, dist + 1))
        return True

    def is_simple_graph(n, edges, expected_degree):
        """Check that graph is simple and k-regular."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v:
                return False
            if v in adj[u]:
                return False
            adj[u].add(v)
            adj[v].add(u)
        for v in range(n):
            if len(adj[v]) != expected_degree:
                return False
        return True

    # Generate candidate rule sets for d=4
    candidates = []
    # Rule indices: for c in 2..d, rule[c] = (a, b) where 1 <= a,b <= c-1
    # Try systematic combinations
    for r2_a in [1]:
        for r2_b in [1]:
            for r3_a in [1, 2]:
                for r3_b in [1, 2]:
                    if r3_a == r3_b:
                        continue
                    for r4_a in [1, 2, 3]:
                        for r4_b in [1, 2, 3]:
                            if r4_a == r4_b:
                                continue
                            rules = {2: (r2_a, r2_b), 3: (r3_a, r3_b), 4: (r4_a, r4_b)}
                            candidates.append(rules)

    best_result = None
    best_n = float('inf')
    
    for rules in candidates:
        if time.time() > deadline:
            break
        n, edges = build_luw(4, rules)
        if n < best_n:
            if is_simple_graph(n, edges, k):
                if check_girth_fast(n, edges, g):
                    best_result = (n, edges)
                    best_n = n
    
    if best_result is not None:
        return best_result

    # Try d=5 with several rule combinations
    d5_candidates = []
    for r4 in [(1, 2), (2, 1), (1, 3), (3, 1)]:
        for r5_a in [1, 2, 3, 4]:
            for r5_b in [1, 2, 3, 4]:
                if r5_a == r5_b:
                    continue
                rules = {2: (1, 1), 3: (2, 1), 4: r4, 5: (r5_a, r5_b)}
                d5_candidates.append(rules)

    for rules in d5_candidates:
        if time.time() > deadline:
            break
        n, edges = build_luw(5, rules)
        if n < best_n:
            if is_simple_graph(n, edges, k):
                if check_girth_fast(n, edges, g):
                    best_result = (n, edges)
                    best_n = n
    
    if best_result is not None:
        return best_result

    # Fallback: return a valid d=5 L-U-W graph (may not have girth g, but must be valid)
    rules = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
    n, edges = build_luw(5, rules)
    return n, edges
# EVOLVE-BLOCK-END
