"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy:
    1. Try LUW construction with a parameter sweep to find the smallest d with girth >= g.
    2. If LUW fails or is too slow, fall back to a random Cayley graph approach on SL(2, q)
       with optimized girth verification.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    if k != 3:
        # Fallback for other k values (not the main task)
        # For now, just use a simple LUW-like approach or return empty
        pass

    def check_girth_fast(adj, n_nodes, target_girth):
        """
        Optimized non-backtracking girth check.
        Uses BFS with parent tracking and early termination.
        Returns True if girth >= target_girth.
        """
        max_depth = (target_girth - 1) // 2
        for s in range(n_nodes):
            dist = [-1] * n_nodes
            parent = [-1] * n_nodes
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    # --- LUW Construction ---
    def build_luw(d, q_val):
        """Build LUW graph CD(d, q) for prime q. Returns (n, edges)."""
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q_val
            sub_fn = lambda x, y: (x - y) % q_val
            mul_fn = lambda x, y: x * y % q_val

        # Rule: for each c in 2..d, rule[c] = (a, b)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # y[c] = x[c] + y[a] * x[b]
                        y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                    else:
                        # y[c] = x[c] - x[a] * y[b]
                        y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
                yield 1 - side, tuple(y)

        # BFS from zero point to get connected component
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try LUW first with a sweep
    best_luw = None
    q_val = 3
    # Start from small d and increase. Smaller d -> fewer vertices.
    # We need girth >= g.
    for d in range(5, 15):
        if time.time() > deadline - 30:
            break
        n_luw, edges_luw = build_luw(d, q_val)
        if n_luw < 2:
            continue
        # Quick degree check
        deg = [0] * n_luw
        for u, v in edges_luw:
            deg[u] += 1
            deg[v] += 1
        if any(dd != k for dd in deg):
            continue
        
        # Build adjacency list for girth check
        adj = [[] for _ in range(n_luw)]
        for u, v in edges_luw:
            adj[u].append(v)
            adj[v].append(u)
            
        if check_girth_fast(adj, n_luw, g):
            # Valid graph found
            return n_luw, edges_luw
        else:
            # Keep track of the largest one we tried (in case we need a fallback, though invalid)
            # Actually, if it's invalid, we can't return it.
            # But we want the smallest valid one.
            # Since d increases, n increases. So the first valid d is the smallest n.
            pass

    # If LUW didn't find a valid graph within the time/sweep limit,
    # or if we want to try Cayley graphs (which might have fewer vertices for some parameters?),
    # let's try the Cayley graph approach on SL(2, 16) or similar.
    # However, SL(2, 16) has 4080 vertices. LUW with d=7 has 2*3^7 = 4374.
    # LUW with d=6 has 2*3^6 = 1458.
    # Let's see if LUW d=6 works for g=16.
    # Girth of CD(d,3) is roughly 2*floor(d/2)+3 or similar.
    # For d=6, girth is likely around 10-12. Not enough for 16.
    # For d=7, girth is likely around 12-14.
    # For d=8, girth is likely around 14-16.
    # So we might need d=8 or 9.
    # d=8 -> 2*6561 = 13122 vertices.
    # d=9 -> 2*19683 = 39366 vertices.
    
    # The inspiration suggests trying Cayley graphs on SL(2, q).
    # Let's implement a quick random search on SL(2, 16) if we have time.
    
    if time.time() < deadline - 20:
        try:
            # SL(2, 16) Cayley Graph Search
            m = 4
            q = 1 << m
            poly = 0b10011  # x^4 + x + 1
            
            # Precompute multiplication table for GF(16)
            mul_table = [[0] * q for _ in range(q)]
            for i in range(q):
                for j in range(q):
                    res = 0
                    a = i
                    b = j
                    while b:
                        if b & 1:
                            res ^= a
                        a <<= 1
                        if a >= q:
                            a ^= poly
                        b >>= 1
                    mul_table[i][j] = res
            
            def gf_add(a, b):
                return a ^ b

            def gf_mul(a, b):
                return mul_table[a][b]

            def gf_inv(a):
                if a == 0:
                    return 0
                for i in range(1, q):
                    if gf_mul(a, i) == 1:
                        return i
                return 0

            def gf_div(a, b):
                if b == 0:
                    return 0
                return gf_mul(a, gf_inv(b))

            # Generate SL(2, 16) elements
            elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a == 0:
                            if gf_mul(b, c) == 1:
                                for d in range(q):
                                    elements.append((a, b, c, d))
                        else:
                            bc = gf_mul(b, c)
                            rhs = gf_add(1, bc)
                            d = gf_div(rhs, a)
                            elements.append((a, b, c, d))
            
            n_cayley = len(elements)
            elem_to_idx = {e: i for i, e in enumerate(elements)}
            
            # Find Involutions
            involutions = []
            for e in elements:
                a, b, c, d = e
                if gf_add(a, d) == 0 and not (a == 1 and b == 0 and c == 0 and d == 1):
                    involutions.append(e)
            
            def multiply(m1, m2):
                a1, b1, c1, d1 = m1
                a2, b2, c2, d2 = m2
                aa = gf_add(gf_mul(a1, a2), gf_mul(b1, c2))
                ab = gf_add(gf_mul(a1, b2), gf_mul(b1, d2))
                ba = gf_add(gf_mul(c1, a2), gf_mul(d1, c2))
                bb = gf_add(gf_mul(c1, b2), gf_mul(d1, d2))
                return (aa, ab, ba, bb)

            import random
            random.seed(42)
            
            # Try a few random sets of 3 involutions
            max_trials = 10
            for trial in range(max_trials):
                if time.time() > deadline - 5:
                    break
                
                S = random.sample(involutions, 3)
                
                # Build adjacency
                adj = [[] for _ in range(n_cayley)]
                for i, e in enumerate(elements):
                    for s in S:
                        f = multiply(e, s)
                        j = elem_to_idx[f]
                        adj[i].append(j)
                
                # Quick connectivity check
                visited = [False] * n_cayley
                queue = deque([0])
                visited[0] = True
                count = 1
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            count += 1
                            queue.append(v)
                
                if count != n_cayley:
                    continue
                    
                # Check girth
                if check_girth_fast(adj, n_cayley, g):
                    edges = []
                    for i in range(n_cayley):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n_cayley, edges

        except Exception:
            pass

    # Final fallback: return the largest LUW graph we built, even if invalid?
    # No, the task requires validity.
    # If we haven't returned, we must return something.
    # Let's try a larger d in LUW if we have time, or just return a known valid one if we found one earlier.
    # Actually, the loop above returns immediately if valid.
    # If we get here, it means no LUW d in 5..14 was valid (unlikely for d=14) or we ran out of time.
    
    # As a last resort, try d=10..14 for LUW if we have time
    for d in range(10, 15):
        if time.time() > deadline - 2:
            break
        n_luw, edges_luw = build_luw(d, q_val)
        adj = [[] for _ in range(n_luw)]
        for u, v in edges_luw:
            adj[u].append(v)
            adj[v].append(u)
        if check_girth_fast(adj, n_luw, g):
            return n_luw, edges_luw

    # If absolutely nothing works, return an empty graph (will score 0)
    # But we should try to return *something* valid.
    # Let's assume d=10 or 11 in LUW works for g=16.
    # If we didn't catch it, something went wrong with the time.
    
    return 0, []
# EVOLVE-BLOCK-END
