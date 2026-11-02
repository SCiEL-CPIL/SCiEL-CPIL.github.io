import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Calculate Moore Bound
    # n >= k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # For odd g
    moore_bound = 0
    term = 1
    for i in range((g - 1) // 2):
        moore_bound += term
        term *= (k - 1)
    moore_bound *= k
    
    # For k=5, g=9, moore_bound = 425
    # n must be even for k=5 circulant
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Search for Circulant Graph
    # k=5 => S = {n/2, s1, -s1, s2, -s2}
    # n must be even
    
    max_n_search = 3000
    if k != 5 or g != 9:
        # For other parameters, just return LUW or simple construction
        # But task specifies k=5, g=9
        pass
        
    # Optimized Girth Check for Circulant Graph
    # Uses non-backtracking BFS: we track (vertex, previous_vertex) to avoid
    # counting the tree edge as a cycle. A cycle of length L is found when
    # we reach a state (v, u) where v was already visited at depth d_v with
    # a different "parent", giving cycle length d_u + d_v + 1.
    def check_girth_circulant(n, steps, girth_limit):
        # For circulant graphs, steps come in pairs: [n//2, s1, -s1, s2, -s2]
        # We need to check: no closed walk of length < girth_limit
        # Using BFS on the directed line graph (states are directed edges)
        # State: (u, v) means we're at u, came from v (non-backtracking)
        # Start: from vertex 0, go to each neighbor
        # If we ever reach (v, u) where (v, u') was already seen with u' != u
        # and depth[v] + depth[u] + 1 < girth_limit, we have a short cycle.
        
        # More efficient: standard BFS with parent tracking
        # dist[v] = shortest distance from 0 to v
        # parent[v] = the vertex we came from to reach v
        # When exploring u -> v:
        #   if v == parent[u]: skip (backtracking)
        #   elif dist[v] == -1: dist[v] = dist[u]+1, parent[v] = u
        #   else: cycle of length dist[u] + dist[v] + 1
        #         BUT only if v != parent[u] AND u != parent[v]
        #         Actually if dist[v] <= dist[u] and v is not parent of u,
        #         we found a cycle.
        
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        queue = [0]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[u]
            
            # Pruning: if du is large enough that any new cycle would be >= girth_limit
            # A cycle found now has length >= du + dist[v] + 1 >= du + 0 + 1 = du+1
            # But we need dist[v] >= 1 typically. More carefully:
            # If du >= girth_limit, any cycle through u has length >= 2*du+1 >= girth_limit+1
            # So we can stop expanding.
            if du >= (girth_limit - 1) // 2 + 1:
                # Actually let's be safe: if du + 1 >= girth_limit, skip expansion
                # because any cycle through u has length at least du + 1 + 1 = du+2 (if v is at dist 1)
                # Hmm, let me think more carefully.
                # If we're at u with dist[u] = du, and we go to v with dist[v] = dv,
                # the cycle length is du + dv + 1.
                # The minimum dv we can encounter (excluding parent) is 0 (if v=0) or 1.
                # So min cycle length from u is du + 0 + 1 = du+1 (if v=0 and 0 is not parent of u)
                # or du + 1 + 1 = du+2 (if v is at distance 1).
                # To be safe, if du + 1 >= girth_limit, no cycle found from u can be < girth_limit.
                if du + 1 >= girth_limit:
                    continue
            
            for s in steps:
                v = (u + s) % n
                # Skip backtracking
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a potential cycle
                    # Only count if this isn't just the reverse tree edge
                    # i.e., u is not the parent of v (which would mean dist[u] = dist[v]+1)
                    # Actually if dist[v] == du + 1, then v is a child of u in the BFS tree,
                    # and the edge u-v is a tree edge (or parallel), not a cycle.
                    # If dist[v] <= du, we have a non-tree edge creating a cycle.
                    if dist[v] <= du:
                        # Also need to check it's not the parent edge in reverse
                        # If v == parent[u], we already skipped.
                        # If u == parent[v], then dist[u] = dist[v] - 1, so dist[v] = dist[u] + 1 > du,
                        # which contradicts dist[v] <= du. So we're fine.
                        cycle_len = du + dist[v] + 1
                        if cycle_len < girth_limit:
                            return False
        
        return True

    # Try Circulants
    # To save time, we sample n and steps
    # We prioritize smaller n
    
    # Deterministic set of "good" steps first?
    # Random is fine.
    
    # Limit attempts per n
    attempts_per_n = 100
    
    for n in range(moore_bound, max_n_search, 2):
        if time.time() > deadline:
            break
            
        # Try a few specific structured steps first?
        # e.g. s1 = n//5, s2 = n//3
        candidates = []
        # Structured candidates - diverse ratios
        structured_pairs = [
            (n//5, n//3), (n//4, n//3), (n//6, n//4),
            (n//5, n//4), (n//7, n//3), (n//6, n//5),
            (n//8, n//3), (n//5, n//7), (n//4, n//5),
            (n//6, n//3), (n//7, n//5), (n//8, n//5),
        ]
        for s1, s2 in structured_pairs:
            if 0 < s1 < n//2 and 0 < s2 < n//2 and s1 != s2 and s1 + s2 != n:
                candidates.append((s1, s2))
        
        # Random
        for _ in range(attempts_per_n):
            s1 = random.randint(1, n//2 - 1)
            s2 = random.randint(1, n//2 - 1)
            if s1 != s2 and s1 + s2 != n:
                candidates.append((s1, s2))
                
        for s1, s2 in candidates:
            if time.time() > deadline:
                break
            if s1 == s2: continue
            if s1 == 0 or s2 == 0: continue
            if s1 == n//2 or s2 == n//2: continue
            
            steps = [n//2, s1, -s1, s2, -s2]
            # Ensure no duplicates in steps (mod n)
            if s1 + s2 == n: continue
            if s1 == s2: continue
            # Also check s1 != -s2 mod n is same as s1+s2 != n (done)
            # Check s1 != 0, s2 != 0, s1 != n//2, s2 != n//2 (done by range)
            
            if check_girth_circulant(n, steps, g):
                edges = []
                for i in range(n):
                    for s in steps:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Fallback: LUW Graph
    # Re-implement LUW for k=5, g=9
    # q = 5, d = 4
    return construct_luw(k, g)

def construct_luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9, d=4
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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
