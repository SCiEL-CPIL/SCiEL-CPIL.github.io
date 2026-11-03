import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit
    
    # Try a few primes q
    for q in [11, 13, 17, 19, 23]:
        if time.time() > deadline - 10:
            break
            
        # Generate SL(2, q) elements efficiently
        elems = []
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    for d in range(q):
                        if (a*d - b*c) % q == 1:
                            elems.append((a, b, c, d))
        
        n_elems = len(elems)
        # Map element to index
        elem_to_idx = {e: i for i, e in enumerate(elems)}
        
        def mul(e1, e2):
            a1, b1, c1, d1 = e1
            a2, b2, c2, d2 = e2
            return (
                (a1*a2 + b1*c2) % q,
                (a1*b2 + b1*d2) % q,
                (c1*a2 + d1*c2) % q,
                (c1*b2 + d1*d2) % q
            )
        
        def inv(e):
            a, b, c, d = e
            return (d, (-b) % q, (-c) % q, a)
        
        identity = (1, 0, 0, 1)
        
        # We need a symmetric set S of size k
        # k is odd, so 1 self-inverse, (k-1)/2 pairs
        # Find self-inverse elements (order 2)
        self_inv = [e for e in elems if mul(e, e) == identity and e != identity]
        
        # Precompute inverse index for each element
        inv_idx = [elem_to_idx[inv(e)] for e in elems]
        
        # Random search for S
        attempts = 0
        while time.time() < deadline - 5 and attempts < 5000:
            attempts += 1
            if not self_inv:
                break
            s0 = random.choice(self_inv)
            # Pick (k-1)/2 pairs
            available = [e for e in elems if e != identity and e != s0 and inv(e) != e]
            # Remove inverses to avoid picking both
            # Just pick random, check if inverse is already picked
            picked = set()
            current_set = [s0]
            for _ in range((k-1)//2):
                if not available:
                    break
                e = random.choice(available)
                e_inv = inv(e)
                if e not in picked and e_inv not in picked:
                    current_set.append(e)
                    current_set.append(e_inv)
                    picked.add(e)
                    picked.add(e_inv)
            
            if len(current_set) != k:
                continue
            
            # Check if S generates the group
            # BFS from identity using S
            visited = {identity}
            queue = deque([identity])
            while queue:
                u = queue.popleft()
                for s in current_set:
                    v = mul(u, s)
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            
            if len(visited) != n_elems:
                continue
            
            # Build adjacency list
            adj = [[] for _ in range(n_elems)]
            for i, u in enumerate(elems):
                for s in current_set:
                    v = mul(u, s)
                    j = elem_to_idx[v]
                    adj[i].append(j)
            
            # Girth check using non-backtracking walks from every vertex
            # A cycle of length L < g exists iff there is a non-backtracking walk
            # of length L that returns to its start
            girth_ok = True
            max_depth = g - 1  # check walks up to length g-1
            for u in range(n_elems):
                # State: (current_node, previous_node, length)
                # We track all non-backtracking walks of length up to max_depth
                # If we find a walk that returns to u with length < g, girth is violated
                # Use BFS on the directed line graph (non-backtracking states)
                # State is (node, prev) where prev is the node we came from
                # Start states: (v, u) for each neighbor v of u, length 1
                # We want to detect if we can return to (u, v') with total length L < g
                # where v' is a neighbor of u, and the walk is non-backtracking
                
                # More efficient: BFS from each directed edge (u, v)
                # and check if we can reach (u, w) for some neighbor w != v
                # with path length < g - 1 (since the first step is already taken)
                
                # Actually, let's use the standard non-backtracking walk detection:
                # For each starting vertex u, do BFS on states (node, prev_node)
                # starting from all (v, u) for v in adj[u]
                # Track distance (number of steps taken so far, starting at 1)
                # If we reach state (u, w) where w is a neighbor of u and w != prev_node_at_start
                # wait, this is getting complicated.
                
                # Simpler approach: for each u, check if there's a cycle through u of length < g
                # by doing BFS in the non-backtracking sense.
                # State: (current, previous)
                # Start: all (v, u) for v in adj[u], depth = 1
                # Goal: reach (u, w) for any w in adj[u], w != v_start, with depth < g
                # But we need to be careful: the cycle length is depth + 1 (to close the loop)
                # No: if we start at u, go to v (step 1), then ... to w (step d), then back to u (step d+1)
                # The cycle length is d+1. We need d+1 < g, so d < g-1.
                # In the BFS, when we're at state (w, x) with depth d, if w == u and x != v_start... 
                # Actually, we just need to check: can we get from (v, u) to (u, w) for some w?
                # The path is u -> v -> ... -> w -> u. Length = d + 1 where d is the BFS depth.
                # We need d + 1 < g => d < g - 1.
                
                # Let's do: for each u, BFS on non-backtracking states.
                # visited state: (node, prev) -> min_depth
                # Start: (v, u) for v in adj[u], depth 1
                # Expand: from (node, prev), go to (nxt, node) for nxt in adj[node], nxt != prev
                # Check: if node == u and depth < g - 1: cycle found!
                # Wait, if node == u, we've returned to start. The cycle is u -> ... -> u.
                # The length is depth. We need depth < g.
                # But we started at depth 1 (u->v). So if we reach (u, w) at depth d,
                # the cycle is u->v->...->w->u? No.
                # State (node, prev) means we're at node, came from prev.
                # If node == u, we're back at start. The path is u -> ... -> u.
                # The number of edges in the path is depth.
                # So if depth < g, we have a cycle of length < g.
                
                visited_nb = {}
                queue_nb = deque()
                for v in adj[u]:
                    state = (v, u)
                    visited_nb[state] = 1
                    queue_nb.append((v, u, 1))
                
                found_short_cycle = False
                while queue_nb:
                    node, prev, depth = queue_nb.popleft()
                    if depth >= g:
                        continue
                    for nxt in adj[node]:
                        if nxt == prev:
                            continue
                        new_state = (nxt, node)
                        if new_state not in visited_nb:
                            visited_nb[new_state] = depth + 1
                            if nxt == u:
                                # Found cycle of length depth + 1
                                if depth + 1 < g:
                                    girth_ok = False
                                    found_short_cycle = True
                                    break
                            queue_nb.append((nxt, node, depth + 1))
                    if found_short_cycle:
                        break
                if not girth_ok:
                    break
            
            if girth_ok:
                # Return edges
                edges = []
                for i in range(n_elems):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n_elems, edges
        
    # Fallback to LUW if search fails
    return luw_construct(k, g)

def luw_construct(k, g):
    # Original LUW code
    q = k
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
