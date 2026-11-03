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
            
        # Generate SL(2, q) elements
        # Elements are (a, b, c, d) with ad - bc = 1 mod q
        elements = []
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # ad - bc = 1 => ad = 1 + bc
                    # If a is invertible, d = (1+bc)*a^-1
                    # If a is not invertible (a=0), then -bc = 1.
                    if a != 0:
                        a_inv = pow(a, q-2, q)
                        for d in range(q):
                            # ad - bc = 1 => d = (1+bc)*a_inv
                            # This determines d uniquely for given a,b,c?
                            # No, we iterate a,b,c and calculate d.
                            pass
        # Better generation:
        # For each a,b,c, check if 1+bc is divisible by a? No.
        # Just iterate all 15 tuples? q^4 is 11^4 = 14641. Very small.
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
        
        # Random search for S
        found = False
        attempts = 0
        while time.time() < deadline - 5 and attempts < 5000:
            attempts += 1
            if not self_inv:
                break
            s0 = random.choice(self_inv)
            # Pick (k-1)/2 pairs
            pairs = []
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
            
            # Check girth
            # Build adjacency list
            adj = [[] for _ in range(n_elems)]
            for i, u in enumerate(elems):
                for s in current_set:
                    v = mul(u, s)
                    j = elem_to_idx[v]
                    adj[i].append(j)
            
            # Girth check: BFS from each vertex up to depth (g-1)//2
            # If we find a cycle of length < g, fail
            girth_ok = True
            max_depth = (g - 1) // 2
            for u in range(n_elems):
                dist = [-1] * n_elems
                dist[u] = 0
                q_bfs = deque([u])
                while q_bfs:
                    cur = q_bfs.popleft()
                    if dist[cur] >= max_depth:
                        continue
                    for v in adj[cur]:
                        if dist[v] == -1:
                            dist[v] = dist[cur] + 1
                            q_bfs.append(v)
                        else:
                            # Cycle found
                            if dist[v] + dist[cur] + 1 < g:
                                girth_ok = False
                                break
                    if not girth_ok:
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
