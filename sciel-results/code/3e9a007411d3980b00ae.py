import time
import random
import collections
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Fallback: LUW construction from seed
    def luw_fallback():
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

    # Primary Strategy: PSL(2, q) Cayley Graphs
    # For k=5, we need 5 generators.
    # The graph is vertex-transitive and regular.
    # We search for q such that n = q(q^2-1)/2 is small and < 1116.
    # q=11 -> n=660 (too small, Moore bound for 5,9 is ~426, but girth likely < 9)
    # q=13 -> n=1092 (target)
    # q=17 -> n=1360 (fallback if q=13 fails)
    
    if k != 5:
        # Fallback to LUW for other k
        return luw_fallback()

    def generate_psl2(q):
        """
        Generates elements of PSL(2, q) as tuples (a, b, c, d) representing
        matrices [[a, b], [c, d]] with det=1.
        Returns:
        - elements: list of tuples
        - idx_map: dict from tuple to index
        - mul_table: 2D list for multiplication (optional, but good for speed if n is small)
        - orders: list of orders of elements
        - inverses: list of inverse indices
        """
        elements = []
        idx_map = {}
        
        # Helper to normalize (a,b,c,d) ~ (-a,-b,-c,-d)
        # We pick the representative where the first non-zero entry is in 1..(q-1)/2
        # If all zero (impossible for det=1), handle edge case.
        
        def normalize(a, b, c, d):
            # Find first non-zero
            vals = [a, b, c, d]
            for i, v in enumerate(vals):
                if v != 0:
                    if v > (q - 1) // 2:
                        # Flip signs
                        return ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
                    break
            return (a, b, c, d)

        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # ad - bc = 1 => ad = 1 + bc
                    if a == 0:
                        if b == 0:
                            continue
                        # -bc = 1 => bc = q-1
                        if (b * c) % q != (q - 1):
                            continue
                        d = 0
                    else:
                        # d = (1 + bc) * a^{-1}
                        a_inv = pow(a, q - 2, q)
                        d = ((1 + b * c) % q) * a_inv % q
                    
                    # Verify det
                    if (a * d - b * c) % q != 1:
                        continue
                    
                    m = normalize(a, b, c, d)
                    if m not in idx_map:
                        idx_map[m] = len(elements)
                        elements.append(m)
        
        n = len(elements)
        
        # Precompute multiplication table for speed
        # mul_table[i][j] = index of elements[i] * elements[j]
        mul_table = [[0]*n for _ in range(n)]
        inverses = [0]*n
        
        for i in range(n):
            a1, b1, c1, d1 = elements[i]
            for j in range(n):
                a2, b2, c2, d2 = elements[j]
                # Matrix mult
                na = (a1*a2 + b1*c2) % q
                nb = (a1*b2 + b1*d2) % q
                nc = (c1*a2 + d1*c2) % q
                nd = (c1*b2 + d1*d2) % q
                prod = normalize(na, nb, nc, nd)
                mul_table[i][j] = idx_map[prod]
                
        # Compute inverses and orders
        orders = [1]*n
        # Inverse of [[a,b],[c,d]] is [[d,-b],[-c,a]]
        for i in range(n):
            a, b, c, d = elements[i]
            inv = normalize(d, (q-b)%q, (q-c)%q, a)
            inv_idx = idx_map[inv]
            inverses[i] = inv_idx
            
            # Order calculation can be expensive, but we only need to know if order is 2
            # Order 2 if M^2 = I
            # M^2 = [[a^2+b^2, ab+bd], [ca+dc, cb+d^2]]
            # Actually, for PSL(2,q), an element has order 2 if it is an involution.
            # We can just check if mul_table[i][i] == 0 (identity)
            if mul_table[i][i] == 0:
                orders[i] = 2
            else:
                # We don't strictly need full order, just that it's not 2
                # But for generator selection, we need distinctness and non-involution for s1, s2
                # We can assume order > 2 if not involution? No, could be 3, 4, etc.
                # But we just need s1 != s1^-1.
                orders[i] = 3 # Placeholder for "not 2"
                
        return elements, idx_map, mul_table, inverses, orders

    def check_girth_cayley(mul_table, inverses, generators, g):
        """
        Checks if the Cayley graph with given generators has girth >= g.
        Uses BFS from identity.
        """
        n = len(mul_table)
        identity = 0
        # BFS
        dist = [-1] * n
        dist[identity] = 0
        queue = collections.deque([identity])
        
        # We need to check for cycles.
        # A cycle of length L exists if there is a word of length L in generators
        # that equals identity.
        # Alternatively, BFS from identity: if we find a path back to identity of length < g, 
        # that's a cycle. But BFS finds shortest path to each node.
        # If dist[u] + 1 + dist[v] < g for an edge (u,v) not in tree? 
        # Actually, simpler: 
        # If there is a word w of length L < g such that w = e, then girth <= L.
        # We can enumerate words up to length g-1.
        # Number of words is k * (2k-1)^(L-1). For k=5, L=8: 5 * 9^7 ~ 400M. Too slow.
        
        # Better: BFS from identity.
        # If we reach a node u at distance d, and we have an edge from u to v where dist[v] is already set,
        # and dist[u] + 1 + dist[v] < g? No.
        
        # Standard Girth Check for Cayley Graph:
        # The girth is the length of the shortest non-trivial relation.
        # We can do BFS from identity.
        # If we find a cycle, it corresponds to a closed walk.
        # A simple cycle of length L corresponds to a word of length L that is identity.
        # However, BFS tree edges don't form cycles. Non-tree edges (u,v) create a cycle of length dist[u] + 1 + dist[v].
        # If dist[u] + 1 + dist[v] < g, then girth < g.
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g - 1:
                continue
            for s in generators:
                v = mul_table[u][s]
                if dist[v] == -1:
                    dist[v] = du + 1
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    # Cycle length is du + 1 + dist[v]
                    # But we must ensure it's a simple cycle? 
                    # In a BFS tree, the path from root to u and root to v are shortest.
                    # The union of these paths plus edge (u,v) forms a cycle.
                    # The length is dist[u] + dist[v] + 1.
                    # If this is < g, we have a short cycle.
                    if dist[u] + dist[v] + 1 < g:
                        return False
        return True

    # Try q values
    # q must be prime for PSL(2,q) to be simple group? No, PSL(2,q) is defined for prime powers.
    # But for small q, primes are sufficient.
    candidates_q = [13, 17, 19, 23]
    
    best_n = float('inf')
    best_edges = []
    
    for q in candidates_q:
        if time.time() > deadline:
            break
            
        n = q * (q * q - 1) // 2
        if n >= best_n:
            continue
        if n > 1116: # We want to beat 1116
            # If we haven't found anything better, we might still want to return this if it's valid?
            # But 1116 is the seed. We want < 1116.
            # If q=13 fails, q=17 gives 1360 which is worse.
            # So we only care about q=13 for improvement.
            # But if q=13 is too hard to find generators, we fallback to LUW.
            pass

        elements, idx_map, mul_table, inverses, orders = generate_psl2(q)
        
        # Find involutions (order 2)
        involutions = [i for i in range(n) if orders[i] == 2]
        
        if not involutions:
            continue
            
        # Search for 5 generators: s1, s1^-1, s2, s2^-1, s3
        # s1, s2 must not be involutions (order != 2)
        # s3 must be an involution
        # All 5 must be distinct.
        
        found = False
        attempts = 0
        max_attempts = 2000 # Adjust based on time
        
        while attempts < max_attempts and not found:
            if time.time() > deadline:
                break
            attempts += 1
            
            # Pick random s1, s2 (non-involutions)
            # Pick random s3 (involution)
            
            # To ensure diversity, we can pick from distinct conjugacy classes? 
            # For now, random sampling.
            
            s1 = random.randint(0, n - 1)
            if orders[s1] == 2:
                continue
            s2 = random.randint(0, n - 1)
            if orders[s2] == 2:
                continue
            s3 = random.choice(involutions)
            
            s1_inv = inverses[s1]
            s2_inv = inverses[s2]
            
            candidates = [s1, s1_inv, s2, s2_inv, s3]
            
            # Check distinctness
            if len(set(candidates)) != 5:
                continue
                
            # Check if s3 is in {s1, s1_inv, s2, s2_inv}
            # Already covered by set size.
            
            # Check girth
            # This is the expensive part.
            if check_girth_cayley(mul_table, inverses, candidates, g):
                # Construct edges
                edges = []
                for i in range(n):
                    for s in candidates:
                        neighbor = mul_table[i][s]
                        j = neighbor
                        if i < j:
                            edges.append((i, j))
                
                # Verify degree
                # Each vertex has 5 neighbors.
                # Edges count should be n * 5 / 2
                if len(edges) == n * 5 // 2:
                    best_n = n
                    best_edges = edges
                    found = True
                    break
                    
        if found:
            return best_n, best_edges

    # Fallback to LUW
    return luw_fallback()

