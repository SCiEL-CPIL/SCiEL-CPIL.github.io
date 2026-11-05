import itertools
import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit

    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                if t == 1:
                    return True
                return False
        return False

    def modinv(a, m):
        if m <= 1:
            return None
        try:
            return pow(a, -1, m)
        except ValueError:
            return None

    def moore_bound(k, g):
        if k <= 1: return 0
        if g == 1: return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    def action_on_point(x, a, b, c, d, q):
        if x == q:
            if c == 0:
                return q
            else:
                inv_c = modinv(c, q)
                if inv_c is None:
                    return None
                return (a * inv_c) % q
        else:
            num = (a * x + b) % q
            den = (c * x + d) % q
            if den == 0:
                return q
            else:
                inv_den = modinv(den, q)
                if inv_den is None:
                    return None
                return (num * inv_den) % q

    def check_girth_fast(adj, n, g):
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            
            while queue:
                x = queue.popleft()
                dx = dist[x]
                if dx >= (g - 1) // 2 + 1:
                    break
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dx + 1
                        parent[y] = x
                        queue.append(y)
                    elif y != parent[x] and parent[y] != x:
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < g:
                            return False
        return True

    def luw_construct(q, d):
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

    def find_short_relation(gens, q, max_len):
        """
        Finds a word of length < max_len that evaluates to identity.
        Returns the word (list of generator indices) or None.
        """
        # BFS in the Cayley graph from identity (represented by q=infinity or 0)
        # Identity element in PGL(2,q) action:
        # We check if applying the sequence of generators to a point returns it.
        # It's enough to check for one point, say 0.
        # But for Cayley graphs, girth is the length of the shortest non-trivial relation.
        # We can search for words w such that w(0) = 0 and w is not the identity permutation.
        # Actually, if w(0)=0, w(1)=1, w(inf)=inf, it's identity.
        # Checking w(0)=0 is necessary but not sufficient.
        # However, for random generators, if w(0)=0 and len(w) < g, it's likely a short cycle.
        
        # Let's just do BFS on the points 0..q-1, inf.
        # We want to find a cycle of length < max_len.
        # BFS from 0.
        # If we visit a node at distance d, and we have an edge to a node at distance d' such that d + d' + 1 < max_len, we found a cycle.
        # This is equivalent to check_girth_fast but on the Cayley graph directly without building adjacency list.
        
        dist = {0: 0}
        parent = {0: None}
        queue = deque([(0, 0)]) # (point, dist)
        
        # We need to track the path or just detect collision.
        # Standard BFS girth check.
        
        while queue:
            x, dx = queue.popleft()
            if dx >= max_len - 1:
                continue
            for i, gen in enumerate(gens):
                y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                if y is None:
                    continue
                if y not in dist:
                    dist[y] = dx + 1
                    parent[y] = (x, i)
                    queue.append((y, dx + 1))
                else:
                    # Found a back edge or cross edge
                    # Cycle length is dist[x] + dist[y] + 1
                    # But we need to ensure it's not just the parent.
                    if parent[y] is not None and parent[y][0] != x:
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < max_len:
                            # Reconstruct path to identify generators?
                            # For now, just return True/False or the word
                            # To mutate, we need to know which generators.
                            # We can store the path.
                            # For simplicity, we'll just return the length and let the caller retry.
                            # Or we can return the path.
                            path_x = []
                            curr = x
                            while curr != 0:
                                prev, gen_idx = parent[curr]
                                path_x.append(gen_idx)
                                curr = prev
                            path_x.reverse()
                            
                            path_y = []
                            curr = y
                            while curr != 0:
                                prev, gen_idx = parent[curr]
                                path_y.append(gen_idx)
                                curr = prev
                            path_y.reverse()
                            
                            # The cycle is path_x + [gen_i] + reversed(path_y)
                            # The generator used to go x->y is i.
                            # The generators in path_x and path_y are involved.
                            # We can return the set of indices involved.
                            involved = set(path_x) | set(path_y) | {i}
                            return list(involved)
        return None

    def build_and_check_pgl(q, k, g, attempts=100):
        n = q + 1
        
        for attempt in range(attempts):
            if time.time() - start_time > timeout:
                return None
            
            # Generate random generators
            gens = []
            valid = True
            for _ in range(k):
                while True:
                    a = random.randint(0, q - 1)
                    b = random.randint(0, q - 1)
                    c = random.randint(0, q - 1)
                    d = random.randint(0, q - 1)
                    det = (a * d - b * c) % q
                    if det != 0:
                        break
                gens.append((a, b, c, d))
            
            # Local search
            for step in range(50): # Max 50 mutations
                if time.time() - start_time > timeout:
                    return None
                
                involved = find_short_relation(gens, q, g)
                if involved is None:
                    # No short cycle found
                    # Verify with full BFS just in case (find_short_relation might miss some if BFS depth is limited? No, it goes up to g-1)
                    # But find_short_relation only checks for cycles passing through 0?
                    # No, BFS from 0 covers all nodes in the connected component.
                    # If the graph is not connected, we might miss cycles in other components.
                    # PGL(2,q) is transitive, so the Cayley graph is connected if gens generate the group.
                    # Random gens likely generate the group.
                    
                    # Build graph and verify
                    adj = [set() for _ in range(n)]
                    is_simple = True
                    for i in range(n):
                        x = i if i < q else q
                        for j in range(k):
                            img = action_on_point(x, gens[j][0], gens[j][1], gens[j][2], gens[j][3], q)
                            if img is None:
                                is_simple = False
                                break
                            if img == i:
                                is_simple = False
                                break
                            adj[i].add(img)
                            adj[img].add(i)
                        if not is_simple:
                            break
                    
                    if not is_simple:
                        # Mutate a random generator
                        idx = random.randint(0, k-1)
                        while True:
                            a = random.randint(0, q - 1)
                            b = random.randint(0, q - 1)
                            c = random.randint(0, q - 1)
                            d = random.randint(0, q - 1)
                            det = (a * d - b * c) % q
                            if det != 0:
                                break
                        gens[idx] = (a, b, c, d)
                        continue

                    for i in range(n):
                        if len(adj[i]) != k:
                            is_simple = False
                            break
                    
                    if not is_simple:
                         idx = random.randint(0, k-1)
                         while True:
                            a = random.randint(0, q - 1)
                            b = random.randint(0, q - 1)
                            c = random.randint(0, q - 1)
                            d = random.randint(0, q - 1)
                            det = (a * d - b * c) % q
                            if det != 0:
                                break
                         gens[idx] = (a, b, c, d)
                         continue

                    # Check connectivity
                    visited = [False] * n
                    queue = deque([0])
                    visited[0] = True
                    count = 0
                    while queue:
                        u = queue.popleft()
                        count += 1
                        for v in adj[u]:
                            if not visited[v]:
                                visited[v] = True
                                queue.append(v)
                    if count != n:
                         # Not connected, mutate
                         idx = random.randint(0, k-1)
                         while True:
                            a = random.randint(0, q - 1)
                            b = random.randint(0, q - 1)
                            c = random.randint(0, q - 1)
                            d = random.randint(0, q - 1)
                            det = (a * d - b * c) % q
                            if det != 0:
                                break
                         gens[idx] = (a, b, c, d)
                         continue

                    if check_girth_fast(adj, n, g):
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return n, edges
                else:
                    # Mutate one of the involved generators
                    idx = random.choice(involved)
                    while True:
                        a = random.randint(0, q - 1)
                        b = random.randint(0, q - 1)
                        c = random.randint(0, q - 1)
                        d = random.randint(0, q - 1)
                        det = (a * d - b * c) % q
                        if det != 0:
                            break
                    gens[idx] = (a, b, c, d)
        
        return None

    # Strategy: Try LUW with q=k and various d values
    best_n = float('inf')
    best_result = None
    
    # Try LUW with q=k
    q = k
    d_min = max(3, g - 5)
    d_max = d_min + 5
    
    for d in range(d_min, d_max + 1):
        if time.time() - start_time > timeout:
            break
        n_luw, edges_luw = luw_construct(q, d)
        if n_luw < best_n:
            adj = [set() for _ in range(n_luw)]
            for i, j in edges_luw:
                adj[i].add(j)
                adj[j].add(i)
            valid = True
            for i in range(n_luw):
                if len(adj[i]) != k:
                    valid = False
                    break
            if valid and check_girth_fast(adj, n_luw, g):
                best_n = n_luw
                best_result = (n_luw, edges_luw)
    
    # If LUW didn't give a good result, try PGL search
    if best_n == float('inf') or best_n > n_min * 2:
        q_min = max(2, n_min - 1)
        q_max = min(best_n - 2 if best_n != float('inf') else q_min + 100, q_min + 200)
        
        if q_max >= q_min:
            count_tried = 0
            for curr_q in range(q_min, q_max + 1):
                if time.time() - start_time > timeout:
                    break
                if not is_prime_power(curr_q):
                    continue
                
                res = build_and_check_pgl(curr_q, k, g, attempts=50)
                if res:
                    n, edges = res
                    if n < best_n:
                        best_n = n
                        best_result = res
                        return best_result
                
                count_tried += 1
                if count_tried > 50:
                    break
    
    if best_result is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_result = luw_construct(k, d)
    
    return best_result

# EVOLVE-BLOCK-END
