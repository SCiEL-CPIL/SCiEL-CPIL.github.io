import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0: return False
        return True

    def is_prime_power(n):
        if n < 2: return False
        if is_prime(n): return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0: t //= p
                return t == 1
        return False

    def is_quadratic_residue(a, q):
        a = a % q
        if a == 0: return True
        if not is_prime_power(q): return True
        p = None
        for cand in range(2, int(math.isqrt(q)) + 1):
            if q % cand == 0:
                t = q
                while t % cand == 0: t //= cand
                if t == 1:
                    p = cand
                    break
        if p is None: p = q
        a_p = a % p
        if a_p == 0: return True
        return pow(a_p, (p - 1) // 2, p) == 1

    def modinv(a, m):
        if m <= 1: return None
        a = a % m
        if a == 0: return None
        old_r, r = a, m
        old_s, s = 1, 0
        while r != 0:
            qv = old_r // r
            old_r, r = r, old_r - qv * r
            old_s, s = s, old_s - qv * s
        if old_r != 1: return None
        return old_s % m

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

    def check_girth_fast(adj, n, g):
        for u in range(n):
            visited = set()
            queue = deque()
            for v in adj[u]:
                queue.append((v, u))
                visited.add((v, u))
            depth = 1
            while queue and depth < g:
                new_queue = deque()
                for (cur, prev) in queue:
                    for nxt in adj[cur]:
                        if nxt == prev: continue
                        if nxt == u: return False
                        state = (nxt, cur)
                        if state not in visited:
                            visited.add(state)
                            new_queue.append(state)
                queue = new_queue
                depth += 1
        return True

    def is_connected(adj, n):
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k: return False
        if not is_connected(adj, n): return False
        if not check_girth_fast(adj, n, g): return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j: edges.append((i, j))
        return (n, edges)

    def luw_construct(k, g):
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

    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    def local_search(n, k, g, budget):
        """
        Local search to find a k-regular graph with girth >= g.
        Starts with a random graph and tries to break short cycles.
        """
        if n * k % 2 != 0:
            return None
        if n < n_min:
            return None
            
        end_time = time.time() + budget
        while time.time() < end_time:
            # Generate random k-regular graph
            adj = [set() for _ in range(n)]
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                
            if not valid:
                continue
                
            if not is_connected(adj, n):
                continue
                
            # Local search to increase girth
            for _ in range(1000): # max iterations
                if time.time() > end_time:
                    break
                if check_girth_fast(adj, n, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return (n, edges)
                
                # Find a short cycle
                # We can just try random edge swaps that don't create loops/multiedges
                # and hope to break short cycles
                # A more targeted approach: find a cycle of length < g and break it
                # For simplicity, we do random 2-switches
                u = random.randrange(n)
                v = random.choice(list(adj[u]))
                w = random.randrange(n)
                if w == u or w == v: continue
                x = random.choice(list(adj[w]))
                if x == u or x == v or x == w: continue
                if v in adj[w] or x in adj[u]: continue
                
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                
                adj[u].add(w)
                adj[w].add(u)
                adj[v].add(x)
                adj[x].add(v)
                
                if not is_connected(adj, n):
                    # Revert
                    adj[u].remove(w)
                    adj[w].remove(u)
                    adj[v].remove(x)
                    adj[x].remove(v)
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[w].add(x)
                    adj[x].add(w)
                    continue

        return None

    # Try local search for n values near the Moore bound
    # The LUW graph is large, so we try to find a smaller one
    # We try n from n_min to min(best_n, n_min + 100)
    for trial_n in range(n_min, min(best_n, n_min + 50)):
        if time.time() > deadline - 10.0:
            break
        budget = min(deadline - 10.0 - time.time(), 5.0)
        if budget <= 0:
            break
        res = local_search(trial_n, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            # If we found one, we can try to go lower? No, we started from n_min.
            # So this is the best possible for this n.
            # We can break or continue to see if we can find a smaller n?
            # We are iterating n upwards, so the first one found is the smallest.
            break

    return best_result

# EVOLVE-BLOCK-END
