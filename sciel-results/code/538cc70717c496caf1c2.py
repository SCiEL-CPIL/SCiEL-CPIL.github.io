import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2
            return k * ((k - 1) ** (h - 1) - 1) // (k - 2) + 2
        else:
            h = (g - 1) // 2
            total = 1
            for i in range(h):
                total += k * (k - 1) ** i
            return total

    lb = moore_bound(k, g)

    def check_valid(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v:
                return False
            if v in adj[u]:
                return False
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Girth check
        for s in range(n):
            # BFS from s
            # dist[v] = shortest distance from s to v
            # To check for cycles < g, we check if any edge (u,v) has dist[u] + 1 + dist[v] < g?
            # No, that checks for odd cycles or even cycles passing through s?
            # Better: Non-backtracking BFS.
            visited = set()
            q = deque()
            q.append((s, -1, 1))
            visited.add((s, -1))
            while q:
                u, prev, depth = q.popleft()
                if depth >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s:
                        if depth + 1 < g:
                            return False
                    if (v, u) not in visited:
                        visited.add((v, u))
                        q.append((v, u, depth + 1))
        return True

    def luw_construct(k, g):
        q = k
        d = g - 5
        if d < 3:
            d = 3
        if d % 2 == 0:
            d += 1 # Ensure odd d for better girth properties? Or just follow standard.
        
        # Actually, for k=5, q=5.
        # Standard LUW CD(d,q) has degree q.
        # If q=5, degree 5.
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0: return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0: return False
            return True

        if not is_prime(q):
            return None

        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    best_result = None

    # 1. Try LUW
    luw = luw_construct(k, g)
    if luw is not None:
        n_luw, edges_luw = luw
        if check_valid(n_luw, edges_luw):
            best_result = (n_luw, edges_luw)

    # 2. Try Circulant graphs for k=5
    if k == 5:
        # S = {s1, -s1, s2, -s2, N/2}
        # N must be even.
        start_n = lb
        # Limit search time
        max_n = 1000
        for N in range(start_n, max_n + 1, 2):
            if time.time() > deadline - 10:
                break
            if N % 2 != 0:
                continue
            half_N = N // 2
            # Try small s1, s2
            # s1, s2 should be distinct and not 0, not N/2
            # To avoid duplicates and symmetry, assume 1 <= s1 < s2 < N/2
            # But we can just iterate a few candidates.
            candidates_s1 = [1, 2, 3, 5, 7, 11]
            candidates_s2 = [1, 2, 3, 5, 7, 11, 13, 17]
            
            found = False
            for s1 in candidates_s1:
                if s1 >= half_N: continue
                for s2 in candidates_s2:
                    if s2 <= s1 or s2 >= half_N: continue
                    # Construct edges
                    edges = []
                    valid = True
                    for i in range(N):
                        # Neighbors: i+s1, i-s1, i+s2, i-s2, i+half_N
                        for s in [s1, -s1, s2, -s2, half_N]:
                            v = (i + s) % N
                            u = i
                            if u > v: u, v = v, u
                            edges.append((u, v))
                    
                    # Deduplicate and check simple
                    edge_set = set()
                    for u, v in edges:
                        if u == v:
                            valid = False
                            break
                        if (u, v) in edge_set:
                            valid = False
                            break
                        edge_set.add((u, v))
                    if not valid: continue
                    if len(edge_set) != N * 5 // 2: continue
                    
                    # Check girth
                    if check_valid(N, list(edge_set)):
                        if best_result is None or N < best_result[0]:
                            best_result = (N, list(edge_set))
                        found = True
                        break
                if found: break
            if found and N < (best_result[0] if best_result else 10**9):
                # We found a valid one, can we stop?
                # Not necessarily, but if N is close to lb, it's good.
                # However, we want the smallest.
                # If we iterate N upwards, the first valid one is the smallest in this family.
                # But we might miss a smaller N if we skipped it? No, we iterate from lb.
                # So the first valid N is the best circulant.
                # We can break the N loop if we found one?
                # Yes, because we iterate N increasing.
                break

    if best_result is None:
        # Fallback
        if luw is not None:
            best_result = luw
        else:
            # Trivial
            n = k + 1
            edges = []
            for i in range(n):
                for j in range(1, k // 2 + 1):
                    a = i
                    b = (i + j) % n
                    if a > b: a, b = b, a
                    edges.append((a, b))
            best_result = (n, edges)

    return best_result
