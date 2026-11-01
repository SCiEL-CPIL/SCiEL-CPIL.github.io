import time
from collections import deque
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_n = float('inf')
    best_edges = None
    
    def verify_graph(n, edges, k, g):
        if n == 0:
            return False
        adj = [[] for _ in range(n)]
        seen_edges = set()
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u, v), max(u, v))
            if key in seen_edges:
                return False
            seen_edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        max_depth = (g - 1) // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([(u, -1)])
            while q:
                curr, par = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                for v in adj[curr]:
                    if v == par:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append((v, curr))
                    else:
                        if v == parent[curr]:
                            continue
                        cycle_len = dist[curr] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_luw_graph(q_val, d_val):
        if q_val == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q_val
            sub = lambda x,y: (x-y)%q_val
            mul = lambda x,y: x*y%q_val
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        def get_rule(dv):
            r = dict(rule)
            for c in range(5, dv+1):
                rc = (c-5)%4
                c0 = c-(c-5)%4
                if rc==0: r[c]=(1,c0-2)
                elif rc==1: r[c]=(c0-1,1)
                elif rc==2: r[c]=(c0,1)
                else: r[c]=(1,c0+1)
            return r
        r = get_rule(d_val)
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0]*(d_val-1)
                for c in range(2, d_val+1):
                    a,b = r[c]
                    if side==0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,)+(0,)*d_val)
        index = {start: 0}
        order = [start]
        edges = []
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def try_circulant(n, k, g):
        # Try to find a 4-regular circulant graph
        # C(n, a, b) means edges i to i+a, i-a, i+b, i-b
        # Must ensure a, b distinct and not 0, n/2
        if n < 2*k:
            return None
        # Heuristic: try small a, b
        candidates = []
        for a in range(1, n//2):
            if math.gcd(a, n) != 1: continue
            for b in range(a+1, n//2):
                if math.gcd(b, n) != 1: continue
                if a == b or a == n-b or b == n-a: continue
                # Check if edges are unique
                # i+a == i+b? No if a!=b
                # i+a == i-b? a == -b mod n?
                if (a + b) % n == 0: continue
                candidates.append((a, b))
        
        # Shuffle to find one quickly? Or just iterate?
        # We need to verify girth.
        # Verification is expensive.
        # Limit number of trials.
        random.shuffle(candidates)
        for a, b in candidates[:100]:
            edges = []
            for i in range(n):
                edges.append((i, (i+a)%n))
                edges.append((i, (i+b)%n))
            # Deduplicate?
            # (i, i+a) and (i+a, i) are same.
            # My verify_graph handles dedup check.
            # But I should provide unique edges to verify_graph to be safe.
            unique_edges = set()
            for u, v in edges:
                if u > v: u, v = v, u
                unique_edges.add((u, v))
            if len(unique_edges) != n*k//2:
                continue
            if verify_graph(n, list(unique_edges), k, g):
                return n, list(unique_edges)
        return None

    # 1. Try LUW
    max_d = int(math.log(5000 / 2, k)) if k > 1 else 10
    min_d = max(3, (g-2)//2)
    
    for d_val in range(min_d, min(max_d + 1, min_d + 10)):
        if time.time() > deadline - 10:
            break
        est_n = 2 * (k ** d_val)
        if est_n > 10000:
            continue
        try:
            n, edges = build_luw_graph(k, d_val)
            if n < best_n and verify_graph(n, edges, k, g):
                best_n = n
                best_edges = edges
                if n <= 500:
                    break
        except Exception:
            pass

    # 2. Try Circulants if LUW didn't find something small
    if best_n > 1000:
        # Moore bound
        moore = 1 + k * sum((k-1)**i for i in range((g-1)//2))
        for n in range(moore, moore + 200):
            if time.time() > deadline - 5:
                break
            if n % 2 != 0: continue # 4-regular requires even N? No, sum of degrees = 4N must be even. N can be odd.
            # But circulant C(n, a, b) works for any N.
            res = try_circulant(n, k, g)
            if res:
                n_c, edges_c = res
                if n_c < best_n:
                    best_n = n_c
                    best_edges = edges_c
                break

    if best_edges is None:
        return 0, []
    return best_n, best_edges
