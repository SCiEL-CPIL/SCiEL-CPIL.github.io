import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    DEADLINE = 145.0

    def elapsed():
        return time.time() - t0

    # ---------- L-U-W fallback ----------
    def luw_construct(k, g):
        q = k
        if q not in [2, 3, 4, 5, 7, 11, 13]:
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0: return False
                return True
            q = 4
            while not (is_prime(q) or q == 4):
                q += 1

        if g % 2 == 1:
            d = max(3, g - 5)
            if d % 2 == 0: d += 1
        else:
            d = max(4, g - 4)
            if d % 2 == 1: d += 1

        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def add(x,y): return x^y
            def sub(x,y): return x^y
            def mul(x,y): return table[x][y]
        else:
            def add(x,y): return (x+y)%q
            def sub(x,y): return (x-y)%q
            def mul(x,y): return (x*y)%q

        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r = (c-5)%4
            c0 = c - (c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1-side), tuple(y)

        start = (0, (None,)+(0,)*d)
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

    # ---------- Fast girth checker ----------
    def check_girth(adj, n, min_girth):
        for start in range(n):
            dist = [-1]*n
            parent = [-1]*n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= min_girth - 1:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        clen = du + dist[v] + 1
                        if clen < min_girth:
                            return False
        return True

    # ---------- Local search for (4,10)-cage ----------
    def local_search_4_10(time_limit):
        n = 320
        half = 160
        random.seed(42)

        # Build random 4-regular bipartite graph
        perms = []
        for _ in range(4):
            p = list(range(half))
            random.shuffle(p)
            perms.append(p)

        adj = [[] for _ in range(n)]
        for i in range(half):
            for pi in perms:
                j = pi[i]
                rv = half + j
                adj[i].append(rv)
                adj[rv].append(i)

        start_time = time.time()
        
        # Helper to perform a 2-switch
        def do_swap(a1, b1, a2, b2):
            # Remove (a1, b1) and (a2, b2)
            # Add (a1, b2) and (a2, b1)
            # a1, a2 are left (0..159), b1, b2 are right (160..319)
            if b2 in adj[a1] or b1 in adj[a2]:
                return False
            adj[a1].remove(b1)
            adj[a1].append(b2)
            adj[b1].remove(a1)
            adj[b1].append(a2)
            
            adj[a2].remove(b2)
            adj[a2].append(b1)
            adj[b2].remove(a2)
            adj[b2].append(a1)
            return True

        max_iter = 100000
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            if elapsed() > DEADLINE - 10:
                break

            # Check girth
            if check_girth(adj, n, g):
                edges = []
                for i in range(half):
                    for j in adj[i]:
                        edges.append((i, j))
                return n, edges

            # Find a short cycle
            # We can pick a random edge and check if it's in a short cycle
            # Or just pick random edges to swap and hope.
            # Better: Find a specific short cycle.
            # For speed, we can just do random swaps.
            # Random 2-switch:
            a1 = random.randint(0, half-1)
            b1 = random.choice(adj[a1])
            a2 = random.randint(0, half-1)
            if a2 == a1:
                continue
            b2 = random.choice(adj[a2])
            if b2 == b1:
                continue
            
            if do_swap(a1, b1, a2, b2):
                # Swap successful
                pass
            else:
                continue

        # Final check
        if check_girth(adj, n, g):
            edges = []
            for i in range(half):
                for j in adj[i]:
                    edges.append((i, j))
            return n, edges
        return None

    # ---------- Main logic ----------
    if k == 4 and g == 10:
        if elapsed() < DEADLINE - 30:
            result = local_search_4_10(60)
            if result is not None:
                return result

    # Fallback to L-U-W
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
