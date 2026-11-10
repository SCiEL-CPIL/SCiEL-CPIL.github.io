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

    # ---------- Improved Local Search ----------
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
        
        def do_swap(a1, b1, a2, b2):
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

        def find_short_cycle():
            # BFS from random start to find a short cycle
            for _ in range(10):
                start = random.randint(0, n-1)
                dist = [-1]*n
                parent = [-1]*n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    du = dist[u]
                    if du >= 9: # We care about cycles < 10, so depth 9 is enough to detect
                        break
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            queue.append(v)
                        elif parent[u] != v and parent[v] != u:
                            clen = du + dist[v] + 1
                            if clen < 10:
                                # Reconstruct cycle
                                cycle = []
                                cur = u
                                while cur != v:
                                    cycle.append(cur)
                                    cur = parent[cur]
                                cycle.append(v)
                                # cycle is path from u to v via parent, plus edge (u,v)
                                # Actually, the cycle is u -> ... -> v -> u
                                # We return the edges in the cycle
                                edges_in_cycle = []
                                prev = u
                                for node in cycle:
                                    if node != v:
                                        edges_in_cycle.append((node, parent[node]))
                                edges_in_cycle.append((u, v))
                                return edges_in_cycle
            return None

        max_iter = 50000
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
            cycle_edges = find_short_cycle()
            if not cycle_edges:
                # No short cycle found by random BFS, might be girth >= 10
                # But check_girth is the final arbiter. 
                # If check_girth passed, we would have returned.
                # So if check_girth failed but find_short_cycle didn't find one,
                # it's a heuristic miss. Continue.
                continue

            # Pick a random edge from the cycle to swap
            # We need to find a compatible edge to swap with.
            # For a bipartite graph, we swap (u,v) and (u',v') where u,u' in Left, v,v' in Right.
            # Let's pick a random edge (u,v) from the cycle.
            (u, v) = random.choice(cycle_edges)
            # Ensure u is in Left (0..159) and v in Right (160..319)
            if u >= half:
                u, v = v, u
                
            # Pick a random other edge (u', v')
            # u' should be different from u, v' different from v
            # And (u,v') and (u',v) must not exist
            swapped = False
            for _ in range(20): # Try a few random swaps
                u2 = random.randint(0, half-1)
                if u2 == u:
                    continue
                v2 = random.randint(half, n-1)
                if v2 == v:
                    continue
                if v2 in adj[u] or v in adj[u2]:
                    continue
                if do_swap(u, v, u2, v2):
                    swapped = True
                    break
            # If no swap succeeded, try another cycle or continue

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
            # Try local search with a generous time limit
            result = local_search_4_10(100)
            if result is not None:
                return result

    # Fallback to L-U-W
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
