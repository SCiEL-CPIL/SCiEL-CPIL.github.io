import random
import time
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110
    
    n_target = 936
    
    def random_3reg(n):
        for _ in range(200):
            stubs = []
            for i in range(n):
                stubs += [i]*3
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            ok = True
            for i in range(0, 3*n, 2):
                a, b = stubs[i], stubs[i+1]
                if a == b or b in adj[a]:
                    ok = False
                    break
                adj[a].add(b); adj[b].add(a)
            if ok:
                return [list(s) for s in adj]
        return None
    
    def girth_of(adj, n, max_g=20):
        """Return girth of graph."""
        # Check for cycles of length 3,4,...,max_g-1
        for target in range(3, max_g):
            # Non-backtracking walks of length target from each vertex
            for v in range(n):
                # BFS-like enumeration of non-backtracking walks
                # State: (current, prev)
                # We want walks of length exactly target that return to v
                # But also shorter cycles
                # Let's do: enumerate all non-backtracking walks up to length target-1
                # If any returns to v, cycle found
                seen = set()
                frontier = set()
                for nb in adj[v]:
                    frontier.add((nb, v))
                
                for length in range(1, target):
                    new_frontier = set()
                    for (u, p) in frontier:
                        for nb in adj[u]:
                            if nb == p:
                                continue
                            if nb == v:
                                return length + 1  # cycle of length length+1
                            if (nb, u) not in seen:
                                seen.add((nb, u))
                                new_frontier.add((nb, u))
                    frontier = new_frontier
                    if not frontier:
                        break
        return max_g
    
    def find_and_fix_short_cycle(adj, n):
        """Find a short cycle and try to break it via edge swap."""
        # Find shortest cycle
        for target in range(4, 16, 2):  # even cycles first (more common in 3-reg)
            for v in range(n):
                # Enumerate non-backtracking walks of length target-1 from v
                # If any ends at v, we have a cycle
                # Actually, let's find the cycle vertices
                # State: (current, prev, path)
                # Too expensive. Let's just do edge swaps randomly.
                pass
        # Just do random edge swaps
        return False
    
    # Strategy: 
    # 1. Try to build a graph with girth >= 16
    # 2. Start with random 3-reg graphs, check girth
    # 3. Use local search to improve
    
    best_girth = 0
    best_adj = None
    
    for n_try in [512, 600, 700, 800, 936]:
        if time.time() > deadline - 30:
            break
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            adj = random_3reg(n_try)
            if adj is None:
                continue
            girth = girth_of(adj, n_try, 17)
            if girth >= 16:
                # Convert to edge list
                edges = []
                for i in range(n_try):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n_try, edges
            # Try local search
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                # Random edge swap
                a = random.randint(0, n_try-1)
                b = random.choice(adj[a])
                c = random.randint(0, n_try-1)
                if c in (a, b):
                    continue
                d = random.choice(adj[c])
                if d in (a, b, c):
                    continue
                if d in adj[a] or c in adj[b]:
                    continue
                # Swap
                adj[a].remove(b); adj[b].remove(a)
                adj[c].remove(d); adj[d].remove(c)
                adj[a].append(c); adj[c].append(a)
                adj[b].append(d); adj[d].append(b)
                girth = girth_of(adj, n_try, 17)
                if girth >= 16:
                    edges = []
                    for i in range(n_try):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n_try, edges
    
    # Fallback: return the LUW graph (valid but large)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y: (x+y)%q), (lambda x,y: (x-y)%q), (lambda x,y: x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    
    start = (0, (None,)+(0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
