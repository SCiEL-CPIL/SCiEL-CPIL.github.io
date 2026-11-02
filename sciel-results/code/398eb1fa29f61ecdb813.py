"""Portfolio SEARCH: try multiple explicit families, keep best (smallest) valid graph."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    deadline = time.time() + 115.0  # safety margin before 150s kill
    rng = random.Random(12345)

    def check_deadline():
        return time.time() > deadline

    # --- Utility: validate graph (simple, k-regular, girth >= g) ---
    def validate(n, edges, k, g):
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Girth check using BFS with early termination
        for s in range(n):
            dist = {}
            q = deque()
            for nb in adj[s]:
                dist[(nb, s)] = 1
                q.append((nb, s))
            while q:
                u, prev = q.popleft()
                d = dist[(u, prev)]
                if d >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    nd = d + 1
                    if v == s:
                        if nd < g:
                            return False
                        continue
                    key = (v, u)
                    if key not in dist or dist[key] > nd:
                        dist[key] = nd
                        q.append((v, u))
        return True

    def try_graph(n, edges, k, g):
        if not validate(n, edges, k, g):
            return None
        return n, edges

    # --- Family: Incidence graph of a projective plane (for k = q+1, prime power q) ---
    def is_prime_power(m):
        if m < 2:
            return False
        for p in range(2, int(math.isqrt(m)) + 1):
            if m % p == 0:
                x = m
                while x % p == 0:
                    x //= p
                if x == 1:
                    return True
                break
        if m > 1:
            for p in range(2, int(math.isqrt(m)) + 1):
                if m % p == 0:
                    return False
            return True
        return False

    def incidence_family():
        results = []
        if k < 3:
            return results
        q = k - 1
        if not is_prime_power(q):
            return results
        n_pts = q * q + q + 1
        
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        
        lines = []
        for a in range(q):
            for b in range(q):
                lines.append((a, b, 1))
            lines.append((a, 1, 0))
        lines.append((1, 0, 0))
        
        n_total = 2 * n_pts
        edges = []
        for i, pt in enumerate(points):
            x, y, z = pt
            for j, ln in enumerate(lines):
                a, b, c = ln
                if (a * x + b * y + c * z) % q == 0:
                    edges.append((i, n_pts + j))
        
        results.append((n_total, edges))
        return results

    def random_regular_graph(n, k, rng, max_iters=2000):
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
        for attempt in range(max_iters):
            half_edges = []
            for v in range(n):
                half_edges.extend([v] * k)
            rng.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if not valid:
                continue
            has_multi = False
            for u in range(n):
                if len(adj[u]) != len(set(adj[u])):
                    has_multi = True
                    break
            if has_multi:
                continue
            for _ in range(100):
                e1 = (rng.randint(0, n-1), rng.randint(0, n-1))
                e2 = (rng.randint(0, n-1), rng.randint(0, n-1))
                a, b = e1
                c, d = e2
                if a == c or a == d or b == c or b == d:
                    continue
                if b not in adj[a] or d not in adj[c]:
                    continue
                if d in adj[a] or b in adj[c]:
                    continue
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].remove(d)
                adj[d].remove(c)
                adj[a].append(d)
                adj[d].append(a)
                adj[c].append(b)
                adj[b].append(c)
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        return None

    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if n < best_n:
            result = try_graph(n, edges, k, g)
            if result is not None:
                best = result
                best_n = n
        if check_deadline():
            raise TimeoutError

    # --- Family 1: Lazebnik-Ustimenko-Woldar (from seed) ---
    def luw_family(q=None, d=None):
        if q is None:
            q = k
        if d is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    # --- Family 2: Circulant / folded circulant constructions ---
    def circulant_family():
        results = []
        if g % 2 == 1:
            r = (g-1)//2
            moore = 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g//2
            moore = k * sum((k-1)**i for i in range(r))
        
        max_n = moore + 500
        if k % 2 == 0:
            m = k // 2
            for n in range(max(moore, 2*m+1), max_n+1):
                if check_deadline():
                    break
                shift_sets = []
                if m <= n//2:
                    shift_sets.append(list(range(1, m+1)))
                if m == 1 and n > 2:
                    shift_sets.append([1])
                if m == 2 and n > 4:
                    shift_sets.append([1, 2])
                    shift_sets.append([1, 3])
                    shift_sets.append([2, 3])
                if m == 3 and n > 6:
                    shift_sets.append([1, 2, 3])
                    shift_sets.append([1, 2, 4])
                    shift_sets.append([1, 3, 4])
                    shift_sets.append([2, 3, 4])
                
                for shifts in shift_sets:
                    valid_shifts = True
                    for s in shifts:
                        if 2 * s % n == 0:
                            valid_shifts = False
                            break
                    if not valid_shifts:
                        continue
                    if len(set(s % n for s in shifts)) != len(shifts):
                        continue
                    if 0 in shifts:
                        continue
                    
                    edges = []
                    for i in range(n):
                        for s in shifts:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    results.append((n, edges))
        else:
            pass
        return results

    # --- Family 3: Random regular graph search ---
    def random_regular_search():
        if g % 2 == 1:
            r = (g-1)//2
            moore = 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g//2
            moore = k * sum((k-1)**i for i in range(r))
        
        results = []
        for n in range(max(moore, k+1), moore + 200):
            if check_deadline():
                break
            if n * k % 2 != 0:
                continue
            if k >= n:
                continue
            result = random_regular_graph(n, k, rng, max_iters=500)
            if result is not None:
                results.append(result)
        return results

    # --- Family 4: Voltage graph construction over cyclic group ---
    def voltage_graph_family():
        """
        Construct a k-regular graph of girth at least g using voltage graph technique.
        The base graph is a cycle of length L, with k voltages assigned to each edge.
        The lift has n = L * |G| vertices where G = Z_p for some prime p.
        """
        results = []
        
        # For k=5, g=9: try L=9, p=13 (n=117) or similar
        # The voltage assignments must be chosen so that no closed walk of length < g
        # in the base graph has trivial total voltage.
        
        # Try small primes p and cycle lengths L
        for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if check_deadline():
                break
            for L in range(g, g + 10):
                if check_deadline():
                    break
                n = L * p
                if n < 100:  # skip too small
                    continue
                if n > 2000:  # skip too large for now
                    continue
                
                # We need a k-regular voltage graph on a cycle of length L
                # with p vertices in the fiber. The base graph must be k-regular
                # in some sense, but for a cycle the base degree is 2, so we need
                # multiple edges or a different base.
                
                # Alternative: use a complete graph K_k as base with voltage assignments
                # This gives a k-regular graph with n = k * p vertices
                n = k * p
                if n < 100 or n > 2000:
                    continue
                
                # Base graph: K_k (complete graph on k vertices)
                # Each edge (i,j) in the base gets a voltage v_ij in Z_p
                # The lift has vertices (i, t) where i in 0..k-1, t in Z_p
                # Edge from (i, t) to (j, t + v_ij)
                
                # For the graph to be simple, we need v_ij != v_ji and no two edges
                # between the same pair of base vertices have the same voltage offset.
                
                # Let's assign voltages such that the product of voltages around
                # any short cycle is non-trivial.
                
                # Use a systematic assignment: v_ij = (i * j) % p for i != j
                # Check if this works
                edges = []
                adj = [[] for _ in range(n)]
                for i in range(k):
                    for j in range(i+1, k):
                        v_ij = (i * j + i + j) % p
                        v_ji = (-v_ij) % p  # antisymmetric
                        for t in range(p):
                            u1 = i * p + t
                            u2 = j * p + ((t + v_ij) % p)
                            if u1 != u2:
                                edges.append((u1, u2))
                                adj[u1].append(u2)
                                adj[u2].append(u1)
                
                # Check if it's k-regular
                valid = True
                for u in range(n):
                    if len(adj[u]) != k:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check for multi-edges
                edge_set = set()
                has_multi = False
                for u, v in edges:
                    key = (min(u,v), max(u,v))
                    if key in edge_set:
                        has_multi = True
                        break
                    edge_set.add(key)
                if has_multi:
                    continue
                
                results.append((n, edges))
        return results

    # --- Run all families ---
    try:
        n, e = luw_family()
        consider(n, e)
    except Exception as ex:
        pass

    for d_try in range(max(3, g-6), g+2):
        if check_deadline():
            break
        try:
            n, e = luw_family(d=d_try)
            consider(n, e)
        except Exception as ex:
            pass

    try:
        inc_results = incidence_family()
        for n, e in inc_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    try:
        circ_results = circulant_family()
        for n, e in circ_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    try:
        volt_results = voltage_graph_family()
        for n, e in volt_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    try:
        rand_results = random_regular_search()
        for n, e in rand_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    if check_deadline():
        if best is not None:
            return best
        n, e = luw_family()
        return n, e

    if best is not None:
        return best

    n, e = luw_family()
    return n, e

# EVOLVE-BLOCK-END
