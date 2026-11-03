# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from collections import deque
from itertools import product

def construct(k, g):
    deadline = time.time() + 140.0
    
    # Guaranteed-valid fallback: Lazebnik-Ustimenko-Woldar CD graph
    def _lw_fallback():
        q = k
        d = max(3, g - 5)
        
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
    
    # Fast girth certification using non-backtracking walks
    def _girth(adj, n):
        INF = 10**9
        for L in range(3, 2*g+1):
            for s in range(n):
                dist = {}
                q = deque()
                for nb in adj[s]:
                    q.append((s, nb))
                    dist[(s, nb)] = 1
                while q:
                    prev, cur = q.popleft()
                    dl = dist[(prev, cur)]
                    if dl == L:
                        break
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        key = (cur, nb)
                        if key in dist:
                            cyc = dl + 1 - dist[key]
                            if 3 <= cyc < L:
                                return cyc
                        else:
                            dist[key] = dl + 1
                            q.append(key)
        return 2*g
    
    # Bipartite double cover construction
    def _bipartite_double_cover(cycle_lengths):
        """
        Given a list of cycle lengths (each >= g), construct a 4-regular graph
        via the bipartite double cover of the disjoint union of these cycles.
        
        The base graph H is a 2-regular graph (disjoint union of cycles) on m vertices.
        The bipartite double cover B(H) has 2m vertices: for each vertex v in H,
        we create (v, 0) and (v, 1). For each edge (u, v) in H, we add edges
        (u,0)-(v,1) and (u,1)-(v,0) in B(H).
        
        B(H) is 4-regular if H is 2-regular.
        The girth of B(H) relates to the girth of H.
        """
        m = sum(cycle_lengths)
        if m == 0:
            return 0, []
        
        # Build the base 2-regular graph H as disjoint union of cycles
        edges_h = []
        base_offset = 0
        for cl in cycle_lengths:
            if cl < 3:
                continue
            for i in range(cl):
                u = base_offset + i
                v = base_offset + (i + 1) % cl
                edges_h.append((u, v))
            base_offset += cl
        
        # Build bipartite double cover
        n = 2 * m
        edges = []
        
        # For each edge (u, v) in H, add edges (u,0)-(v,1) and (u,1)-(v,0)
        for u, v in edges_h:
            u0, u1 = 2*u, 2*u+1
            v0, v1 = 2*v, 2*v+1
            edges.append((u0, v1))
            edges.append((u1, v0))
        
        # Verify it's 4-regular and check girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return -1, []
        
        # Check girth
        girth = _girth(adj, n)
        if girth < g:
            return -1, []
        
        return n, edges
    
    # Search over cycle length partitions
    def _search_cycle_partitions():
        """
        Search for cycle length partitions that give good girth-to-size ratio.
        For the bipartite double cover, we need cycles in H of length >= g/2 
        (roughly), but the actual girth constraint is more complex.
        
        Let's think: if H has a cycle of length c, then B(H) has a cycle of length 2c.
        So we need all cycles in H to have length >= g/2.
        
        But there can be shorter cycles in B(H) that don't correspond to cycles in H.
        We need to verify with the girth checker.
        
        Strategy: try small sets of cycles with lengths >= ceil(g/2).
        """
        min_cycle_len = max(3, (g + 1) // 2)
        
        # Try single cycle first
        for cl in range(min_cycle_len, 200):
            if time.time() > deadline - 5:
                break
            n, edges = _bipartite_double_cover([cl])
            if n > 0:
                return n, edges
        
        # Try two cycles
        for cl1 in range(min_cycle_len, min_cycle_len + 50):
            if time.time() > deadline - 5:
                break
            for cl2 in range(min_cycle_len, min_cycle_len + 50):
                if time.time() > deadline - 5:
                    break
                n, edges = _bipartite_double_cover([cl1, cl2])
                if n > 0:
                    return n, edges
        
        # Try three cycles
        for cl1 in range(min_cycle_len, min_cycle_len + 30):
            if time.time() > deadline - 5:
                break
            for cl2 in range(min_cycle_len, min_cycle_len + 30):
                if time.time() > deadline - 5:
                    break
                for cl3 in range(min_cycle_len, min_cycle_len + 30):
                    if time.time() > deadline - 5:
                        break
                    n, edges = _bipartite_double_cover([cl1, cl2, cl3])
                    if n > 0:
                        return n, edges
        
        return -1, []
    
    # Driver: try multiple approaches, return best
    best_n = float('inf')
    best_edges = []
    
    # Approach 1: Bipartite double cover search
    try:
        n, edges = _search_cycle_partitions()
        if n > 0 and n < best_n:
            best_n = n
            best_edges = edges
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        if best_n < float('inf'):
            return best_n, best_edges
        return _lw_fallback()
    
    # Approach 2: LUW fallback
    try:
        n, edges = _lw_fallback()
        if n < best_n:
            best_n = n
            best_edges = edges
    except Exception:
        pass
    
    if best_n < float('inf'):
        return best_n, best_edges
    
    return _lw_fallback()
# EVOLVE-BLOCK-END
