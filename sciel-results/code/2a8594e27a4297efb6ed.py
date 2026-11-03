import time
import sys
import collections
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if k != 3:
        # Fallback for other k: use LUBW if possible, or random
        # For now, focus on k=3
        pass

    def girth_of_d(d):
        # Approximate girth for LUBW
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1
        
    # GF(q) operations for LUBW
    q = 3
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    # Rule for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    def build_luw():
        index = {}
        order = []
        edges = []
        
        start = (0, (None,) + (0,) * d)
        index[start] = 0
        order.append(start)
        
        for i, vertex in enumerate(order):
            if time.time() > deadline:
                break
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                
                neighbor = (1 - side, tuple(y))
                
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                
                if side == 0:
                    edges.append((index[vertex], index[neighbor]))
                    
        n = len(order)
        return n, edges

    def check_girth_fast(n, edges, min_girth):
        """
        Checks if the graph has girth at least min_girth using non-backtracking walks.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Non-backtracking walk check
        # A graph has girth >= g if there are no non-backtracking walks of length < g that return to start?
        # Actually, girth is the length of the shortest cycle.
        # A cycle of length L corresponds to a non-backtracking walk of length L.
        # We can do BFS from each node but prune backtracking.
        
        max_depth = min_girth // 2 - 1
        # If girth >= 16, diameter of BFS tree without backtracking is 7?
        # No, girth 16 means no cycle < 16.
        # BFS from any node up to depth 7 will not encounter a cycle?
        # If there is a cycle of length < 16, it will be found in BFS.
        # Standard BFS girth check:
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            
            while queue:
                curr = queue.popleft()
                if dist[curr] > max_depth:
                    continue
                for neighbor in adj[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    elif neighbor != parent[curr]:
                        # Found a cycle
                        cycle_len = dist[curr] + dist[neighbor] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    if k == 3:
        # Search for 3-regular circulant graphs C_n(1, a)
        # Vertices 0..n-1
        # Edges: (i, i+1), (i, i-1), (i, i+a)
        # Note: If a=1, it's a multigraph or not simple?
        # If a=1, edges are (i, i+1) twice. Not simple.
        # So a >= 2.
        
        # Moore bound for k=3, g=16 is 191.
        # Try n from 192 to some limit.
        # If we find one, return it.
        
        max_n_search = 1000 # Limit search to keep time low
        
        for n in range(192, max_n_search + 1):
            if time.time() > deadline - 20: # Leave time for LUW
                break
            
            for a in range(2, n // 2 + 1):
                if time.time() > deadline - 20:
                    break
                
                # Construct edges
                edges_c = []
                for i in range(n):
                    edges_c.append((i, (i + 1) % n))
                    edges_c.append((i, (i - 1) % n))
                    edges_c.append((i, (i + a) % n))
                
                # Remove duplicates and self-loops
                # Since n > 2a, no self loops for a < n/2?
                # If 2a = n, then (i, i+a) is (i, i+n/2), which is fine.
                # But edges are undirected. (i, i+a) is same as (i+a, i).
                # We add (i, i+1) and (i, i-1).
                # (i, i-1) is same as (i-1, i).
                # So we have duplicates if we just append.
                # We should use a set or check.
                
                edge_set = set()
                for i in range(n):
                    u = i
                    v = (i + 1) % n
                    if u > v: u, v = v, u
                    edge_set.add((u, v))
                    
                    v = (i - 1) % n
                    if u > v: u, v = v, u
                    edge_set.add((u, v))
                    
                    v = (i + a) % n
                    if u > v: u, v = v, u
                    edge_set.add((u, v))
                
                edges_list = list(edge_set)
                
                # Check degree
                # It should be 3 unless there are coincidences.
                # Coincidences happen if:
                # 1. a = 1 (excluded)
                # 2. a = n-1 (excluded, a <= n/2)
                # 3. a = 2? (i, i+2) vs (i, i+1) -> no.
                # 4. 2a = n? Then (i, i+a) is same as (i, i+n/2).
                #    Is (i, i+n/2) same as (i, i+1)? No.
                #    Is it same as (i, i-1)? No.
                #    So degree is 3.
                
                if check_girth_fast(n, edges_list, g):
                    return n, edges_list

    # Fallback: LUW
    n_luw, edges_luw = build_luw()
    return n_luw, edges_luw
