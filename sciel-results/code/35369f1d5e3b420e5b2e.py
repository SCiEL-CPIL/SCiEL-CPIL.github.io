"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # LUW construction: try multiple d values, pick smallest valid graph
    # For LUW CD(d,q): girth >= d+5 if d odd, >= d+4 if d even (conservative)
    # We need girth >= g
    
    q = k
    
    # GF(q) arithmetic
    if q == 4:
        mul_table = (
            0, 0, 0, 0,
            0, 1, 2, 3,
            0, 2, 3, 1,
            0, 3, 1, 2
        )
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return mul_table[x * 4 + y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return (x * y) % q
    
    def build_luw(d):
        """Build LUW graph with parameter d, return (n, edges) or None if timeout."""
        # Precompute rules
        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                a, b = 1, c0 - 2
            elif r == 1:
                a, b = c0 - 1, 1
            elif r == 2:
                a, b = c0, 1
            else:
                a, b = 1, c0 + 1
            rule_a[c] = a
            rule_b[c] = b
        
        # BFS from origin
        start = (0, tuple([0] * (d + 1)))
        index = {start: 0}
        order = [start]
        edges = []
        
        head = 0
        while head < len(order):
            if time.time() > deadline - 5:
                return None
            
            side, x = order[head]
            head += 1
            x_idx = index[(side, x)]
            
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
                
                other = (1 - side, tuple(y))
                
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                
                if side == 0:
                    edges.append((x_idx, index[other]))
        
        n = len(order)
        return n, edges
    
    def verify_girth(n, edges, min_girth):
        """Check that no cycle shorter than min_girth exists using BFS from each vertex."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degrees
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check for cycles shorter than min_girth using BFS
        # For each vertex, do BFS up to depth (min_girth-1)//2 and check for back-edges
        max_depth = (min_girth - 1) // 2
        
        for start_v in range(n):
            if time.time() > deadline - 2:
                return False
            
            # BFS with parent tracking to detect short cycles
            dist = [-1] * n
            parent = [-1] * n
            dist[start_v] = 0
            queue = deque([start_v])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and dist[v] <= dist[u]:
                        # Found a cycle: length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        
        return True
    
    # Try d values from minimum upward
    # Minimum d for girth g: need d+5 >= g (odd d) or d+4 >= g (even d)
    # For g=10: d=5 (odd, 5+5=10) works
    
    min_d = max(3, g - 5)
    if min_d % 2 == 0:
        min_d += 1  # prefer odd d for better girth guarantee
    
    best_n = float('inf')
    best_edges = None
    
    # Try d = min_d, min_d+2, min_d+4, ... (odd values)
    # Also try even values if they might give smaller graphs
    for d in range(min_d, min_d + 20):  # Try up to 10 values
        if time.time() > deadline - 10:
            break
        
        result = build_luw(d)
        if result is None:
            continue
        
        n, edges = result
        
        # Verify girth
        if not verify_girth(n, edges, g):
            continue
        
        if n < best_n:
            best_n = n
            best_edges = edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: return the minimum d result even if unverified
    result = build_luw(min_d)
    if result is not None:
        return result
    
    # Last resort: return a trivial invalid result
    return 2, [(0, 1)]
# EVOLVE-BLOCK-END
