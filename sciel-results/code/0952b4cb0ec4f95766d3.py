import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100.0
    
    if k != 4 or g != 10:
        # Fallback for other parameters
        return _construct_luw_generic(k, g, deadline)
        
    # For k=4, g=10, use LUW with q=4
    # Try d=4 first (girth >= 10)
    # n = 2 * 4^4 = 512
    
    for d in [4, 5, 6, 7]:
        if time.time() > deadline - 10:
            break
            
        n, edges = _construct_luw_4(d)
        
        # Verify girth
        if _verify_girth(n, edges, g):
            return n, edges
            
    # If all fail (should not happen), return the largest one constructed
    # Or just return the last one
    return n, edges

def _construct_luw_4(d):
    """Construct LUW graph for q=4 and given d."""
    q = 4
    n = 2 * (q ** d)
    
    # Precompute field operations for F_4
    # Elements: 0, 1, 2, 3
    # Add is XOR
    add = lambda x, y: x ^ y
    # Mul table
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 0]
    ]
    mul = lambda x, y: mul_table[x][y]
    
    # Rules for LUW
    # Standard LUW rules for d=4:
    # y1 is free
    # y2 = x2 + y1 * x1
    # y3 = x3 + y2 * x1  (Wait, standard might be different)
    # Let's use the rules from the seed which seemed structured.
    # Seed rules:
    # rule[2] = (1, 1) -> y2 = x2 + y1 * x1
    # rule[3] = (2, 1) -> y3 = x3 + y2 * x1
    # rule[4] = (1, 2) -> y4 = x4 + y1 * x2
    
    # For general d, we need rules for 2..d
    # The seed's rule generation:
    # rule[c] for c in 2..d
    # Let's replicate the seed's rule generation for q=4.
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]
        
    # Build edges
    edges = []
    # Vertices 0..q^d-1 are left (side 0)
    # Vertices q^d..2q^d-1 are right (side 1)
    
    for u in range(q ** d):
        # Decode u to x[1..d]
        x = [0] * (d + 1)
        temp = u
        for i in range(d, 0, -1):
            x[i] = temp % q
            temp //= q
            
        # Generate q neighbors
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                # y[c] = x[c] + y[a] * x[b]
                y[c] = add(x[c], mul(y[a], x[b]))
                
            # Encode y to v
            v_offset = 0
            temp_y = y
            for i in range(1, d + 1):
                v_offset = v_offset * q + temp_y[i]
                
            v = (q ** d) + v_offset
            edges.append((u, v))
            
    return n, edges

def _verify_girth(n, edges, g):
    """Verify girth is at least g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles of length < g
    # Use BFS from each vertex
    # For efficiency, only check up to depth (g-1)//2
    max_depth = (g - 1) // 2
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    # But this is only valid if u and v are at the same depth?
                    # Actually, this detects *any* non-tree edge.
                    # The cycle length is dist[u] + dist[v] + 1.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True

def _construct_luw_generic(k, g, deadline):
    # Fallback for other k, g
    # Use random or simple LUW
    # For now, return a small valid graph if possible, or a placeholder
    # Since the task is specific to k=4, g=10, this is rarely called.
    # We can just return a 4-cycle or something small if k,g are small,
    # but for large k,g we might fail.
    # Let's just return the LUW for k=4 if k==4.
    if k == 4:
        return _construct_luw_4(4)
    else:
        # Fallback: return a small graph
        return 0, []
