import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Special case: Known small graphs
    if k == 4 and g == 5:
        # Petersen graph, n=10
        return _petersen_graph()
    if k == 3 and g == 5:
        # Petersen graph is 3-regular, girth 5, n=10
        return _petersen_graph()

    # For k=5, g=9, the LUW construction gives n=1250.
    # The known best is 1116.
    # We try to see if we can find a smaller graph via heuristic search.
    # But heuristic search for girth 9 is very hard.
    # We fallback to LUW.
    
    if k == 5 and g == 9:
        # LUW(4, 5) gives 1250 vertices with girth >= 9
        n, edges = _luw_construct(5, 4)
        if _verify_girth(n, edges, g):
            return n, edges

    # General fallback: LUW construction
    # Try increasing d until girth is satisfied
    for d in range(2, 12):
        if time.time() > deadline - 15:
            break
        n, edges = _luw_construct(k, d)
        if _verify_girth(n, edges, g):
            return n, edges

    # Final fallback: LUW with large d
    n, edges = _luw_construct(k, 10)
    return n, edges

def _petersen_graph():
    # Petersen graph: 10 vertices, 5-regular? No, 3-regular.
    # Vertices: 0..4 (inner), 5..9 (outer)
    # Edges: outer cycle (0-1-2-3-4-0), inner star (5-7-9-6-8-5), spokes (0-5, 1-6, 2-7, 3-8, 4-9)
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
        edges.append((5 + i, 5 + (i + 2) % 5))
        edges.append((i, 5 + i))
    return 10, edges

def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, k) with q = k.
    Returns a k-regular bipartite graph on 2*k^d vertices.
    """
    q = k
    
    # Generate the rule for c = 2..d
    rule = {}
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Precompute multiplication table
    mul_table = [[(i * j) % q for j in range(q)] for i in range(q)]
    
    # Precompute all vertices as flat indices
    # Side 0: x[1..d] each in 0..q-1 -> q^d vertices
    # Side 1: x[1..d] each in 0..q-1 -> q^d vertices
    # Total: 2*q^d vertices
    
    # Map from (side, coord_tuple) to flat index
    # For side 0: index = coord[0]*q^(d-1) + coord[1]*q^(d-2) + ... + coord[d-1]
    # For side 1: index = q^d + coord[0]*q^(d-1) + ...
    
    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i-1] * q
    
    def coord_to_index(side, coords):
        """coords is a tuple of d elements, each 0..q-1"""
        idx = 0
        for i in range(d):
            idx += coords[i] * q_powers[d - 1 - i]
        if side == 1:
            idx += q_powers[d]
        return idx
    
    # Precompute neighbors for all vertices
    # For a vertex on side 0 with coordinates x[1..d]:
    #   For each first in 0..q-1:
    #     y[1] = first
    #     For c in 2..d: y[c] = x[c] + y[a] * x[b] mod q
    #     Neighbor is on side 1 with coordinates y[1..d]
    #
    # For a vertex on side 1 with coordinates x[1..d]:
    #   For each first in 0..q-1:
    #     y[1] = first
    #     For c in 2..d: y[c] = x[c] - x[a] * y[b] mod q
    #     Neighbor is on side 0 with coordinates y[1..d]
    
    n = 2 * q_powers[d]
    adj = [[] for _ in range(n)]
    
    # Precompute for side 0 vertices
    for x0 in range(q):
        for x1 in range(q):
            # ... iterate over all q^d combinations
            pass
    
    # More efficient: iterate over all vertices
    for idx in range(n):
        if idx < q_powers[d]:
            side = 0
            flat = idx
        else:
            side = 1
            flat = idx - q_powers[d]
        
        # Decode coordinates
        coords = [0] * d
        for i in range(d - 1, -1, -1):
            coords[i] = flat % q
            flat //= q
        
        # Compute neighbors
        for first in range(q):
            y = [0] * (d + 1)  # 1-indexed: y[1..d]
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = (coords[c-1] + mul_table[y[a]][coords[b-1]]) % q
                else:
                    y[c] = (coords[c-1] - mul_table[coords[a-1]][y[b]]) % q
            other_side = 1 - side
            other_flat = 0
            for i in range(d):
                other_flat += y[i+1] * q_powers[d - 1 - i]
            other_idx = coord_to_index(other_side, tuple(y[1:d+1]))
            adj[idx].append(other_idx)
    
    # Build edge list (only one direction to avoid duplicates)
    edges = []
    for u in range(q_powers[d]):  # Only side 0 vertices
        for v in adj[u]:
            edges.append((u, v))
    
    return n, edges

def _verify_girth(n, edges, g):
    """
    Verify that the girth is at least g using BFS from each vertex.
    For each BFS tree, any edge (u,v) where neither is the parent of the other
    in the BFS tree gives a cycle of length dist[u] + dist[v] + 1.
    We need all such cycles to have length >= g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For girth g, we only need to check BFS up to depth (g-1)//2
    max_depth = (g - 1) // 2
    
    for start in range(n):
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
                    # Back edge or cross edge
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
