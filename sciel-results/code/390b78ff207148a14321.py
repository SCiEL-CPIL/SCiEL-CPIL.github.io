"""Construct a 4-regular graph of girth 10 (the (4,10)-cage)."""
import numpy as np
from collections import deque
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g.
    
    For k=4, g=10, attempts to find a small component of the LW graph.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    if k == 4 and g == 10:
        return _construct_4_10_cage()
    else:
        return _construct_general(k, g)

def _construct_4_10_cage():
    """Construct the (4,10)-cage by finding a small valid component of an LW graph."""
    q = 4
    d = 5
    
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def gf_add(x, y):
        return x ^ y
    
    def gf_mul(x, y):
        return mul_table[x][y]
    
    rule = {
        2: (1, 1),
        3: (2, 1),
        4: (1, 2),
        5: (1, 3)
    }
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_add(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, tuple([0] * (d + 1)))
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
    
    n_total = len(order)
    
    # If the total graph is small enough, return it
    if n_total <= 320:
        return n_total, edges
    
    # Find connected components
    # Build adjacency list
    adj = {i: [] for i in range(n_total)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    visited = [False] * n_total
    components = []
    
    for i in range(n_total):
        if not visited[i]:
            # BFS
            q_comp = deque([i])
            visited[i] = True
            comp_nodes = []
            while q_comp:
                node = q_comp.popleft()
                comp_nodes.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        q_comp.append(nei)
            components.append(comp_nodes)
            
    # Check each component for validity (4-regular, girth >= 10)
    best_component = None
    best_size = n_total + 1
    
    for comp in components:
        size = len(comp)
        if size >= best_size:
            continue
            
        # Check regularity
        is_regular = True
        for node in comp:
            if len(adj[node]) != 4:
                is_regular = False
                break
        if not is_regular:
            continue
            
        # Check girth
        # Use BFS from each node to find shortest cycle
        # Since graph is small (hopefully), this is feasible
        girth = float('inf')
        # Optimization: only check if size is significantly smaller
        # We need girth >= 10
        # If we find a cycle < 10, reject.
        valid_girth = True
        # To speed up, we can stop BFS at depth 4 (radius 4) for girth 10?
        # Cycle length 2L. If L < 5, cycle < 10.
        # So if we find a path of length 4 between two adjacent nodes?
        # Simpler: BFS from each node, stop at depth 4. If we revisit a node at depth < 5, cycle exists.
        
        for start_node in comp:
            # BFS
            dist = {start_node: 0}
            prev = {start_node: None}
            q_bfs = deque([start_node])
            found_short_cycle = False
            while q_bfs:
                node = q_bfs.popleft()
                if dist[node] >= 4: # We care about cycles < 10, so radius < 5
                    continue
                for nei in adj[node]:
                    if nei not in dist:
                        dist[nei] = dist[node] + 1
                        prev[nei] = node
                        q_bfs.append(nei)
                    else:
                        # Check if this edge closes a cycle
                        # Cycle length = dist[node] + dist[nei] + 1
                        # But we need to ensure they are not just backtracking
                        if prev[node] != nei and prev[nei] != node:
                            cycle_len = dist[node] + dist[nei] + 1
                            if cycle_len < 10:
                                found_short_cycle = True
                                break
            if found_short_cycle:
                valid_girth = False
                break
                
        if valid_girth:
            best_component = comp
            best_size = size
            
    if best_component is not None:
        # Remap indices to 0..n-1
        comp_set = set(best_component)
        mapping = {node: i for i, node in enumerate(best_component)}
        new_edges = []
        for u, v in edges:
            if u in mapping and v in mapping:
                new_edges.append((mapping[u], mapping[v]))
        return len(best_component), new_edges
    else:
        # Fallback to whole graph if no valid small component found
        return n_total, edges

def _construct_general(k, g):
    """General construction for other parameters."""
    n = 2 * k * (g - 1)
    edges = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            edges.append((i, (i + j) % n))
    return n, edges
# EVOLVE-BLOCK-END
