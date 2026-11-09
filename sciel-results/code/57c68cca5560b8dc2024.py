"""Enhanced: Grötzsch core + greedy triangle-free extension with 3-color verification."""

import time
import random

_DEADLINE = None

def _time_ok():
    global _DEADLINE
    if _DEADLINE is None:
        return True
    return time.time() < _DEADLINE

def _is_independent_set(neighbors, adj):
    """Check if set of vertices forms an independent set."""
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if neighbors[j] in adj[neighbors[i]]:
                return False
    return True

def _max_independent_set_exact(adj, vertices):
    """Find a maximum independent set exactly using backtracking."""
    n = len(vertices)
    if n == 0:
        return []
    
    # Map vertices to indices 0..n-1 for efficiency
    idx_map = {v: i for i, v in enumerate(vertices)}
    # Build local adjacency matrix
    local_adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if vertices[j] in adj[vertices[i]]:
                local_adj[i][j] = True
                local_adj[j][i] = True
                
    best_set = []
    best_size = 0
    
    def backtrack(current_set, available):
        nonlocal best_set, best_size
        if not available:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
            
        # Pruning: if len(current_set) + len(available) <= best_size, return
        if len(current_set) + len(available) <= best_size:
            return
            
        # Heuristic: Pick vertex with highest degree in available
        v_idx = -1
        max_deg = -1
        for i in available:
            deg = sum(1 for j in available if local_adj[i][j])
            if deg > max_deg:
                max_deg = deg
                v_idx = i
                
        if v_idx == -1:
            # All isolated
            if max_deg == 0:
                candidate = current_set + [vertices[i] for i in available]
                if len(candidate) > best_size:
                    best_size = len(candidate)
                    best_set = candidate
                return
            else:
                # Should not happen if available is not empty
                return

        v = v_idx
        # Branch 1: Include v
        new_available = [i for i in available if i != v and not local_adj[v][i]]
        backtrack(current_set + [vertices[v]], new_available)
        
        # Branch 2: Exclude v
        new_available = [i for i in available if i != v]
        backtrack(current_set, new_available)
        
    backtrack([], list(range(n)))
    return best_set

def _is_triangle_free(adj, n):
    """Check if graph is triangle-free."""
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
    return True

def _is_3_colorable(adj, n):
    """Backtracking 3-colorability check. Returns True if 3-colorable."""
    color = [-1] * n
    # Sort vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos_in_order = {v: i for i, v in enumerate(order)}
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Determine forbidden colors from already-colored neighbors
        forbidden = [False, False, False]
        for nb in adj[v]:
            if pos_in_order[nb] < idx:
                forbidden[color[nb]] = True
        
        # Try each color
        for c in range(3):
            if not forbidden[c]:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 120  # safety margin: evaluator kills at 150s
    
    random.seed(42)
    
    # Build Grötzsch graph (Mycielski of C5)
    # Cycle: 0-1-2-3-4-0
    # Vertices 5-9: Mycielski middle layer
    # Vertex 10: Mycielski top
    n = 16
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    # C5
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    
    # Mycielski: for edge (u,v) in C5, add 5+u to v, 5+v to u
    # Standard Mycielski of C5:
    # Vertex 5+i connects to the two neighbors of i in C5
    for i in range(5):
        left = (i - 1) % 5
        right = (i + 1) % 5
        add_edge(5 + i, left)
        add_edge(5 + i, right)
    
    # Vertex 10 connects to 5,6,7,8,9
    for i in range(5):
        add_edge(10, 5 + i)
    
    # Verify: 5*2 + 5 = 15 edges from Mycielski part, 5 from cycle = 20 total
    # This is the Grötzsch graph
    
    # Now add vertices 11-15 greedily
    for v in range(11, 16):
        if not _time_ok():
            break
        
        # Find a large independent set among existing vertices (0..v-1)
        existing = list(range(v))
        indep = _max_independent_set_exact(adj, existing)
        
        # Add edges from v to all vertices in the independent set
        for u in indep:
            add_edge(v, u)
    
    # Collect edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                edges.append([u, v])
    
    m = len(edges)
    
    # Verify triangle-free
    if not _is_triangle_free(adj, n):
        # This shouldn't happen, but fallback to seed
        return [[0,1],[1,2],[2,3],[3,4],[4,0],[0,6],[1,7],[2,8],[3,9],[4,10],[5,6],[5,7],[5,8],[5,9],[5,10]]
    
    # Verify non-3-colorable
    if _is_3_colorable(adj, n):
        # Fallback to just Grötzsch + isolated
        edges = []
        adj2 = [set() for _ in range(n)]
        for i in range(5):
            adj2[i].add((i+1)%5)
            adj2[(i+1)%5].add(i)
        for i in range(5):
            left = (i-1)%5
            right = (i+1)%5
            adj2[5+i].add(left)
            adj2[left].add(5+i)
            adj2[5+i].add(right)
            adj2[right].add(5+i)
        for i in range(5):
            adj2[10].add(5+i)
            adj2[5+i].add(10)
        for u in range(n):
            for v in adj2[u]:
                if v > u:
                    edges.append([u, v])
        return edges
    
    return edges
