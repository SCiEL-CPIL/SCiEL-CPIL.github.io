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

def _max_independent_set_size(adj, vertices, budget_nodes=500000):
    """Find a large independent set among given vertices using greedy + local search."""
    best = []
    best_size = 0
    n = len(vertices)
    
    # Greedy: try multiple random orderings
    for trial in range(200):
        if not _time_ok():
            break
        order = list(vertices)
        random.shuffle(order)
        chosen = []
        for v in order:
            if all(v not in adj[c] for c in chosen):
                chosen.append(v)
        if len(chosen) > best_size:
            best = chosen
            best_size = len(chosen)
    
    # Local improvement
    changed = True
    while changed and _time_ok():
        changed = False
        new_set = list(best)
        for v in vertices:
            if v in new_set:
                continue
            if all(v not in adj[c] for c in new_set):
                new_set.append(v)
                best = new_set
                changed = True
                break
    
    # Try to improve by swapping
    for trial in range(100):
        if not _time_ok():
            break
        candidate = list(best)
        for i in range(len(candidate)):
            removed = candidate[i]
            candidate_no_i = [c for c in candidate if c != removed]
            for v in vertices:
                if v == removed or v in candidate_no_i:
                    continue
                if all(v not in adj[c] for c in candidate_no_i):
                    new_candidate = candidate_no_i + [v]
                    if len(new_candidate) > len(best):
                        best = new_candidate
                        changed = True
                        break
            if changed:
                break
        if not changed:
            break
    
    return best

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

def _can_add_edge(adj, u, v, n):
    """Check if adding edge (u,v) preserves triangle-free property."""
    if v in adj[u]:
        return False
    # Check no common neighbor (would form triangle)
    for w in adj[u]:
        if w in adj[v]:
            return False
    return True

def _local_edge_search(adj, n, time_budget=30):
    """Try to add more edges while preserving triangle-free and non-3-colorable."""
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        # Try all non-edges
        for u in range(n):
            if time.time() - start_time > time_budget:
                return
            for v in range(u + 1, n):
                if v in adj[u]:
                    continue
                if not _can_add_edge(adj, u, v, n):
                    continue
                # Try adding this edge
                adj[u].add(v)
                adj[v].add(u)
                
                # Check if still not 3-colorable
                if not _is_3_colorable(adj, n):
                    improved = True
                    break
                else:
                    # Revert
                    adj[u].discard(v)
                    adj[v].discard(u)
            if improved:
                break
    
    return adj

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 115  # safety margin: evaluator kills at 150s
    
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
    
    # Now add vertices 11-15 greedily
    for v in range(11, 16):
        if not _time_ok():
            break
        
        # Find a large independent set among existing vertices (0-10)
        existing = list(range(11))
        indep = _max_independent_set_size(adj, existing)
        
        # Add edges from v to all vertices in the independent set
        for u in indep:
            add_edge(v, u)
    
    # Local edge search: try to add more edges
    remaining_time = _DEADLINE - time.time()
    if remaining_time > 10:
        search_budget = min(remaining_time - 5, 40)
        adj = _local_edge_search(adj, n, time_budget=search_budget)
    
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
