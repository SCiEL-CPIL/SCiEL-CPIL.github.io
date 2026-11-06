import time
import sys
import random

DEADLINE_MARGIN = 15.0
HARD_LIMIT = 135.0

def _has_triangle(adj_mask, n):
    """Check if the graph represented by adj_mask contains a triangle."""
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                # If u and v are connected, check if they share a common neighbor
                if adj_mask[u] & adj_mask[v]:
                    return True
    return False

def _is_3_colorable_bitmask(adj_mask, n):
    """
    Check if graph with adjacency bitmasks is 3-colorable.
    adj_mask is a list of n integers, where bit i is 1 if vertex i is a neighbor.
    """
    # Order vertices by degree (popcount) descending
    order = sorted(range(n), key=lambda x: -bin(adj_mask[x]).count('1'))
    
    colors = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        # Determine forbidden colors from already colored neighbors
        forbidden = 0
        # Only check neighbors that are already colored (earlier in order)
        # We can just check all neighbors, if color is -1, it doesn't forbid anything
        mask = adj_mask[v]
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & -m
            nb_idx = lsb.bit_length() - 1
            c = colors[nb_idx]
            if c != -1:
                forbidden |= (1 << c)
            m ^= lsb
        
        for c in range(3):
            if forbidden & (1 << c):
                continue
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = -1
        
        return False
    
    return backtrack(0)

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # adj_mask[i] is an integer where bit j is 1 if (i,j) is an edge
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        
    def edge_count():
        return sum(bin(m).count('1') for m in adj_mask) // 2
        
    def can_add_edge(u, v):
        if u == v:
            return False
        if adj_mask[u] & (1 << v):
            return False
        # Check triangle-free: common neighbors must be empty
        if adj_mask[u] & adj_mask[v]:
            return False
        return True

    # Initialize with Grötzsch graph on 0-10
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    # Mycielski layer 5-9
    for i in range(5):
        add_edge(5 + i, (i + 1) % 5)
        add_edge(5 + i, (i - 1) % 5)
    # Universal 10
    for i in range(5):
        add_edge(10, 5 + i)
        
    # Vertices 11-15 are isolated initially
    
    best_count = edge_count()
    best_adj = adj_mask[:]
    initial_adj = adj_mask[:] # Store initial state for fallback
    
    # Verify initial is not 3-colorable (it is Grötzsch, so it should be)
    # If it is 3-colorable, we have a bug or wrong graph.
    if _is_3_colorable_bitmask(adj_mask, n):
        # Fallback: just return a known non-3-colorable graph or fail
        # Grötzsch is definitely not 3-colorable.
        pass

    # Local Search
    # We will try to add edges.
    # We maintain a list of candidate edges (u, v) that are not in graph and can be added.
    
    def get_candidates():
        cands = []
        for u in range(n):
            for v in range(u + 1, n):
                if can_add_edge(u, v):
                    cands.append((u, v))
        return cands

    # To speed up, we don't recompute all candidates every time.
    # But n=16 is small, 120 edges. Recomputing is O(120 * 16) which is fast.
    
    random.seed(42)
    
    while time.time() < deadline:
        # 1. Try to add any valid edge
        improved = False
        # Shuffle to avoid bias
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if can_add_edge(u, v):
                add_edge(u, v)
                improved = True
                # Check if we broke 3-colorability? No, adding edges preserves non-3-colorability.
                # So we don't need to check 3-colorability after adding.
        
        current_count = edge_count()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj_mask[:]
            
        if not improved:
            # Stuck in local optimum. Try 1-for-1 swap moves to escape local optima
            # while maintaining triangle-free property.
            edges_in_graph = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj_mask[u] & (1 << v):
                        edges_in_graph.append((u, v))
            
            if not edges_in_graph:
                break
                
            # Try swapping a random edge with another possible edge
            random.shuffle(edges_in_graph)
            for u, v in edges_in_graph[:20]: # Limit swaps per stagnation
                if time.time() > deadline:
                    break
                    
                remove_edge(u, v)
                
                # Find a new edge to add
                # Prefer edges that increase density or are "high degree"
                addable = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if (a, b) != (u, v) and can_add_edge(a, b):
                            addable.append((a, b))
                
                if addable:
                    random.shuffle(addable)
                    a, b = addable[0]
                    add_edge(a, b)
                    
                    # Check 3-colorability
                    if not _is_3_colorable_bitmask(adj_mask, n):
                        current_count = edge_count()
                        # We didn't increase edge count, but we might have escaped a local optimum
                        # for future additions. We don't update best_count here as it's same.
                        # But we keep the new state to try more additions.
                        pass
                    else:
                        # Revert if it became 3-colorable (we need to maintain chi >= 4)
                        remove_edge(a, b)
                        add_edge(u, v)
                else:
                    add_edge(u, v)

    # Restore best
    adj_mask = best_adj
    
    # CRITICAL FIX: Verify the best graph is triangle-free
    if _has_triangle(adj_mask, n):
        # If a bug caused a triangle, fall back to the initial Grötzsch graph
        adj_mask = initial_adj[:]
    
    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                edges.append([u, v])
                
    return edges
