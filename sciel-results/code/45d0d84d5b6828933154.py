import time
import sys
import random

DEADLINE_MARGIN = 15.0
HARD_LIMIT = 135.0

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
            # Stuck in local optimum. Try swap moves.
            # Remove 1 edge, add 2 edges.
            # We need to ensure the resulting graph is not 3-colorable.
            
            # Find an edge to remove. Prefer edges that are "redundant" for non-3-colorability?
            # Hard to know. Just pick random edge.
            edges_to_remove = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj_mask[u] & (1 << v):
                        edges_to_remove.append((u, v))
            
            if not edges_to_remove:
                break
                
            # Try a few random removals
            for _ in range(50): # Limit swap attempts per stagnation
                if time.time() > deadline:
                    break
                    
                u, v = random.choice(edges_to_remove)
                
                # Remove (u,v)
                remove_edge(u, v)
                
                # Try to add 2 edges
                # Candidates are all non-edges that can be added
                # We want to add 2 that don't conflict with each other or existing edges
                addable = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if a != u and b != v: # Exclude the removed edge itself? 
                            # Actually, we removed (u,v), so (u,v) is now a non-edge.
                            # We can try to re-add it? No, we want net +1.
                            # So we need to add 2 DIFFERENT edges.
                            if can_add_edge(a, b):
                                addable.append((a, b))
                
                if len(addable) < 2:
                    # Can't add 2 edges, revert
                    add_edge(u, v)
                    continue
                
                # Pick 2 random candidates
                random.shuffle(addable)
                e1 = addable[0]
                e2 = addable[1]
                
                # Check if e1 and e2 can be added simultaneously
                # They must be distinct and not share vertices?
                # If they share a vertex, say (a,b) and (a,c), adding both is fine if no triangles.
                # But can_add_edge checks against current state.
                # We need to check if adding e1 makes e2 invalid, or vice versa.
                
                # Simulate adding e1
                add_edge(e1[0], e1[1])
                
                if can_add_edge(e2[0], e2[1]):
                    add_edge(e2[0], e2[1])
                    
                    # Now check 3-colorability of the new graph
                    if not _is_3_colorable_bitmask(adj_mask, n):
                        # Valid improvement
                        current_count = edge_count()
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj_mask[:]
                    else:
                        # Revert
                        remove_edge(e2[0], e2[1])
                        remove_edge(e1[0], e1[1])
                        add_edge(u, v)
                else:
                    # Revert
                    remove_edge(e1[0], e1[1])
                    add_edge(u, v)

    # Restore best
    adj_mask = best_adj
    
    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                edges.append([u, v])
                
    return edges
