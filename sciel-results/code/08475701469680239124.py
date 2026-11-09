import time
import random

def _is_not_3_colorable(adj_masks, n):
    """
    Returns True if the graph is NOT 3-colorable.
    adj_masks: list of int, where bit i is set if edge (v, i) exists.
    """
    colors = [-1] * n
    
    # Order vertices by degree descending for faster pruning
    degrees = [bin(adj_masks[i]).count('1') for i in range(n)]
    order = list(range(n))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    def backtrack(idx):
        if idx == n:
            return False # Successfully colored, so it IS 3-colorable
        
        v = order[idx]
        forbidden = 0
        # Check neighbors
        mask = adj_masks[v]
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if colors[u] != -1:
                forbidden |= (1 << colors[u])
            m ^= lsb
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)

def _greedy_fill(adj_masks, n):
    """
    Greedily add edges to adj_masks to maximize edges while maintaining triangle-freeness.
    Returns the number of edges added.
    """
    added = 0
    # List of potential edges
    # We can just iterate 0..n-1, 1..n-1 etc
    # To be slightly smarter, we can prioritize edges that don't block many others?
    # For now, simple iteration.
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_masks[u] & (1 << v)):
                # Check if adding (u,v) creates a triangle
                # Common neighbors of u and v
                common = adj_masks[u] & adj_masks[v]
                if common == 0:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    added += 1
    return added

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = 0
    best_adj_masks = None
    
    # Initial Clebsch graph
    S = {1, 2, 4, 8, 15}
    init_masks = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if (u ^ v) in S:
                init_masks[u] |= (1 << v)
                init_masks[v] |= (1 << u)
    
    init_edges = sum(bin(m).count('1') for m in init_masks) // 2
    if not _is_not_3_colorable(init_masks, n):
         # Should be true for Clebsch
         pass 
    else:
         best_edges = init_edges
         best_adj_masks = init_masks[:]

    # Branch and Bound
    # We will try to build the graph by deciding edges one by one.
    # To make it effective, we need a good edge ordering.
    # We'll use random restarts with different edge orderings.
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        adj_masks = [0] * n
        current_edges = 0
        
        def bnb(idx):
            nonlocal best_edges, best_adj_masks
            if time.time() > deadline:
                return
            
            if idx == len(all_edges):
                if current_edges > best_edges:
                    if _is_not_3_colorable(adj_masks, n):
                        best_edges = current_edges
                        best_adj_masks = adj_masks[:]
                return
            
            u, v = all_edges[idx]
            
            # Option 1: Include edge (u, v)
            # Check if it creates a triangle
            common = adj_masks[u] & adj_masks[v]
            if common == 0:
                # Include
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
                # Pruning: Upper bound
                # Max possible edges = current_edges + 1 + (len(all_edges) - idx - 1)
                if current_edges + 1 + (len(all_edges) - idx - 1) > best_edges:
                    # Check if already not 3-colorable
                    if _is_not_3_colorable(adj_masks, n):
                        # Greedy fill to get a candidate
                        trial_masks = adj_masks[:]
                        added = _greedy_fill(trial_masks, n)
                        trial_edges = current_edges + 1 + added
                        if trial_edges > best_edges:
                            best_edges = trial_edges
                            best_adj_masks = trial_masks[:]
                        # Prune if even greedy fill can't beat best?
                        # Actually, greedy fill is not optimal, so we can't prune based on it being < best.
                        # But if current_edges + 1 + (len(all_edges) - idx - 1) <= best_edges, we would have pruned above.
                        # We still need to explore further to see if a non-greedy fill is better?
                        # Yes, so we continue.
                    bnb(idx + 1)
                
                # Undo
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
            
            # Option 2: Exclude edge (u, v)
            # Pruning: Upper bound
            if current_edges + (len(all_edges) - idx) > best_edges:
                bnb(idx + 1)

        bnb(0)
        
        # If we found a solution with many edges, we might break early?
        # No, let it run until deadline.
        
    # Convert best_adj_masks to edge list
    edges = []
    if best_adj_masks:
        for u in range(n):
            for v in range(u + 1, n):
                if best_adj_masks[u] & (1 << v):
                    edges.append([u, v])
    else:
        # Fallback to Clebsch
        for u in range(n):
            for v in range(u + 1, n):
                if (u ^ v) in S:
                    edges.append([u, v])
    
    return edges
