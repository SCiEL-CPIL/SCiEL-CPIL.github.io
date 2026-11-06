import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_edges = None
    best_count = float('inf')
    best_n = None

    # We try n from 9 to 12. 
    # n=9 is guaranteed non-2-colorable for K_9^(5).
    # Larger n might allow fewer edges in the sparsest non-2-colorable subgraph.
    
    for n in [9, 10, 11, 12]:
        if time.time() > deadline - 5:
            break
            
        # Generate all edges
        all_edges = [list(e) for e in combinations(range(n), 5)]
        m_total = len(all_edges)
        
        # Precompute edge masks
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Precompute bad_masks for all colorings
        # bad_masks[c] is a bitmask of edges that are monochromatic under coloring c
        num_colorings = 1 << n
        bad_masks = [0] * num_colorings
        
        for c in range(num_colorings):
            bm = 0
            for i in range(m_total):
                em = edge_masks[i]
                inter = c & em
                if inter == 0 or inter == em:
                    bm |= (1 << i)
            bad_masks[c] = bm
        
        # Check if the complete graph is non-2-colorable
        # If it is 2-colorable, no subset can be non-2-colorable.
        full_mask = (1 << m_total) - 1
        is_full_non_colorable = True
        for c in range(num_colorings):
            if (bad_masks[c] & full_mask) == 0:
                is_full_non_colorable = False
                break
        
        if not is_full_non_colorable:
            continue
            
        def is_non_colorable(mask):
            """Check if hypergraph defined by edge bitmask 'mask' is NOT 2-colorable."""
            for c in range(num_colorings):
                if (bad_masks[c] & mask) == 0:
                    return False
            return True

        # Greedy Reduction with Restarts
        # We want to minimize the number of edges.
        # Start with full_mask and remove edges one by one if possible.
        
        # To avoid getting stuck in the same local minimum, we do multiple restarts
        # with different random orders.
        
        rng = random.Random(n)
        best_local_mask = full_mask
        best_local_count = m_total
        
        # How many restarts?
        # Each restart involves shuffling m_total items and checking removal for each.
        # Cost per restart ~ m_total * cost_check.
        # cost_check ~ num_colorings.
        # For n=12, m=792, num_colorings=4096.
        # 792 * 4096 ~ 3.2M ops.
        # In Python, this might take ~0.1-0.5s per restart.
        # We have ~100s budget. We can do ~200-1000 restarts.
        
        # Let's do up to 500 restarts or until time runs out.
        max_restarts = 500
        
        for _ in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            curr_mask = full_mask
            # Shuffle edge indices
            indices = list(range(m_total))
            rng.shuffle(indices)
            
            # Try to remove edges in random order
            for i in indices:
                if curr_mask & (1 << i):
                    new_mask = curr_mask & ~(1 << i)
                    if is_non_colorable(new_mask):
                        curr_mask = new_mask
            
            # Update best
            count = bin(curr_mask).count('1')
            if count < best_local_count:
                best_local_count = count
                best_local_mask = curr_mask
                
            # If we reached the known best (51), we can stop early for this n
            if best_local_count <= 51:
                break

        # Update global best
        if best_local_count < best_count:
            best_count = best_local_count
            best_n = n
            best_edges = [all_edges[i] for i in range(m_total) if best_local_mask & (1 << i)]
            
        # If we found 51 edges, it's very hard to beat. We can stop.
        if best_count <= 51:
            break

    # Fallback: If for some reason we didn't find anything (should not happen for n=9)
    if best_edges is None:
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        best_edges = edges
        best_n = n

    return {'n': best_n, 'edges': best_edges}
