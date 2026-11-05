import time
import random
import sys

def construct():
    """
    Construct a C4-free graph on n vertices with min degree >= n - 39.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # Try n=47 first, then 46, etc.
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        required_min_deg = n - 39
        
        # Try to construct for this n
        result = _robust_construct(n, required_min_deg, deadline)
        
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
            
    # Fallback: return a valid small graph
    # n=39, min_deg=0 is always valid (empty graph)
    return {'n': 39, 'edges': []}

def _check_add_valid(masks, n, u, v):
    """
    Check if adding edge (u,v) to the graph represented by masks keeps it C4-free.
    masks[u] is an integer bitmask of neighbors of u.
    """
    # 1. Intersection of N(u) and N(v) must be <= 1
    if (masks[u] & masks[v]).bit_count() > 1:
        return False
        
    # 2. For all w in N(v), N(u) and N(w) must be disjoint (in current graph)
    #    Because v will be added to N(u), and w is in N(v) (so v in N(w) is NOT true, wait.
    #    Let's re-verify the logic.
    #    New N(u) = Old N(u) U {v}.
    #    New N(w) = Old N(w).
    #    Intersection New N(u) & New N(w) = (Old N(u) & Old N(w)) U ({v} & Old N(w)).
    #    If v in Old N(w), then v is in the intersection.
    #    So we need Old N(u) & Old N(w) to be empty.
    #    If v not in Old N(w), then intersection is just Old N(u) & Old N(w), which is <= 1.
    #    So condition: If v in N(w), then N(u) & N(w) == 0.
    
    # Iterate over neighbors of v
    mask_v = masks[v]
    while mask_v:
        lsb = mask_v & -mask_v
        w = lsb.bit_length() - 1
        if (masks[u] & masks[w]) != 0:
            return False
        mask_v ^= lsb
        
    # 3. Symmetric check for neighbors of u
    mask_u = masks[u]
    while mask_u:
        lsb = mask_u & -mask_u
        w = lsb.bit_length() - 1
        if (masks[v] & masks[w]) != 0:
            return False
        mask_u ^= lsb
        
    return True

def _greedy_fill(masks, n, target, deadline):
    """
    Greedily add edges to increase min degree.
    Returns True if target min degree is reached.
    """
    # List of all possible edges
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # We do multiple passes or random shuffles to try to reach the target
    for attempt in range(100):
        if time.time() > deadline:
            return False
            
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                return False
                
            # Check if adding (u,v) is valid
            if _check_add_valid(masks, n, u, v):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
        # Check min degree
        min_deg = min(masks[i].bit_count() for i in range(n))
        if min_deg >= target:
            return True
            
        # If we didn't add any edges in this pass, we might be stuck
        # We can check this, but just returning False to trigger local search is fine
        # However, if we added 0 edges, shuffling again won't help.
        # Let's track if any edge was added.
        # For simplicity, we just do a few passes and then move to local search.
        
    return False

def _local_search(masks, n, target, deadline):
    """
    Local search using edge swaps to increase min degree.
    """
    for _ in range(5000):  # Max iterations
        if time.time() > deadline:
            return False
            
        # Calculate degrees
        degrees = [masks[i].bit_count() for i in range(n)]
        min_deg = min(degrees)
        
        if min_deg >= target:
            return True
            
        # Find vertex with minimum degree
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        u = random.choice(min_vertices)
        
        # Find non-neighbors of u
        # ~masks[u] & ((1 << n) - 1) & ~(1 << u)
        non_neighbors_mask = ((1 << n) - 1) & ~masks[u] & ~(1 << u)
        
        if non_neighbors_mask == 0:
            return False
            
        # Get list of non-neighbors
        non_neighbors = []
        mm = non_neighbors_mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            non_neighbors.append(v)
            mm ^= lsb
            
        random.shuffle(non_neighbors)
        
        improved = False
        for v in non_neighbors:
            if _check_add_valid(masks, n, u, v):
                # Just add it
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                improved = True
                break
            else:
                # Try to find a swap
                # The blockage is due to some w in N(v) such that N(u) & N(w) != 0
                # Or w in N(u) such that N(v) & N(w) != 0
                
                # Let's find the specific conflict
                # Check neighbors of v
                mask_v = masks[v]
                swap_found = False
                while mask_v:
                    lsb = mask_v & -mask_v
                    w = lsb.bit_length() - 1
                    common = masks[u] & masks[w]
                    if common:
                        # Conflict exists. We have edge (u,v) blocked by path u-z-w-v where z in common.
                        # We can try to remove an edge from this 4-cycle.
                        # Candidates: (u, z) for z in common, (z, w), (w, v)
                        
                        # Try removing (z, w) and adding (u, v)
                        # We need to pick a specific z.
                        # Let's pick the first one.
                        lsb_z = common & -common
                        z = lsb_z.bit_length() - 1
                        
                        # Tentative swap: remove (z, w), add (u, v)
                        # 1. Remove (z, w)
                        masks[z] &= ~(1 << w)
                        masks[w] &= ~(1 << z)
                        
                        # 2. Check if (u, v) is now valid
                        if _check_add_valid(masks, n, u, v):
                            # Add (u, v)
                            masks[u] |= (1 << v)
                            masks[v] |= (1 << u)
                            swap_found = True
                            improved = True
                            break
                        else:
                            # Revert
                            masks[z] |= (1 << w)
                            masks[w] |= (1 << z)
                            # Try other z in common?
                            # For now, just break and try next v or next iteration
                            # We could iterate through all z in common, but let's keep it simple first
                            break
                    mask_v ^= lsb
                    if swap_found:
                        break
                
                if not swap_found:
                    # Try symmetric conflict from N(u) side?
                    # Or just move to next v
                    continue
                    
        if not improved:
            # Stuck for this u, try another random u or break?
            # If we can't improve any min-degree vertex, we might be stuck.
            # Let's try a few more times with different random choices
            if random.random() < 0.1:
                # Random restart of local search state? No, we can't reset masks easily without losing progress.
                # Just break and let outer loop restart greedy?
                break
                
    return False

def _robust_construct(n, target, deadline):
    """
    Combines greedy and local search with multiple restarts.
    """
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        masks = [0] * n
        
        # 1. Greedy Phase
        if _greedy_fill(masks, n, target, deadline):
            return (n, _to_edges(masks, n))
            
        # 2. Local Search Phase
        if _local_search(masks, n, target, deadline):
            return (n, _to_edges(masks, n))
            
    return None

def _to_edges(masks, n):
    edges = []
    for u in range(n):
        mask = masks[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            mask ^= lsb
    return edges

# EVOLVE-BLOCK-END
