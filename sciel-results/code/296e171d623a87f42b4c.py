import time
import random
from itertools import combinations

def _fast_is_non_2colorable(n, edges):
    """
    Fast check using bitmasks.
    edges: list of tuples of 5 vertices.
    """
    if not edges:
        return False
    
    # Precompute edge masks
    # Vertex i is bit i
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    m = len(edge_masks)
    
    # Map vertex to list of edge indices containing it
    vert_to_edges = [[] for _ in range(n)]
    for i, mask in enumerate(edge_masks):
        # Iterate bits in mask
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            vert_to_edges[v].append(i)
            temp ^= lsb
            
    # State: C0, C1, Uncolored
    # We use a stack for backtracking? 
    # For speed, we can do DFS with bitmasks.
    
    def backtrack(C0, C1):
        # Unit Propagation
        changed = True
        while changed:
            changed = False
            for i in range(m):
                mask = edge_masks[i]
                c0_count = bin(mask & C0).count('1')
                c1_count = bin(mask & C1).count('1')
                
                if c0_count == 5 or c1_count == 5:
                    return False
                    
                if c0_count == 4 and c1_count == 0:
                    # Find the uncolored vertex
                    uncolored_in_edge = mask & ~(C0 | C1)
                    if uncolored_in_edge == 0:
                        # Should be 5 bits? No, 4 in C0, 0 in C1.
                        # If 5 bits total, 1 must be uncolored.
                        # If uncolored is 0, it means all 5 are in C0? No, c0_count is 4.
                        # So 1 bit is not in C0 or C1.
                        pass
                    # Assign that vertex to C1
                    C1 |= uncolored_in_edge
                    changed = True
                elif c1_count == 4 and c0_count == 0:
                    uncolored_in_edge = mask & ~(C0 | C1)
                    C0 |= uncolored_in_edge
                    changed = True
                    
        # Check if all colored
        if (C0 | C1) == (1 << n) - 1:
            return True # Found a valid coloring
            
        # Branch
        # Heuristic: Pick vertex with most constraints
        # For speed, just pick first uncolored? Or one involved in many edges?
        # Let's pick one involved in the most "critical" edges (4 colored)
        
        uncolored_mask = (1 << n) - 1 - (C0 | C1)
        if uncolored_mask == 0:
            return True
            
        # Find best vertex
        best_v = -1
        max_crit = 0
        temp = uncolored_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            crit = 0
            for i in vert_to_edges[v]:
                mask = edge_masks[i]
                c0_count = bin(mask & C0).count('1')
                c1_count = bin(mask & C1).count('1')
                if c0_count + c1_count == 4:
                    crit += 1
            if crit > max_crit:
                max_crit = crit
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
             # No critical edges, pick any
             best_v = uncolored_mask.bit_length() - 1 # Approx
             # Better:
             temp = uncolored_mask
             lsb = temp & -temp
             best_v = lsb.bit_length() - 1
             
        v_bit = 1 << best_v
        
        # Try C0
        if backtrack(C0 | v_bit, C1):
            return True
            
        # Try C1
        if backtrack(C0, C1 | v_bit):
            return True
            
        return False
        
    return not backtrack(0, 0)

def construct():
    deadline = time.time() + 140
    
    # Start with K9^5
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    current_edges = list(all_edges)
    
    # Verify it's non-2-colorable (it is, but just in case)
    if not _fast_is_non_2colorable(n, current_edges):
        # Fallback, though this shouldn't happen
        return {'n': n, 'edges': all_edges}
        
    # Greedy Deletion
    # We try to remove edges one by one.
    # To avoid getting stuck in a local minimum that is not the global minimum of this subgraph,
    # we can restart with different random orders if time permits.
    
    best_edges = current_edges
    best_count = len(current_edges)
    
    while time.time() < deadline - 10:
        # Shuffle indices
        indices = list(range(len(current_edges)))
        random.shuffle(indices)
        
        removed_any = False
        for idx in indices:
            if time.time() > deadline - 10:
                break
                
            # Try removing edge at idx
            candidate_edges = current_edges[:idx] + current_edges[idx+1:]
            
            if _fast_is_non_2colorable(n, candidate_edges):
                current_edges = candidate_edges
                removed_any = True
                if len(current_edges) < best_count:
                    best_count = len(current_edges)
                    best_edges = current_edges
                    # Optional: break and restart with new shuffle to escape local minima?
                    # No, keep deleting in this pass.
        
        if not removed_any:
            # No more edges could be removed in this random order.
            # Try a different random order?
            # If we do a full pass and remove nothing, we are at a local minimum for this order.
            # A different order might allow further removal.
            # But if we've done a full pass and removed nothing, it means for EVERY edge, removing it makes it 2-colorable.
            # This implies the graph is "minimal" with respect to edge deletion?
            # Yes, if for every e in E, H \ {e} is 2-colorable, then H is a minimal non-2-colorable hypergraph.
            # So we are done.
            break
            
    return {'n': n, 'edges': best_edges}
