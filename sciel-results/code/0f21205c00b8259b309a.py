import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    num_edges = len(edges)
    
    # Convert edges to bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # Precompute vertex -> list of edges
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edge_masks):
        for v in range(n):
            if em & (1 << v):
                vertex_edges[v].append(i)
                
    # State
    c0 = [0] * num_edges
    c1 = [0] * num_edges
    color = [-1] * n
    
    # Queue for propagation
    # We will use a simple list as a queue
    # To avoid re-processing, we can just loop until stable, 
    # but a queue is more efficient if changes are sparse.
    # Given n=9, m~100, full scan is fast.
    # Let's stick to full scan for simplicity but optimize the inner loop.
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for i in range(num_edges):
                # Check for immediate conflict
                if c0[i] == 5 or c1[i] == 5:
                    return False
                
                # Unit propagation
                if c0[i] == 4 and c1[i] == 0:
                    # Find unassigned vertex
                    # We can use bitmask to find it faster?
                    # mask_unassigned = edge_masks[i] & ~color_mask_0 & ~color_mask_1
                    # But we don't have color_mask_0/1 easily accessible in this structure.
                    # We have color[v].
                    # Let's just loop.
                    unassigned_v = -1
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            unassigned_v = v
                            break
                    if unassigned_v != -1:
                        color[unassigned_v] = 1
                        c1[i] += 1
                        for other_e in vertex_edges[unassigned_v]:
                            c1[other_e] += 1
                        changed = True
                        # Check conflict immediately
                        if c1[i] == 5: # Should be 5 now? No, c1[i] was 0, now 1. Wait.
                            # If c0[i]==4, c1[i]==0, and we assign 1, c1[i] becomes 1.
                            # Total assigned = 4+1=5.
                            # So the edge is satisfied (not monochromatic).
                            # Wait, the constraint is "NOT monochromatic".
                            # Monochromatic means 5-0 or 0-5.
                            # If c0=4, c1=0, and we assign 1, we have 4-1. Not mono.
                            # So no conflict.
                            pass
                        # Check other edges for conflict
                        # We will check in the next iteration of the while loop
                        # Or we can check immediately for the updated edges
                        for other_e in vertex_edges[unassigned_v]:
                            if c1[other_e] == 5:
                                return False
                            
                elif c1[i] == 4 and c0[i] == 0:
                    unassigned_v = -1
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            unassigned_v = v
                            break
                    if unassigned_v != -1:
                        color[unassigned_v] = 0
                        c0[i] += 1
                        for other_e in vertex_edges[unassigned_v]:
                            c0[other_e] += 1
                        changed = True
                        for other_e in vertex_edges[unassigned_v]:
                            if c0[other_e] == 5:
                                return False
        return True

    def backtrack():
        # Symmetry breaking: Fix color[0] = 0 if unassigned
        # We can do this at the start of the search
        # But inside backtrack, we pick a variable.
        
        # Find most constrained edge
        best_edge = -1
        best_unassigned = 6
        best_max_color = 0
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            unassigned = 5 - c0[i] - c1[i]
            max_color = max(c0[i], c1[i])
            
            if unassigned < best_unassigned or (unassigned == best_unassigned and max_color > best_max_color):
                best_unassigned = unassigned
                best_max_color = max_color
                best_edge = i
                
        if best_edge == -1:
            return True # All edges satisfied
            
        if best_unassigned == 0:
            return False
            
        candidates = []
        for v in range(n):
            if color[v] == -1 and (edge_masks[best_edge] & (1 << v)):
                candidates.append(v)
                
        if not candidates:
            return True
            
        # Try assigning each candidate
        # Heuristic: Try the color that appears less often in the edge?
        # Or just 0 then 1.
        
        for v in candidates:
            # Try color 0
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            
            color[v] = 0
            ok = True
            for e_idx in vertex_edges[v]:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    ok = False
                    break
            
            if ok and propagate():
                if backtrack():
                    return True
            
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
            # Try color 1
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            
            color[v] = 1
            ok = True
            for e_idx in vertex_edges[v]:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    ok = False
                    break
            
            if ok and propagate():
                if backtrack():
                    return True
            
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
        return False

    # Initial symmetry breaking
    if n > 0:
        color[0] = 0
        for e_idx in vertex_edges[0]:
            c0[e_idx] += 1
            if c0[e_idx] == 5:
                return False # Edge with 5 zeros is monochromatic? 
                # Wait. If an edge has 5 zeros, it is monochromatic.
                # So if we assign 0 to v, and v is in an edge, and that edge already has 4 zeros,
                # then the edge has 5 zeros. Conflict.
                # But we check c0[e_idx] == 5.
                # If c0[e_idx] becomes 5, it is monochromatic.
                # So we should return False (not 2-colorable)?
                # No, we are checking IF it is 2-colorable.
                # If we find a conflict, this branch is invalid.
                # So we return False from the branch, meaning "no coloring found in this branch".
                # But if ALL branches fail, it is not 2-colorable.
                # So `backtrack` returns False if no coloring exists.
                # `is_2_colorable_fast` returns `backtrack()`.
                # So if `backtrack` returns False, `is_2_colorable_fast` returns False.
                # This is correct.
                
    if not propagate():
        return False
        
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = None
    best_n = 0
    best_m = 10**9
    
    # Try n=9 first as it's the smallest n where K_n^5 is not 2-colorable
    n = 9
    all_possible = list(combinations(range(n), 5))
    current_edges = [list(e) for e in all_possible]
    current_m = len(current_edges)
    
    # Multiple restarts
    for attempt in range(20): # Increased restarts
        if time.time() > deadline:
            break
            
        # Reset to full graph? Or continue from best?
        # Continuing from best is better if we are improving.
        # But random restarts can escape local minima.
        # Let's try to improve the current best.
        
        order = list(range(len(current_edges)))
        random.shuffle(order)
        
        improved = True
        while improved:
            improved = False
            for idx in order:
                if time.time() > deadline:
                    break
                if idx >= len(current_edges):
                    continue
                    
                new_edges = current_edges[:idx] + current_edges[idx+1:]
                
                if not is_2_colorable_fast(n, new_edges):
                    current_edges = new_edges
                    current_m = len(current_edges)
                    improved = True
                    order = list(range(len(current_edges)))
                    random.shuffle(order)
                    break
                    
        if current_m < best_m:
            best_m = current_m
            best_edges = current_edges[:]
            best_n = n
            
    if best_edges is None:
        best_edges = [list(e) for e in all_possible]
        best_n = n
        best_m = len(best_edges)
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
