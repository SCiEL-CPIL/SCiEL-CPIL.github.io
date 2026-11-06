import time
import random
from itertools import combinations

def is_2_colorable(edges, n):
    """
    Optimized 2-colorability check using bitmasks and backtracking.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
        
    # Precompute edge masks
    edge_masks = [0] * len(edges)
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks[i] = mask
        
    # Map vertex to list of edge indices containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
            
    # State: mask0 (vertices colored 0), mask1 (vertices colored 1)
    # We use recursion with backtracking.
    # To speed up, we use a stack or recursion with careful state management.
    
    # Heuristic: Pick unassigned vertex with highest "constraint pressure"
    # Constraint pressure: number of edges containing v that have 4 assigned vertices of one color.
    
    def get_pressure(v, m0, m1):
        pressure = 0
        for e_idx in vertex_edges[v]:
            em = edge_masks[e_idx]
            # Count assigned in this edge
            assigned_in_edge = em & (m0 | m1)
            # If 4 are assigned, check if they are all same color
            # We can check popcount of intersection with m0 and m1
            c0 = bin(assigned_in_edge & m0).count('1')
            c1 = bin(assigned_in_edge & m1).count('1')
            if c0 == 4 or c1 == 4:
                pressure += 1
        return pressure

    def backtrack(m0, m1):
        # Find unassigned vertex
        unassigned = ~(m0 | m1) & ((1 << n) - 1)
        if unassigned == 0:
            return True # All assigned, no contradiction found
        
        # Find best vertex to assign
        best_v = -1
        best_pressure = -1
        # Iterate through unassigned vertices
        # Note: iterating through all n vertices every time is O(n). 
        # We can maintain a list of unassigned vertices, but for simplicity:
        for v in range(n):
            if (unassigned >> v) & 1:
                p = get_pressure(v, m0, m1)
                if p > best_pressure:
                    best_pressure = p
                    best_v = v
                    
        if best_v == -1:
            # Should not happen if unassigned != 0
            return True
            
        # Try color 0
        if (m1 >> best_v) & 1 == 0: # Not already 1 (should be unassigned)
            new_m0 = m0 | (1 << best_v)
            new_m1 = m1
            # Check for immediate contradiction and propagation
            valid = True
            # Check edges containing best_v
            for e_idx in vertex_edges[best_v]:
                em = edge_masks[e_idx]
                c0 = bin(em & new_m0).count('1')
                c1 = bin(em & new_m1).count('1')
                if c0 == 5 or c1 == 5:
                    valid = False
                    break
                # Propagation
                if c0 == 4:
                    # Remaining must be 1
                    rem = em & ~(new_m0 | new_m1)
                    if rem != 0:
                        # If multiple remaining, we can't force just one? 
                        # Wait, if c0=4, there is exactly 1 unassigned in this edge.
                        # So rem must be a single bit.
                        if rem & (rem - 1) == 0:
                            # Force this vertex to 1
                            # But we need to check if it's already 0?
                            # If it's already 0, c0 would be 5.
                            # If it's already 1, c1 would be 1, c0=4, total 5 assigned? No.
                            # If rem is single bit, say u.
                            # If u is in m0, c0 would be 5.
                            # If u is in m1, c1 would be 1, c0=4, total 5 assigned.
                            # So u must be unassigned.
                            new_m1 |= rem
                            # Check if this causes conflict in other edges? 
                            # This is where simple propagation fails: we need to propagate recursively.
                            # For speed, let's just check if new_m1 conflicts with new_m0?
                            if new_m0 & new_m1:
                                valid = False
                                break
                            # We don't recurse immediately in this simple check, 
                            # we just update the state for the next recursive call.
                            # But this might miss immediate contradictions from the forced assignment.
                            # A more robust propagation is needed.
                if c1 == 4:
                    rem = em & ~(new_m0 | new_m1)
                    if rem != 0:
                        if rem & (rem - 1) == 0:
                            new_m0 |= rem
                            if new_m0 & new_m1:
                                valid = False
                                break

            if valid:
                if backtrack(new_m0, new_m1):
                    return True
                    
        # Try color 1
        if (m0 >> best_v) & 1 == 0:
            new_m0 = m0
            new_m1 = m1 | (1 << best_v)
            valid = True
            for e_idx in vertex_edges[best_v]:
                em = edge_masks[e_idx]
                c0 = bin(em & new_m0).count('1')
                c1 = bin(em & new_m1).count('1')
                if c0 == 5 or c1 == 5:
                    valid = False
                    break
                if c0 == 4:
                    rem = em & ~(new_m0 | new_m1)
                    if rem != 0:
                        if rem & (rem - 1) == 0:
                            new_m1 |= rem
                            if new_m0 & new_m1:
                                valid = False
                                break
                if c1 == 4:
                    rem = em & ~(new_m0 | new_m1)
                    if rem != 0:
                        if rem & (rem - 1) == 0:
                            new_m0 |= rem
                            if new_m0 & new_m1:
                                valid = False
                                break

            if valid:
                if backtrack(new_m0, new_m1):
                    return True

        return False

    return backtrack(0, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial best: K_9^5
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    # Local Search
    for trial in range(50): # Limit trials to ensure we finish
        if time.time() > deadline - 10.0:
            break
            
        # Choose n
        # n=9 is small, dense. n=10, 11 larger.
        # Start with n=10 or 11 to have more room to delete?
        # Or start with n=9 and delete?
        # Let's try n=10.
        n = random.choice([9, 10, 11])
        
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Start with all edges
        current_edges = [list(e) for e in all_edges]
        
        # Verify non-2-colorable
        if is_2_colorable(current_edges, n):
            continue
            
        # Greedy Deletion
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            random.shuffle(current_edges)
            
            for i in range(len(current_edges)):
                if time.time() > deadline - 10.0:
                    break
                if len(current_edges) <= best_count:
                    break
                    
                e_to_remove = current_edges[i]
                new_edges = current_edges[:i] + current_edges[i+1:]
                
                if not is_2_colorable(new_edges, n):
                    pass
                else:
                    current_edges = new_edges
                    improved = True
                    break # Restart pass with new shuffle
                    
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_n = n
            best_edges = current_edges

    return {'n': best_n, 'edges': best_edges}
