import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Uses bitmask representation for speed.
    Returns True if there exists a 2-coloring with no monochromatic edge.
    """
    if not edges:
        return True
        
    # Convert edges to bitmasks
    # Vertex v corresponds to bit v
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # Pre-compute all_vertices_mask
    all_mask = (1 << n) - 1
    
    def propagate(mask0, mask1):
        # Returns (False, 0, 0) if contradiction, (True, mask0, mask1) otherwise
        changed = True
        while changed:
            changed = False
            for e in edge_masks:
                # Check if monochromatic 0: no vertex in e is in mask1
                if (e & mask1) == 0:
                    count0 = (e & mask0).bit_count()
                    if count0 == 5:
                        return False, 0, 0
                    if count0 == 4:
                        unassigned_vertex = e & ~(mask0 | mask1)
                        mask1 |= unassigned_vertex
                        changed = True
                        continue
                        
                # Check if monochromatic 1: no vertex in e is in mask0
                if (e & mask0) == 0:
                    count1 = (e & mask1).bit_count()
                    if count1 == 5:
                        return False, 0, 0
                    if count1 == 4:
                        unassigned_vertex = e & ~(mask0 | mask1)
                        mask0 |= unassigned_vertex
                        changed = True
                        continue
                        
        return True, mask0, mask1

    def solve(mask0, mask1):
        # Heuristic: Pick vertex involved in most "critical" edges
        unassigned_mask = all_mask & ~(mask0 | mask1)
        if unassigned_mask == 0:
            return True
        
        # Calculate scores for vertex selection
        # Score[v] = number of edges where v is the unique unassigned vertex and 4 others are same color.
        # This indicates high constraint.
        # We also consider general degree as a tie-breaker or secondary metric.
        
        scores = {}
        
        # Iterate through edges to find critical vertices
        for e in edge_masks:
            unassigned_in_e = e & unassigned_mask
            if unassigned_in_e == 0:
                continue # Fully assigned
            if (unassigned_in_e & (unassigned_in_e - 1)) != 0:
                continue # More than 1 unassigned, not a unit clause candidate for "critical" score in the strict sense
                # However, edges with 2 unassigned might still be useful for heuristic?
                # Let's stick to the "unit clause" heuristic for selection as it's most direct.
                
            v = unassigned_in_e.bit_length() - 1
            
            # Check if the other 4 are same color
            # If (e & mask1) == 0: All assigned in e are 0. Since only v is unassigned, others are 0.
            #   So v must be 1.
            # If (e & mask0) == 0: All assigned in e are 1. Since only v is unassigned, others are 1.
            #   So v must be 0.
            
            # If neither, it's not a unit clause forcing a specific value, but it's a "tight" edge.
            # Actually, if unassigned_in_e is a single bit, then:
            # count0 = (e & mask0).bit_count()
            # count1 = (e & mask1).bit_count()
            # count0 + count1 + 1 = 5 => count0 + count1 = 4
            # If count0 == 4, count1 == 0 -> v must be 1
            # If count1 == 4, count0 == 0 -> v must be 0
            # If count0 == 3, count1 == 1 -> v can be 0 or 1 (not forced)
            # If count0 == 2, count1 == 2 -> v can be 0 or 1
            
            # For heuristic, we want to pick v that is in many "tight" situations.
            # Let's score v for every edge where it is the *only* unassigned vertex.
            # This is a strong signal that this vertex is central to the current constraint state.
            
            if v in scores:
                scores[v] += 1
            else:
                scores[v] = 1
                
        if not scores:
            # No vertex is the unique unassigned in any edge.
            # Fall back to degree or random.
            # Let's pick the vertex with the highest degree in the remaining unassigned subgraph?
            # Or just first unassigned.
            # To avoid bias, let's pick a random unassigned vertex.
            # Or better: count how many edges contain each unassigned vertex.
            deg = {v: 0 for v in range(n) if (unassigned_mask >> v) & 1}
            for e in edge_masks:
                unassigned_in_e = e & unassigned_mask
                while unassigned_in_e:
                    bit = unassigned_in_e & -unassigned_in_e
                    v = bit.bit_length() - 1
                    deg[v] += 1
                    unassigned_in_e ^= bit
            
            if deg:
                v = max(deg, key=deg.get)
            else:
                v = (unassigned_mask & -unassigned_mask).bit_length() - 1
        else:
            # Pick vertex with max score
            v = max(scores, key=scores.get)
        
        # Try 0
        new_mask0 = mask0 | (1 << v)
        ok, m0, m1 = propagate(new_mask0, mask1)
        if ok and solve(m0, m1):
            return True
            
        # Try 1
        new_mask1 = mask1 | (1 << v)
        ok, m0, m1 = propagate(mask0, new_mask1)
        if ok and solve(m0, m1):
            return True
            
        return False

    ok, m0, m1 = propagate(0, 0)
    if not ok:
        return False
        
    return solve(m0, m1)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Search n from 9 to 26
    for n in range(9, 27):
        if time.time() > deadline:
            break
            
        restarts = 50
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Start with a moderate number of edges
            k = min(len(all_possible), 40 + random.randint(0, 20))
            edges = [list(e) for e in all_possible[:k]]
            
            edge_set = set(tuple(e) for e in edges)
            
            # Ensure non-2-colorable
            # If colorable, add edges
            while is_2_colorable_fast(n, edges):
                if time.time() > deadline:
                    break
                if len(edges) < len(all_possible):
                    # Pick random edge
                    while True:
                        new_edge = tuple(random.choice(all_possible))
                        if new_edge not in edge_set:
                            edges.append(list(new_edge))
                            edge_set.add(new_edge)
                            break
                else:
                    break
            
            if time.time() > deadline:
                break
                
            # Verify non-2-colorable
            if is_2_colorable_fast(n, edges):
                continue
                
            # Greedy Deletion
            current_edges = set(tuple(e) for e in edges)
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                removal_list = list(current_edges)
                random.shuffle(removal_list)
                
                for edge_tuple in removal_list:
                    if time.time() > deadline:
                        break
                        
                    if edge_tuple not in current_edges:
                        continue
                        
                    test_edges = current_edges - {edge_tuple}
                    test_list = [list(e) for e in test_edges]
                    
                    if not is_2_colorable_fast(n, test_list):
                        current_edges = test_edges
                        improved = True
                        break # Restart pass to re-shuffle and check new state
                        
            final_edges = [list(e) for e in current_edges]
            if not is_2_colorable_fast(n, final_edges):
                if len(final_edges) < best_count:
                    best_count = len(final_edges)
                    best_edges = final_edges
                    best_n = n
                    print(f"Found better: n={best_n}, edges={best_count}")
                    
    if best_edges is None:
        # Fallback: K_9^5
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        
    return {'n': best_n, 'edges': best_edges}
