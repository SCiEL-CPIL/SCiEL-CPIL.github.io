import time
import random
from itertools import combinations

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    best_edges = None
    best_score = -1
    
    # --- Helper Functions ---
    
    def edge_count_from_mask(adj_mask):
        count = 0
        for i in range(n):
            count += bin(adj_mask[i]).count('1')
        return count // 2
    
    def is_triangle_free_mask(adj_mask):
        for i in range(n):
            mask_i = adj_mask[i]
            temp = mask_i
            while temp:
                bit = temp & (-temp)
                j = bit.bit_length() - 1
                if j > i:
                    if adj_mask[j] & mask_i:
                        return False
                temp ^= bit
        return True
    
    def is_3_colorable_mask(adj_mask):
        """
        Fast backtracking 3-coloring with bitmasks.
        Returns True if 3-colorable, False otherwise.
        """
        color = [-1] * n
        uncolored = list(range(n))
        
        def can_color():
            if not uncolored:
                return True
            
            # Pick uncolored vertex with fewest available colors (MRV heuristic)
            best_idx = -1
            best_count = 100
            best_avail = 0
            for idx, v in enumerate(uncolored):
                # Compute forbidden colors from colored neighbors
                forbidden = 0
                mask = adj_mask[v]
                for u in range(n):
                    if color[u] != -1 and (mask >> u) & 1:
                        forbidden |= (1 << color[u])
                avail = 7 & ~forbidden  # 7 = 0b111, available colors
                count = bin(avail).count('1')
                if count == 0:
                    return False
                if count < best_count:
                    best_count = count
                    best_idx = idx
                    best_avail = avail
                    if count == 1:
                        break
            
            v = uncolored[best_idx]
            uncolored.pop(best_idx)
            
            # Try each available color
            avail = best_avail
            while avail:
                c_bit = avail & (-avail)  # lowest set bit
                avail ^= c_bit
                c = c_bit.bit_length() - 1
                color[v] = c
                if can_color():
                    return True
                color[v] = -1
            
            uncolored.insert(best_idx, v)
            return False
        
        return can_color()

    def saturate(adj_mask):
        """
        Iteratively add edges that maintain triangle-freeness and 4-chromaticity.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                if time.time() > deadline:
                    return adj_mask
                for v in range(u + 1, n):
                    if adj_mask[u] & (1 << v):
                        continue
                    # Check triangle-freeness: no common neighbors
                    if adj_mask[u] & adj_mask[v]:
                        continue
                    
                    # Tentatively add edge
                    new_mask_u = adj_mask[u] | (1 << v)
                    new_mask_v = adj_mask[v] | (1 << u)
                    adj_mask[u] = new_mask_u
                    adj_mask[v] = new_mask_v
                    
                    # Check 3-colorability
                    if is_3_colorable_mask(adj_mask):
                        # Revert
                        adj_mask[u] = new_mask_u ^ (1 << v)
                        adj_mask[v] = new_mask_v ^ (1 << u)
                    else:
                        improved = True
                        # Continue scanning, as other edges might now be addable
                        # (Actually, adding edges restricts future additions, 
                        # but an edge rejected earlier might become acceptable? 
                        # No, if G+e is 3-colorable, G'+e is 3-colorable? 
                        # Wait. If G+e is 3-colorable, there is a coloring c.
                        # If we add more edges to G, c might not work.
                        # So G'+e might NOT be 3-colorable.
                        # So we should restart the scan or continue carefully.
                        # Since we are in a nested loop, we can just continue.
                        # But we might miss edges that were rejected before but are now valid?
                        # Actually, if an edge was rejected, it's because G+e was 3-colorable.
                        # Now we have G' = G + e_new.
                        # Is G'+e 3-colorable?
                        # If G+e is 3-colorable, let c be the coloring.
                        # c colors G. c colors e.
                        # c might not color e_new (if e_new is a conflict in c).
                        # But G' is a supergraph of G.
                        # If G+e is 3-colorable, it doesn't imply G'+e is 3-colorable.
                        # So yes, we might need to re-check previously rejected edges.
                        # However, simply continuing the loop and restarting if improved 
                        # handles this. The `while improved` loop restarts the scan.
                        # So we just need to break out of the inner loops to restart?
                        # Or we can just let the outer while loop handle it.
                        # If I add an edge, I set improved=True.
                        # The current pass continues.
                        # At the end of the pass, if improved, it restarts.
                        # This is correct.
        return adj_mask

    # --- Main Logic ---
    
    # 1. Generate Grötzsch graph (11 vertices)
    grotzsch_edges = []
    for i in range(5):
        grotzsch_edges.append((i, (i+1)%5))
    grotzsch_edges.append((5, 1))
    grotzsch_edges.append((5, 4))
    grotzsch_edges.append((6, 0))
    grotzsch_edges.append((6, 2))
    grotzsch_edges.append((7, 1))
    grotzsch_edges.append((7, 3))
    grotzsch_edges.append((8, 2))
    grotzsch_edges.append((8, 4))
    grotzsch_edges.append((9, 3))
    grotzsch_edges.append((9, 0))
    for i in range(5):
        grotzsch_edges.append((10, 5+i))
    
    # 2. Blow up to 16 vertices
    # We have 11 vertices in Grötzsch. We need 5 more.
    # Strategy: Connect new vertices to a maximum independent set of Grötzsch.
    # Grötzsch has independence number 4? 
    # Let's find max independent set of Grötzsch.
    
    base_mask = [0] * n
    for u, v in grotzsch_edges:
        base_mask[u] |= (1 << v)
        base_mask[v] |= (1 << u)
    
    def find_max_independent_set_mask(adj_mask, vertices):
        best_set = []
        
        def backtrack(remaining, current_set):
            nonlocal best_set
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            
            if not remaining:
                return
            
            v = remaining[0]
            rest = remaining[1:]
            
            # Option 1: Include v
            new_remaining = [u for u in rest if not (adj_mask[v] >> u) & 1]
            backtrack(new_remaining, current_set + [v])
            
            # Option 2: Exclude v
            backtrack(rest, current_set)
        
        backtrack(list(vertices), [])
        return best_set
    
    grotzsch_vertices = list(range(11))
    max_iset = find_max_independent_set_mask(base_mask, grotzsch_vertices)
    new_vertices = list(range(11, 16))
    
    # Construct initial graph
    init_mask = base_mask[:]
    for nv in new_vertices:
        for iv in max_iset:
            init_mask[nv] |= (1 << iv)
            init_mask[iv] |= (1 << nv)
    
    # Verify initial graph
    if is_triangle_free_mask(init_mask) and not is_3_colorable_mask(init_mask):
        m = edge_count_from_mask(init_mask)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if (init_mask[i] >> j) & 1]
        
        # Saturate
        saturated_mask = saturate(init_mask)
        m = edge_count_from_mask(saturated_mask)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if (saturated_mask[i] >> j) & 1]
    
    # 3. Try other independent sets if time permits
    if time.time() < deadline:
        def find_all_independent_sets_mask(adj_mask, vertices, min_size=4):
            results = []
            
            def backtrack(remaining, current_set):
                if len(current_set) >= min_size:
                    results.append(current_set[:])
                if not remaining:
                    return
                v = remaining[0]
                rest = remaining[1:]
                new_remaining = [u for u in rest if not (adj_mask[v] >> u) & 1]
                backtrack(new_remaining, current_set + [v])
                backtrack(rest, current_set)
            
            backtrack(list(vertices), [])
            return results
        
        isets = find_all_independent_sets_mask(base_mask, grotzsch_vertices, min_size=4)
        
        for iset in isets:
            if time.time() > deadline:
                break
            if len(iset) < 4:
                continue
            
            test_mask = base_mask[:]
            for nv in new_vertices:
                for iv in iset:
                    test_mask[nv] |= (1 << iv)
                    test_mask[iv] |= (1 << nv)
            
            if not is_triangle_free_mask(test_mask):
                continue
            
            if is_3_colorable_mask(test_mask):
                continue
            
            # Saturate
            saturated_mask = saturate(test_mask)
            m = edge_count_from_mask(saturated_mask)
            if m > best_score:
                best_score = m
                best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if (saturated_mask[i] >> j) & 1]
    
    if best_edges is None:
        # Fallback: return Grötzsch edges
        best_edges = [[u, v] for u, v in grotzsch_edges]
    
    return best_edges
