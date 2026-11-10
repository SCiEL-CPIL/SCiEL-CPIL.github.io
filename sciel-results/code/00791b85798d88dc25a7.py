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
        Returns the saturated mask.
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
        return adj_mask

    def mask_to_edges(adj_mask):
        return [[i, j] for i in range(n) for j in range(i+1, n) if (adj_mask[i] >> j) & 1]

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
    
    base_mask = [0] * n
    for u, v in grotzsch_edges:
        base_mask[u] |= (1 << v)
        base_mask[v] |= (1 << u)

    # 2. Generate initial candidates by extending Grötzsch to 16 vertices
    # We have 5 new vertices (11..15). We connect them to independent sets of the Grötzsch graph.
    # To maximize density, we should connect them to large independent sets.
    
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
    new_vertices = list(range(11, 16))
    
    # Find a maximum independent set in the Grötzsch graph
    max_iset = find_max_independent_set_mask(base_mask, grotzsch_vertices)
    
    # Initial candidate: Connect all new vertices to the max independent set
    init_mask = base_mask[:]
    for nv in new_vertices:
        for iv in max_iset:
            init_mask[nv] |= (1 << iv)
            init_mask[iv] |= (1 << nv)
    
    # Verify initial graph
    if is_triangle_free_mask(init_mask) and not is_3_colorable_mask(init_mask):
        # Saturate
        saturated_mask = saturate(init_mask)
        m = edge_count_from_mask(saturated_mask)
        if m > best_score:
            best_score = m
            best_edges = mask_to_edges(saturated_mask)

    # 3. Local Search: Remove edge, re-saturate
    # We will perform this on the current best mask.
    # To diversify, we can also try different initial extensions.
    
    # Let's also try a few random extensions if time permits
    random_extensions = []
    # Generate a few random independent sets of size >= 3
    all_isets = []
    def find_isets(adj_mask, vertices, min_size, max_size, limit=20):
        results = []
        def backtrack(remaining, current_set):
            nonlocal results
            if len(results) >= limit:
                return
            if len(current_set) >= min_size:
                results.append(current_set[:])
            if not remaining or len(current_set) == max_size:
                return
            v = remaining[0]
            rest = remaining[1:]
            new_remaining = [u for u in rest if not (adj_mask[v] >> u) & 1]
            backtrack(new_remaining, current_set + [v])
            backtrack(rest, current_set)
        backtrack(list(vertices), [])
        return results

    isets = find_isets(base_mask, grotzsch_vertices, 3, 5, 10)
    
    current_mask = None
    if best_edges:
        # Reconstruct mask from best_edges for local search
        current_mask = [0] * n
        for u, v in best_edges:
            current_mask[u] |= (1 << v)
            current_mask[v] |= (1 << u)
    else:
        current_mask = saturated_mask

    # Local search loop
    while time.time() < deadline:
        # Pick a random edge to remove
        edges_in_current = mask_to_edges(current_mask)
        if not edges_in_current:
            break
            
        # Prefer removing edges that are not in the Grötzsch core?
        # Or just random.
        u, v = random.choice(edges_in_current)
        
        # Remove edge
        test_mask = [m | 0 for m in current_mask] # Copy
        test_mask[u] &= ~(1 << v)
        test_mask[v] &= ~(1 << u)
        
        # Re-saturate
        new_mask = saturate(test_mask)
        new_m = edge_count_from_mask(new_mask)
        
        # Check validity
        if is_triangle_free_mask(new_mask) and not is_3_colorable_mask(new_mask):
            if new_m > best_score:
                best_score = new_m
                best_edges = mask_to_edges(new_mask)
                current_mask = new_mask
            elif new_m == best_score:
                # Tie-breaking: keep current, or swap? 
                # Swapping might help escape local optima.
                # Let's accept ties with 50% probability to explore
                if random.random() < 0.5:
                    current_mask = new_mask
        else:
            # If invalid, we shouldn't have removed an edge that broke 4-chromaticity
            # unless re-saturation failed to fix it or created a triangle (shouldn't happen with saturate)
            # But saturate only adds edges if they keep it triangle-free.
            # So if new_mask is 3-colorable, it means removing that edge allowed a 3-coloring
            # that couldn't be blocked by adding other edges?
            # Actually, if G is 4-chromatic, G-e might be 3-colorable.
            # If G-e is 3-colorable, saturate(G-e) might still be 3-colorable.
            # In that case, we discard this move.
            pass

    if best_edges is None:
        # Fallback: return Grötzsch edges
        best_edges = [[u, v] for u, v in grotzsch_edges]
    
    return best_edges
