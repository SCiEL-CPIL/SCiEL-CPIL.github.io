import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds limit to be safe
    
    n = 256
    # Bitmask representation
    # Vertex i corresponds to bit i (0-indexed for 1..256? No, let's use 0..255 for 1..256)
    # Value v (1..256) -> index v-1
    
    # Precompute neighbors bitmask
    neighbors = [0] * n
    for a in range(1, n + 1):
        ia = a - 1
        for b in range(a + 1, n + 1):
            ib = b - 1
            if (a * b) % (a + b) == 0:
                neighbors[ia] |= (1 << ib)
                neighbors[ib] |= (1 << ia)
                
    all_mask = (1 << n) - 1
    best_size = 0
    best_mask = 0
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
        
    # Greedy coloring bound
    # Returns the number of colors used
    def color_bound(mask):
        if mask == 0:
            return 0
        # Simple greedy coloring
        # We want to color the graph induced by mask
        # To get a tight bound, we should order vertices by degree in the subgraph
        # But for speed, a simple heuristic works.
        # Let's just iterate and assign smallest available color.
        # Actually, for MIS, the bound is the chromatic number of the complement? No.
        # The size of the maximum independent set is <= the number of colors in a proper coloring of the graph.
        # So we color the graph G induced by mask.
        
        colors = 0
        remaining = mask
        while remaining:
            colors += 1
            # Pick a vertex with max degree in remaining? Or just first?
            # Let's pick the first set bit for simplicity, or one with high degree.
            # High degree is better for coloring tightness?
            # Actually, we just need any valid coloring.
            # Let's use a simple list of vertices.
            verts = []
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                verts.append(v)
                temp ^= lsb
                
            # Sort by degree in remaining descending?
            # Degree in remaining: popcount(neighbors[v] & remaining)
            verts.sort(key=lambda v: popcount(neighbors[v] & remaining), reverse=True)
            
            color_class = 0
            uncolored = set(verts)
            for v in verts:
                if v not in uncolored:
                    continue
                # Check if v can be added to current color class
                # It must not be connected to any vertex already in color_class
                # color_class is a bitmask
                if (neighbors[v] & color_class) == 0:
                    color_class |= (1 << v)
                    uncolored.remove(v)
                    
            remaining &= ~color_class
            
        return colors

    def search(mask, current_size):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = 0 # We don't strictly need the mask, just size, but good for construction
            return
            
        # Upper bound: current_size + popcount(mask)
        if current_size + popcount(mask) <= best_size:
            return
            
        # Coloring bound
        cb = color_bound(mask)
        if current_size + cb <= best_size:
            return
            
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in mask
        verts = []
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts.append(v)
            temp ^= lsb
            
        # Sort by degree
        verts.sort(key=lambda v: popcount(neighbors[v] & mask), reverse=True)
        v = verts[0]
        v_bit = 1 << v
        
        # Branch 1: Include v
        # Remove v and its neighbors from mask
        new_mask = mask & ~(neighbors[v] | v_bit)
        search(new_mask, current_size + 1)
        
        # Branch 2: Exclude v
        new_mask = mask & ~v_bit
        search(new_mask, current_size)
        
    # Initial call
    # We can start with a greedy solution to set a good initial best_size
    # Greedy: pick vertices with lowest degree?
    # Or just start with 0.
    
    # Let's try a simple greedy to get a good start
    curr_mask = all_mask
    curr_size = 0
    while curr_mask:
        # Pick vertex with min degree in curr_mask
        verts = []
        temp = curr_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts.append(v)
            temp ^= lsb
        verts.sort(key=lambda v: popcount(neighbors[v] & curr_mask))
        v = verts[0]
        curr_size += 1
        curr_mask &= ~(neighbors[v] | (1 << v))
        
    best_size = curr_size
    # best_mask construction from greedy is complex, but we only need size for bound.
    # For final answer, we need the set.
    # Let's store the set in the search or reconstruct.
    # Reconstructing from mask is easy.
    
    # Actually, let's just run the search. The initial bound helps.
    # But we need the actual set.
    # We can store the path in the search.
    
    # To keep it simple and fast, we'll just find the max size.
    # Then we can run a second pass or store the mask.
    # Storing the mask in nonlocal is fine.
    
    # However, the search above updates best_size but not best_mask correctly in all paths easily without passing the mask of chosen vertices.
    # Let's modify search to take a `chosen_mask` parameter.
    
    # Revised Search
    def search2(mask, chosen_mask, current_size):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = chosen_mask
            return
            
        if current_size + popcount(mask) <= best_size:
            return
            
        cb = color_bound(mask)
        if current_size + cb <= best_size:
            return
            
        # Pick vertex
        verts = []
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts.append(v)
            temp ^= lsb
        verts.sort(key=lambda v: popcount(neighbors[v] & mask), reverse=True)
        v = verts[0]
        v_bit = 1 << v
        
        # Include
        new_mask = mask & ~(neighbors[v] | v_bit)
        search2(new_mask, chosen_mask | v_bit, current_size + 1)
        
        # Exclude
        new_mask = mask & ~v_bit
        search2(new_mask, chosen_mask, current_size)
        
    search2(all_mask, 0, 0)
    
    # Construct result from best_mask
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    # If no result found (shouldn't happen), return greedy
    if not result:
        result = list(range(1, 257, 2))
        
    return result
