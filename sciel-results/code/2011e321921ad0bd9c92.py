import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Construct Paley Tournament on p=67
    p = 67
    squares = {x * x % p for x in range(1, p)}
    
    # orientation list for return
    # i < j, 0 if i->j, 1 if j->i
    # i->j iff (j-i) % p is a square
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    # Bitsets for out-neighbors
    # out_bits[v] is an integer where bit u is 1 if v -> u
    out_bits = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            idx = i * (p - 1) - i * (i + 1) // 2 + (j - i - 1)
            # This index calculation is error prone. 
            # Let's just build out_bits directly from the condition.
            pass
            
    # Re-build out_bits correctly
    out_bits = [0] * p
    for i in range(p):
        for j in range(p):
            if i == j: continue
            # i -> j iff (j-i) % p in squares
            if (j - i) % p in squares:
                out_bits[i] |= (1 << j)
                
    V_current = list(range(p))
    mask_current = (1 << p) - 1
    
    # Helper to check if a subset S (as bitmask) is dominated by some w in V_current \ {v}
    # S is dominated by w if S is subset of out_bits[w] & mask_current
    # We want to know if ALL 4-subsets of out_bits[v] & mask_current are dominated by some w != v
    
    def is_removable(v):
        cur_out = out_bits[v] & mask_current
        # Get list of vertices in cur_out
        out_vertices = []
        temp = cur_out
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            out_vertices.append(u)
            temp ^= lsb
            
        if len(out_vertices) < 4:
            return True
            
        # Generate 4-subsets
        for S in itertools.combinations(out_vertices, 4):
            mask_S = 0
            for u in S:
                mask_S |= (1 << u)
                
            # Check if there exists w != v such that S subset of out_bits[w] & mask_current
            found_cover = False
            for w in V_current:
                if w == v:
                    continue
                # Optimization: if w is not in V_current, skip (but V_current is list of active)
                # Check subset
                if (mask_S & ~(out_bits[w] & mask_current)) == 0:
                    found_cover = True
                    break
                    
            if not found_cover:
                return False
        return True

    # Greedy removal
    while True:
        if time.time() > deadline:
            break
            
        removed = False
        # Try to remove a vertex
        # Order of removal can matter. Let's try all and pick the first removable one.
        # Or pick the one with smallest out-degree?
        removable_vertices = []
        for v in V_current:
            if is_removable(v):
                removable_vertices.append(v)
                
        if not removable_vertices:
            break
            
        # Pick one to remove
        # Heuristic: remove the one with the smallest out-degree in current graph?
        # Or just the first one.
        v_to_remove = removable_vertices[0]
        
        V_current.remove(v_to_remove)
        mask_current &= ~(1 << v_to_remove)
        removed = True
        
        if not removed:
            break
            
    # Construct final orientation
    n = len(V_current)
    if n < 5:
        # Fallback to original if it breaks down too much? 
        # But problem says 5 <= n. 
        # If we prune below 5, we should probably return a valid small one or the original.
        # Given constraints, we likely won't go below 5.
        pass
        
    final_orientation = []
    # Map old indices to new indices? 
    # The return format is just the list of bits for pairs (i,j) in the new vertex set 0..n-1.
    # We need to relabel vertices 0..n-1.
    # V_current contains the original indices.
    # Let's create a mapping.
    old_to_new = {old: new for new, old in enumerate(V_current)}
    
    for i_old in V_current:
        for j_old in V_current:
            if i_old < j_old:
                # Check orientation i_old -> j_old
                # From original construction: i->j iff (j-i)%p in squares
                if (j_old - i_old) % p in squares:
                    final_orientation.append(0)
                else:
                    final_orientation.append(1)
                    
    return {'n': n, 'orientation': final_orientation}
