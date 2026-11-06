import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 50.0  # 50s safety margin for 60s limit (or 90s if evaluator kills at 90)
    # The prompt says "KILLS ... after 90s wall-clock". 
    # It also says "Each evaluation must finish within 60 s".
    # Let's assume 60s is the hard limit for the function to return.
    # I will use a 55s deadline to be safe.
    
    N = 256
    # 1. Identify forbidden vertices (2a is a square)
    forbidden = set()
    for a in range(1, N + 1):
        s = 2 * a
        r = int(math.isqrt(s))
        if r * r == s:
            forbidden.add(a)
            
    # Valid vertices
    valid_verts = [v for v in range(1, N + 1) if v not in forbidden]
    n = len(valid_verts)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(valid_verts)}
    
    # 2. Build adjacency bitmasks
    # squares up to 256+256=512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
        
    # neighbors[i] is bitmask of vertices j such that valid_verts[i] + valid_verts[j] is a square
    # We only need j > i or just all, but for MIS we usually use symmetric or directed.
    # Let's store full neighborhood including self? No, MIS removes N[v] which includes v.
    # Let's store open neighborhood `neigh[i]` (j != i).
    neigh = [0] * n
    
    for i in range(n):
        vi = valid_verts[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = valid_verts[j]
            if vi + vj in squares:
                mask |= (1 << j)
        neigh[i] = mask
        
    # closed_neigh[i] = neigh[i] | (1 << i)
    closed_neigh = [neigh[i] | (1 << i) for i in range(n)]
    
    best_size = 0
    best_set = 0
    
    # Tomita's algorithm for Maximum Independent Set
    # R: current independent set (bitmask)
    # C: candidate set (bitmask)
    
    def greedy_color(C):
        """
        Greedily color the subgraph induced by C.
        Returns list of bitmasks representing color classes.
        """
        color_classes = []
        uncolored = C
        while uncolored:
            # Pick a vertex with highest degree in uncolored? 
            # Or just first available? Tomita usually picks from uncolored.
            # For simplicity and speed, we can just pick the lowest index or use a heuristic.
            # A better heuristic: pick vertex with max degree in current uncolored set.
            # But that's expensive. Let's stick to a simple sequential coloring which is often effective enough with the bound.
            # Actually, the order of coloring affects the tightness of the bound.
            # Let's iterate through vertices in C in increasing index order.
            
            color_mask = 0
            # We need to pick a vertex v from uncolored.
            # To make it fast, we can just take the lowest set bit.
            # This is O(n^2) in worst case if we check degrees, but O(n) if we just pick lowest.
            # Let's pick lowest set bit for speed.
            v_bit = uncolored & (-uncolored)
            v_idx = v_bit.bit_length() - 1
            
            # Add v to current color class
            color_mask |= v_bit
            uncolored ^= v_bit
            
            # Remove neighbors of v from uncolored (they can't have same color)
            uncolored &= ~neigh[v_idx]
            
            color_classes.append(color_mask)
            
        return color_classes

    def expand(R, C):
        nonlocal best_size, best_set
        
        # Bound: |R| + |C| <= best_size -> prune
        # Stronger bound: |R| + max_colors(C) <= best_size
        if not C:
            if R.bit_count() > best_size:
                best_size = R.bit_count()
                best_set = R
            return
            
        # Check time
        if time.time() > deadline:
            return
            
        color_classes = greedy_color(C)
        max_colors = len(color_classes)
        
        if R.bit_count() + max_colors <= best_size:
            return
            
        # Iterate through color classes from highest to lowest?
        # Standard Tomita:
        # While C is not empty:
        #   1. Check bound
        #   2. Pick v from last color class (highest color)
        #   3. Expand(R | {v}, C \ N[v])
        #   4. C = C \ {v}
        
        # We need to manage C carefully.
        # The `color_classes` list corresponds to the coloring of the *current* C.
        # When we remove a vertex, the coloring becomes invalid.
        # So we re-color in each recursive call? That's what `expand` does at the start.
        # But inside the loop, we pick v, recurse, then remove v from C.
        # The `max_colors` bound is only valid for the initial C.
        # After removing v, the bound might be tighter, but we don't recalculate until the next `expand` call.
        # However, we can simply loop while C is not empty.
        
        # To pick v from the "last" color class, we need to know which vertices are in the last class.
        # `color_classes[-1]` contains the vertices with the highest color.
        
        # Optimization: If the last color class has multiple vertices, we can iterate through them.
        # But standard Tomita picks one, recurses, removes it, and continues.
        # The set of vertices in the last color class is `last_class = color_classes[-1]`.
        
        # We will iterate through vertices in `last_class`?
        # No, we iterate through vertices in C.
        # But we prioritize vertices with higher colors.
        
        # Let's just iterate through the bits in C.
        # To respect the heuristic, we can iterate through `color_classes` in reverse order, 
        # and for each class, iterate through its vertices.
        
        # However, `C` changes.
        # Let's use a simpler approach:
        # While C:
        #   if bound fails: break
        #   pick v with highest degree in C (or from last color class)
        #   expand(R|v, C \ N[v])
        #   C = C \ v
        
        # To pick v from last color class:
        # We have `color_classes`.
        # We can extract vertices from `color_classes[-1]` first, then `[-2]`, etc.
        # But `C` is the union of all classes.
        
        # Let's just pick a vertex from C. 
        # Heuristic: vertex with max degree in C.
        # This is O(|C|^2) per node. Might be slow.
        # Using the color class is O(1) to pick if we just take one bit from `color_classes[-1]`.
        
        # Let's try picking from `color_classes[-1]`.
        
        # We need to ensure we don't pick a vertex that has already been removed from C in a previous iteration of this loop.
        # So we check if v is still in C.
        
        # We'll process color classes from last to first.
        # But we need to maintain the state of C.
        
        # Alternative: Just loop while C is not empty.
        # Pick v = argmax degree in C?
        # Let's try picking the vertex from the last color class that is still in C.
        
        # Since we re-color at the start of `expand`, `color_classes` is valid for the current C.
        # We will iterate through the vertices of the last color class.
        
        last_class = color_classes[-1]
        
        # We need to iterate through vertices in C.
        # Let's just pick the lowest index vertex from the last color class that is in C.
        # If none, pick from second last, etc.
        
        # Actually, a simpler robust method:
        # While C:
        #   if time exceeded: return
        #   # Update bound? No, bound is fixed for this C instance.
        #   # Pick v
        #   # To pick v, we can just take the lowest bit of C.
        #   # But to be smart, take from last color class.
        
        # Let's construct a list of candidates from last to first color class.
        # But C shrinks.
        
        # Let's just pick the vertex with the highest degree in C.
        # It's often faster than complex color management if n is small (245).
        # Degree calculation:
        # deg(v) = popcount(neigh[v] & C)
        
        # Let's try this.
        
        while C:
            if time.time() > deadline:
                return
            
            # Pick v with max degree in C
            # If tie, lowest index?
            max_deg = -1
            v_idx = -1
            # Iterate through bits in C
            temp = C
            while temp:
                bit = temp & (-temp)
                idx = bit.bit_length() - 1
                d = (neigh[idx] & C).bit_count()
                if d > max_deg:
                    max_deg = d
                    v_idx = idx
                temp ^= bit
            
            if v_idx == -1:
                break
                
            v_bit = 1 << v_idx
            
            # Branch 1: Include v
            new_R = R | v_bit
            new_C = C & ~closed_neigh[v_idx]
            expand(new_R, new_C)
            
            # Branch 2: Exclude v
            C = C & ~v_bit
            
            # Check bound with updated C?
            # The initial bound `R.bit_count() + max_colors <= best_size` is still a valid upper bound 
            # because removing vertices from C cannot increase the MIS size.
            # So if the initial bound was tight enough to prune, we would have returned.
            # If not, we continue.
            
            # We can re-check a simple bound: R.bit_count() + C.bit_count() <= best_size
            if R.bit_count() + C.bit_count() <= best_size:
                break

    # Initial call
    # R = 0, C = all valid vertices
    initial_C = (1 << n) - 1
    expand(0, initial_C)
    
    # Convert best_set bitmask back to values
    result = []
    for i in range(n):
        if best_set & (1 << i):
            result.append(valid_verts[i])
            
    # If no result found (shouldn't happen), return a single valid element
    if not result:
        return [valid_verts[0]]
        
    return sorted(result)

# EVOLVE-BLOCK-END
