"""Model-and-optimize: construct large triangle-free 4-colorings via block construction + local search."""
import random
import time
import math

# EVOLVE-BLOCK-START

def _baseline32():
    """Greenwood-Gleason K16 doubled -> K32."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}

def _block_coloring(n, rng):
    """
    Block construction: partition n vertices into 4 subsets.
    Color c used for edges within subset c.
    Inter-subset edges use a fixed triangle-free 3-coloring pattern.
    Returns (colors_list, subsets) or None if n too small.
    """
    if n < 4:
        return None
    
    # Partition into 4 subsets as evenly as possible
    base = n // 4
    rem = n % 4
    sizes = [base + (1 if i < rem else 0) for i in range(4)]
    
    # Build subsets
    subsets = []
    pos = 0
    for s in sizes:
        subsets.append(list(range(pos, pos + s)))
        pos += s
    
    # Precompute: for each vertex, which subset it's in
    vert_to_sub = [0] * n
    for c in range(4):
        for v in subsets[c]:
            vert_to_sub[v] = c
    
    # Fixed triangle-free 3-coloring for K4 (inter-subset edges)
    # We need a 3-coloring of K4 with no monochromatic triangle.
    # K4 has 4 vertices and 6 edges. With 3 colors, we can use:
    #   (0,1)->0, (0,2)->1, (0,3)->2, (1,2)->2, (1,3)->1, (2,3)->0
    # Check: triangles in K4 are (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    #   (0,1,2): colors 0,1,2 - no mono
    #   (0,1,3): colors 0,2,1 - no mono
    #   (0,2,3): colors 1,2,0 - no mono
    #   (1,2,3): colors 2,1,0 - no mono
    # Good!
    inter_color = {
        (0, 1): 0, (0, 2): 1, (0, 3): 2,
        (1, 2): 2, (1, 3): 1, (2, 3): 0
    }
    
    # Build edge index
    m = n * (n - 1) // 2
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    
    colors = [0] * m
    
    # Assign colors
    for i in range(n):
        for j in range(i + 1, n):
            si = vert_to_sub[i]
            sj = vert_to_sub[j]
            if si == sj:
                # Same subset: use that subset's color
                c = si
            else:
                # Different subsets: use inter-subset coloring
                a, b = sorted((si, sj))
                c = inter_color[(a, b)]
            colors[idx[i][j]] = c
    
    return colors, subsets

def _solve_n(n, rng, deadline):
    """
    Attempt to find a valid 4-coloring of K_n with no monochromatic triangles.
    Uses block construction as starting point, then local search.
    """
    m = n * (n - 1) // 2
    if m == 0:
        return []
    
    # Precompute edge indices
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    
    # Try block construction first
    block_result = _block_coloring(n, rng)
    if block_result is not None:
        colors, subsets = block_result
    else:
        # Fallback to random
        colors = [rng.randint(0, 3) for _ in range(m)]
        subsets = None
    
    # Initialize masks
    masks = [[0] * 4 for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = colors[idx[u][v]]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)
    
    # Calculate initial triangle count using bitmask
    tri_count = 0
    for u in range(n):
        for c in range(4):
            mask_u = masks[u][c]
            if mask_u == 0:
                continue
            # Count pairs (v,w) in mask_u such that (v,w) is also color c
            # For each v in mask_u, count w in (mask_u & masks[v][c])
            # But this double-counts each pair, so divide by 2
            # And each triangle is counted 3 times (once per vertex)
            count = 0
            v_mask = mask_u
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                count += (masks[v][c] & mask_u).bit_count()
                v_mask ^= v_bit
            tri_count += count // 2
    
    # Each triangle counted 3 times (once per vertex), so actual count is tri_count / 3
    # But we're summing over all u and c, so each triangle appears 3 times total
    actual_tri = tri_count // 3
    
    if actual_tri == 0:
        return colors
    
    # Simulated Annealing / Local Search
    T = 1.0
    cooling = 0.9995
    steps = 0
    max_steps = 500000
    
    while T > 0.001 and time.time() < deadline - 2.0:
        # Pick random edge
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        i_idx = idx[u][v]
        old_c = colors[i_idx]
        new_c = rng.randint(0, 3)
        if old_c == new_c:
            continue
        
        # Calculate delta in triangle count
        lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
        gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
        
        delta = gained - lost
        
        # Accept if improvement or with probability exp(-delta/T)
        accept = False
        if delta < 0:
            accept = True
        else:
            if delta > 50 * T:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta / T)
        
        if accept:
            colors[i_idx] = new_c
            actual_tri += delta // 3  # Approximate update; we'll recompute if needed
            
            # Update masks
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][new_c] |= (1 << v)
            masks[v][new_c] |= (1 << u)
            
            # Recompute actual triangle count periodically for accuracy
            if steps % 100 == 0:
                tri_count = 0
                for u2 in range(n):
                    for c2 in range(4):
                        mask_u2 = masks[u2][c2]
                        if mask_u2 == 0:
                            continue
                        count2 = 0
                        v_mask2 = mask_u2
                        while v_mask2:
                            v_bit2 = v_mask2 & (-v_mask2)
                            v2 = v_bit2.bit_length() - 1
                            count2 += (masks[v2][c2] & mask_u2).bit_count()
                            v_mask2 ^= v_bit2
                        tri_count += count2 // 2
                actual_tri = tri_count // 3
                
                if actual_tri == 0:
                    return colors
            
            T *= cooling
            steps += 1
    
    # Final check
    tri_count = 0
    for u in range(n):
        for c in range(4):
            mask_u = masks[u][c]
            if mask_u == 0:
                continue
            count = 0
            v_mask = mask_u
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                count += (masks[v][c] & mask_u).bit_count()
                v_mask ^= v_bit
            tri_count += count // 2
    
    if tri_count // 3 == 0:
        return colors
    
    return None

def construct():
    start = time.time()
    deadline = start + 115.0
    rng = random.Random(42)
    
    best = _baseline32()
    best_n = 32
    
    current_n = 33
    while current_n <= 62:
        if time.time() > deadline - 5.0:
            break
        
        # Allocate time for this n
        sub_deadline = time.time() + min(10.0, deadline - time.time() - 2.0)
        
        res = None
        while time.time() < sub_deadline:
            res = _solve_n(current_n, rng, sub_deadline)
            if res is not None:
                break
        
        if res is not None:
            best = {'n': current_n, 'colors': res}
            best_n = current_n
            current_n += 1
        else:
            break
    
    # Final validation
    if time.time() < deadline - 1.0:
        n = best['n']
        colors = best['colors']
        if len(colors) != n * (n - 1) // 2:
            best = _baseline32()
        else:
            idx = [[0] * n for _ in range(n)]
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    idx[i][j] = k
                    idx[j][i] = k
                    k += 1
            
            valid = True
            for u in range(n):
                for v in range(u + 1, n):
                    for w in range(v + 1, n):
                        c1 = colors[idx[u][v]]
                        if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                best = _baseline32()
    
    return best

# EVOLVE-BLOCK-END
