"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _mul(a, b):
    r = 0
    for i in range(4):
        if b >> i & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if r >> i & 1:
            r ^= 0b10011 << (i - 4)
    return r

def _build_k32_colors():
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return colors

def _build_color_bitmasks(colors, n):
    """
    Build bitmasks: for each vertex v and color c, a bitmask of vertices u
    such that edge (v,u) has color c.
    Returns list of 4 lists, each of length n, of int bitmasks.
    """
    # masks[v][c] = bitmask of u where edge(v,u) has color c
    masks = [[0]*4 for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
            c = colors[idx]
            masks[i][c] |= (1 << j)
            masks[j][c] |= (1 << i)
    return masks

def _check_extendable_fast(masks, n, new_edges):
    """
    Fast check: can we add vertex n with edge colors new_edges[0..n-1]?
    For each color c, let S_c = {i : new_edges[i] == c}.
    For any pair (a,b) in S_c, we need edge(a,b) != c.
    Equivalently, for each c, S_c must be an independent set in the c-colored subgraph.
    """
    for c in range(4):
        # Build bitmask of vertices connected to new vertex with color c
        mask_c = 0
        for i in range(n):
            if new_edges[i] == c:
                mask_c |= (1 << i)
        # For each vertex i in S_c, check if any other vertex j in S_c has edge(i,j) color c
        # i.e., masks[i][c] & mask_c should have only bit i set (or be 0)
        m = mask_c
        while m:
            # Get lowest set bit
            lb = m & (-m)
            i = lb.bit_length() - 1
            # Check if masks[i][c] has any bit in mask_c other than possibly i
            conflict = masks[i][c] & mask_c
            if conflict & ~lb:
                return False
            m ^= lb
    return True

def _try_add_vertex_fast(masks, colors, n, rng, deadline, max_attempts=50000):
    """
    Try to add one vertex using fast bitmask check.
    Returns (new_colors, new_n, new_masks) on success, (None, n, masks) on failure.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        new_edges = [rng.randint(0, 3) for _ in range(n)]
        if _check_extendable_fast(masks, n, new_edges):
            new_colors = colors + new_edges
            new_n = n + 1
            # Rebuild masks (could be optimized incrementally)
            new_masks = _build_color_bitmasks(new_colors, new_n)
            return new_colors, new_n, new_masks
    return None, n, masks

def _local_search_extendable(masks, colors, n, rng, deadline):
    """
    Local search: flip random edges to make the coloring more extendable.
    Measure extendability by how many random new-edge assignments pass the check.
    """
    if time.time() > deadline:
        return masks, colors, n
    
    cur_masks = masks
    cur_colors = colors[:]
    cur_n = n
    
    def score_extendability(masks, n, rng, trials=200):
        """Score: fraction of random edge assignments that are valid."""
        count = 0
        for _ in range(trials):
            if time.time() > deadline:
                break
            new_edges = [rng.randint(0, 3) for _ in range(n)]
            if _check_extendable_fast(masks, n, new_edges):
                count += 1
        return count
    
    # First, try to extend from current state
    test_colors, test_n, test_masks = _try_add_vertex_fast(cur_masks, cur_colors, cur_n, rng, deadline, max_attempts=2000)
    if test_n > cur_n:
        return test_masks, test_colors, test_n
    
    # Local search: flip edges to improve extendability
    best_score = score_extendability(cur_masks, cur_n, rng, trials=100)
    cur_masks = _build_color_bitmasks(cur_colors, cur_n)
    
    for step in range(200):
        if time.time() > deadline - 5:
            break
        
        i = rng.randint(0, cur_n - 2)
        j = rng.randint(i + 1, cur_n - 1)
        idx = i * (cur_n - 1) - i * (i - 1) // 2 + (j - i - 1)
        old_color = cur_colors[idx]
        new_color = rng.randint(0, 3)
        if new_color == old_color:
            continue
        
        cur_colors[idx] = new_color
        
        # Check validity: no monochromatic triangle involving edge (i,j)
        valid = True
        for k in range(cur_n):
            if k == i or k == j:
                continue
            idx_ik = i * (cur_n - 1) - i * (i - 1) // 2 + (k - i - 1) if k > i else k * (cur_n - 1) - k * (k - 1) // 2 + (i - k - 1)
            idx_jk = j * (cur_n - 1) - j * (j - 1) // 2 + (k - j - 1) if k > j else k * (cur_n - 1) - k * (k - 1) // 2 + (j - k - 1)
            c_ik = cur_colors[idx_ik]
            c_jk = cur_colors[idx_jk]
            if c_ik == new_color and c_jk == new_color:
                valid = False
                break
        
        if not valid:
            cur_colors[idx] = old_color
            continue
        
        # Rebuild masks and score
        test_masks = _build_color_bitmasks(cur_colors, cur_n)
        test_score = score_extendability(test_masks, cur_n, rng, trials=100)
        
        if test_score > best_score:
            best_score = test_score
            cur_masks = test_masks
        else:
            cur_colors[idx] = old_color
            cur_masks = _build_color_bitmasks(cur_colors, cur_n)
    
    # Try to extend from improved state
    test_colors, test_n, test_masks = _try_add_vertex_fast(cur_masks, cur_colors, cur_n, rng, deadline, max_attempts=5000)
    if test_n > cur_n:
        return test_masks, test_colors, test_n
    
    return cur_masks, cur_colors, cur_n

def construct():
    deadline = time.time() + 140
    rng = random.Random(42)
    
    # Start with 32-vertex coloring
    base_colors = _build_k32_colors()
    best_colors = base_colors[:]
    best_n = 32
    best_masks = _build_color_bitmasks(base_colors, 32)
    
    current_colors = base_colors[:]
    current_n = 32
    current_masks = best_masks[:]
    
    while current_n < 62 and time.time() < deadline - 15:
        new_colors, new_n, new_masks = _try_add_vertex_fast(
            current_masks, current_colors, current_n, rng, deadline, max_attempts=30000
        )
        if new_colors is None:
            # Try local search
            if time.time() < deadline - 25:
                improved_masks, improved_colors, improved_n = _local_search_extendable(
                    current_masks, current_colors, current_n, rng, deadline
                )
                if improved_n > current_n:
                    current_colors = improved_colors
                    current_n = improved_n
                    current_masks = improved_masks
                    if current_n > best_n:
                        best_colors = current_colors[:]
                        best_n = current_n
                        best_masks = current_masks
                    continue
            break
        
        current_colors = new_colors
        current_n = new_n
        current_masks = new_masks
        
        if current_n > best_n:
            best_colors = current_colors[:]
            best_n = current_n
            best_masks = current_masks
    
    # Multiple random restarts from base
    if time.time() < deadline - 10:
        for seed in range(1, 30):
            if time.time() > deadline - 3:
                break
            rng2 = random.Random(seed * 7919 + 13)
            cur_c = base_colors[:]
            cur_n = 32
            cur_m = _build_color_bitmasks(cur_c, 32)
            
            for _ in range(15):
                if time.time() > deadline - 3:
                    break
                new_c, new_n, new_m = _try_add_vertex_fast(
                    cur_m, cur_c, cur_n, rng2, deadline, max_attempts=8000
                )
                if new_c is None:
                    break
                cur_c = new_c
                cur_n = new_n
                cur_m = new_m
            
            if cur_n > best_n:
                best_colors = cur_c[:]
                best_n = cur_n
                best_masks = cur_m
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
