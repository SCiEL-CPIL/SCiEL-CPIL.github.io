import time
import random
import sys
import math

_START_TIME = time.time()
_DEADLINE = 112.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
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
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return {'n': 32, 'colors': colors}

def _compute_index(n, i, j):
    """Compute the index of pair (i,j) in lexicographic edge ordering."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for a given coloring."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    """Count total monochromatic triangles using bitsets."""
    count = 0
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _find_one_mono_triangle(adj, n):
    """Find one monochromatic triangle, return (i, j, k, c) or None."""
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                if common:
                    k_bit = common & (-common)
                    k = k_bit.bit_length() - 1
                    return (i, j, k, c)
    return None

def _greedy_init(n):
    """
    Greedy initialization: build vertex by vertex, choosing edge colors
    to maximize slack (distance to monochromatic triangle).
    """
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    
    for v in range(1, n):
        # For each u < v, choose color for edge (u, v)
        # Slack for color c on edge (u,v): number of w < v, w != u such that
        #   adj[c][u] has w AND adj[c][v] will have w after coloring
        # We want to minimize the number of "conflicts" - pairs (w) where both
        # (u,w) and (v,w) would be color c.
        
        # First pass: for each u, find the best color that minimizes conflicts
        best_colors = {}
        for u in range(v):
            best_c = 0
            min_conflicts = float('inf')
            for c in range(4):
                # Conflicts: w < v, w != u, where (u,w) is color c
                # After we color (v,w), if we also color (v,w) with c, we get a triangle
                # So conflicts = number of w < v, w != u, with (u,w) color c
                # that we would need to NOT color as c for (v,w)
                conflicts = 0
                for w in range(v):
                    if w == u:
                        continue
                    idx_w = _compute_index(n, min(u,w), max(u,w))
                    if colors[idx_w] == c:
                        conflicts += 1
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_c = c
            best_colors[u] = best_c
        
        # Second pass: assign colors, but try to be consistent
        # Use a more global approach: try all 4^v assignments is too expensive
        # Instead, greedily assign with tie-breaking by randomization
        for u in range(v):
            c = best_colors[u]
            idx = _compute_index(n, u, v)
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
    
    return colors

def _greedy_construction(n, seed=None):
    """
    Greedy randomized construction: build a valid 4-coloring of K_n
    by iteratively adding vertices, choosing edge colors to maximize
    the minimum slack across all color classes.
    
    Slack for a potential new vertex: for each existing vertex u and color c,
    the number of pairs (u, w) with color c where w is also a neighbor of u in color c.
    Higher slack means more room to add future vertices.
    
    Returns colors list if a valid coloring is found, None otherwise.
    """
    if seed is not None:
        random.seed(seed)
    
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    
    # Start with K1 (trivial)
    # Build up to K_n
    for v in range(1, n):
        # Determine colors for edges (u, v) for u = 0..v-1
        # For each u, we need to pick a color c such that no triangle is formed.
        # A triangle (u, w, v) in color c exists if:
        #   - (u, w) is color c
        #   - (w, v) is color c (to be determined)
        #   - (u, v) is color c (to be determined)
        
        # Strategy: for each u, compute the set of "forbidden" colors.
        # Color c is forbidden for edge (u,v) if there exists w < v, w != u,
        # such that (u,w) is color c AND (w,v) is already color c.
        # But (w,v) isn't determined yet for w < v... 
        
        # Better approach: process vertices in a smart order.
        # For each u, the constraint is: if we color (u,v) with color c,
        # then for all w < v, w != u, we CANNOT color (w,v) with color c
        # if (u,w) is already color c.
        
        # So: for each u, let F_u = {c : exists w < v, w != u, (u,w) is color c}
        # Then if we color (u,v) with c in F_u, we must ensure (w,v) != c for all such w.
        
        # This is a constraint satisfaction problem. Let's use a greedy approach:
        # For each u, pick the color c that minimizes the number of constraints
        # it imposes on other edges (w,v).
        
        # Compute for each u, each color c: how many w < v, w != u have (u,w) = c
        # This is the "pressure" that coloring (u,v)=c puts on other edges.
        
        pressure = {}
        for u in range(v):
            pressure[u] = [0, 0, 0, 0]
            for w in range(v):
                if w == u:
                    continue
                idx_w = _compute_index(n, min(u,w), max(u,w))
                c_w = colors[idx_w]
                pressure[u][c_w] += 1
        
        # Now, for each u, choose color c that minimizes pressure[u][c]
        # This minimizes the number of other edges (w,v) that cannot use color c.
        chosen_colors = {}
        for u in range(v):
            best_c = 0
            min_p = float('inf')
            for c in range(4):
                if pressure[u][c] < min_p:
                    min_p = pressure[u][c]
                    best_c = c
            chosen_colors[u] = best_c
        
        # Assign colors
        for u in range(v):
            c = chosen_colors[u]
            idx = _compute_index(n, u, v)
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
        
        # Check for monochromatic triangles involving vertex v
        # For each pair (u, w) with u, w < v, check if all three edges are same color
        has_triangle = False
        for u in range(v):
            for w in range(u + 1, v):
                idx_uw = _compute_index(n, u, w)
                idx_uv = _compute_index(n, u, v)
                idx_wv = _compute_index(n, w, v)
                if colors[idx_uw] == colors[idx_uv] == colors[idx_wv]:
                    has_triangle = True
                    break
            if has_triangle:
                break
        
        if has_triangle:
            # Backtrack: try different color choices
            # This is expensive; instead, return failure and let caller restart
            return None
    
    # Final verification
    adj_final = _build_adj(colors, n)
    tri = _find_one_mono_triangle(adj_final, n)
    if tri is not None:
        return None
    
    return colors

def _local_search(n, time_budget):
    """
    Local search to find a valid 4-coloring of K_n.
    Uses greedy construction initialization and optimizes total triangle count.
    Returns colors list if successful, None otherwise.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    max_restarts = 200
    best_result = None
    best_tri_count = float('inf')
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Try greedy construction first
        colors = _greedy_construction(n, seed=restart * 7919 + 13)
        if colors is not None:
            return colors
        
        # If greedy construction fails, fall back to greedy init + local search
        colors = _greedy_init(n)
        adj = _build_adj(colors, n)
        
        # Track current triangle count
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri_count:
            best_tri_count = tri_count
            best_result = colors[:]
        
        if tri_count == 0:
            return colors
        
        # Local search loop
        no_improve = 0
        max_no_improve = 50
        
        while True:
            if time.time() - start_time > time_budget:
                break
            
            tri = _find_one_mono_triangle(adj, n)
            if tri is None:
                return colors
            
            i, j, k, c = tri
            
            # Choose an edge to recolor. Try all 3 edges and pick the best.
            edges = [(i, j), (i, k), (j, k)]
            best_edge = None
            best_new_c = -1
            best_delta = float('inf')
            
            for u, v in edges:
                for new_c in range(4):
                    if new_c == c:
                        continue
                    removed = (adj[c][u] & adj[c][v]).bit_count()
                    added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                    delta = added - removed
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_edge = (u, v)
                        best_new_c = new_c
            
            if best_edge is None:
                break
            
            u, v = best_edge
            idx = _compute_index(n, u, v)
            
            removed = (adj[c][u] & adj[c][v]).bit_count()
            added = (adj[best_new_c][u] & adj[best_new_c][v]).bit_count()
            tri_count += added - removed
            
            colors[idx] = best_new_c
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            adj[best_new_c][u] |= (1 << v)
            adj[best_new_c][v] |= (1 << u)
            
            if tri_count < best_tri_count:
                best_tri_count = tri_count
                best_result = colors[:]
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                break
        
        # Perturb to escape local minima
        if no_improve > max_no_improve and m > 0:
            for _ in range(5):
                ri = random.randint(0, m - 1)
                # Map index to (u, v)
                u = 0
                remaining = ri
                while u < n - 1:
                    edges_from_u = n - 1 - u
                    if remaining < edges_from_u:
                        v = u + 1 + remaining
                        break
                    remaining -= edges_from_u
                    u += 1
                else:
                    u = n - 2
                    v = n - 1
                
                old_c = colors[ri]
                new_c = random.randint(0, 3)
                if new_c != old_c:
                    colors[ri] = new_c
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
    
    return best_result if best_tri_count == 0 else None

def _simulated_annealing(n, time_budget):
    """
    Simulated annealing approach for finding a valid 4-coloring of K_n.
    Uses greedy construction as initial state.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    # Start with greedy construction
    colors = _greedy_construction(n, seed=42)
    if colors is not None:
        return colors
    
    # Fallback to greedy init
    colors = _greedy_init(n)
    adj = _build_adj(colors, n)
    tri_count = _count_mono_triangles(adj, n)
    
    if tri_count == 0:
        return colors
    
    best_colors = colors[:]
    best_tri_count = tri_count
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    # Precompute index-to-edge mapping for speed
    idx_to_edge = []
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge.append((i, j))
    
    while True:
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random edge to change
        ri = random.randint(0, m - 1)
        u, v = idx_to_edge[ri]
        old_c = colors[ri]
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
        
        # Compute delta in triangle count
        removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
        added = (adj[new_c][u] & adj[new_c][v]).bit_count()
        delta = added - removed
        
        if delta <= 0:
            colors[ri] = new_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            tri_count += delta
            
            if tri_count < best_tri_count:
                best_tri_count = tri_count
                best_colors = colors[:]
                if tri_count == 0:
                    return colors
        else:
            if T > T_min:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    colors[ri] = new_c
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    tri_count += delta
        
        T *= cooling
    
    return best_colors if best_tri_count == 0 else None

def construct():
    try:
        best = _seed_32()
        best_n = 32
        
        # Try greedy construction with random restarts for larger n
        # Targets: 50, 48, 45, 42, 40, 38, 36, 34
        
        elapsed = time.time() - _START_TIME
        remaining = _DEADLINE - elapsed
        if remaining < 5:
            return best
        
        targets = [50, 48, 45, 42, 40, 38, 36, 34]
        
        for target_n in targets:
            if target_n <= best_n:
                continue
            
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining < 3:
                break
            
            # Allocate budget: more time for larger targets
            if target_n >= 48:
                budget = min(remaining - 3, 40)
            elif target_n >= 44:
                budget = min(remaining - 3, 25)
            elif target_n >= 40:
                budget = min(remaining - 3, 15)
            else:
                budget = min(remaining - 3, 8)
            
            if budget < 2:
                break
            
            # Strategy: try greedy construction with multiple random seeds first
            # This is fast per attempt, so we can try many
            num_greedy_tries = int(budget * 3)  # ~3 tries per second
            for attempt in range(num_greedy_tries):
                if time.time() - _START_TIME > _DEADLINE - 2:
                    break
                result = _greedy_construction(target_n, seed=attempt * 31 + 7)
                if result is not None:
                    return {'n': target_n, 'colors': result}
            
            # If greedy construction didn't work, try simulated annealing
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining > 5:
                sa_budget = min(remaining - 5, 15)
                if sa_budget > 2:
                    result = _simulated_annealing(target_n, sa_budget)
                    if result is not None:
                        return {'n': target_n, 'colors': result}
            
            # Try local search if time remains
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining > 5:
                ls_budget = min(remaining - 5, 10)
                if ls_budget > 2:
                    result = _local_search(target_n, ls_budget)
                    if result is not None:
                        return {'n': target_n, 'colors': result}
        
        return best
    except Exception as e:
        try:
            return _seed_32()
        except:
            return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
