"""Combine: K32 seed floor + forward-checking backtracking with random restarts."""
import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 110.0  # safety margin under the 120 s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32 (guaranteed valid floor)."""
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
    return colors

def _compute_index(n, i, j):
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
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
    """Find one monochromatic triangle, return (c, i, j, k) or None."""
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
                    return (c, i, j, k)
    return None

def _greedy_init(n, seed_colors_32=None):
    """
    Build vertex-by-vertex coloring. If seed_colors_32 provided and n>32,
    start with the K32 coloring and extend.
    """
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    
    if n <= 32:
        if seed_colors_32 is not None and len(seed_colors_32) == n * (n-1) // 2:
            colors = seed_colors_32[:]
            adj = _build_adj(colors, n)
        else:
            for v in range(1, n):
                for u in range(v):
                    best_c = 0
                    min_cost = float('inf')
                    for c in range(4):
                        cost = (adj[c][u] & ((1 << u) - 1)).bit_count()
                        if cost < min_cost:
                            min_cost = cost
                            best_c = c
                    idx = _compute_index(n, u, v)
                    colors[idx] = best_c
                    adj[best_c][u] |= (1 << v)
                    adj[best_c][v] |= (1 << u)
    else:
        # Start with K32 base
        base_n = 32
        if seed_colors_32 is not None:
            base_colors = seed_colors_32[:]
        else:
            base_colors = _seed_32()
        
        # Copy base coloring
        base_m = base_n * (base_n - 1) // 2
        for idx in range(base_m):
            colors[idx] = base_colors[idx]
        
        # Build adj for base
        for i in range(base_n):
            for j in range(i + 1, base_n):
                idx = _compute_index(n, i, j)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        # Extend to n
        for v in range(base_n, n):
            for u in range(v):
                best_c = 0
                min_cost = float('inf')
                for c in range(4):
                    cost = (adj[c][u] & ((1 << u) - 1)).bit_count()
                    if cost < min_cost:
                        min_cost = cost
                        best_c = c
                idx = _compute_index(n, u, v)
                colors[idx] = best_c
                adj[best_c][u] |= (1 << v)
                adj[best_c][v] |= (1 << u)
    
    return colors, adj

def _forward_checking_search(n, time_budget, seed_colors_32=None):
    """
    Structural change: Forward-checking backtracking.
    Instead of minimizing triangle count, we directly target zero triangles.
    When a monochromatic triangle exists, we MUST change one of its edges.
    We use forward checking: after changing an edge, we verify no new
    monochromatic triangles were created by that specific change.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    idx_to_edge = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_tri = float('inf')
    best_result = None
    
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget or _time_left() < 1.0:
            break
        
        colors, adj = _greedy_init(n, seed_colors_32 if n > 32 else None)
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_result = colors[:]
        if tri_count == 0:
            return colors, 0
        
        # Forward-checking hill climb
        T = 3.0
        for step in range(8000):
            if time.time() - start_time > time_budget or _time_left() < 1.0:
                break
            
            tri = _find_one_mono_triangle(adj, n)
            if tri is None:
                tri_count = _count_mono_triangles(adj, n)
                if tri_count == 0:
                    return colors, 0
                break
            
            c, i, j, k = tri
            # The three edges of this monochromatic triangle
            edge_ij = _compute_index(n, i, j)
            edge_ik = _compute_index(n, i, k)
            edge_jk = _compute_index(n, j, k)
            
            # For each edge, find the best new color that breaks the triangle
            # and doesn't create new monochromatic triangles
            candidates = []
            for idx, u, v in [(edge_ij, i, j), (edge_ik, i, k), (edge_jk, j, k)]:
                old_c = colors[idx]
                if old_c != c:
                    continue
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    # Check if this change creates new monochromatic triangles
                    # New triangles involving (u,v) with color new_c
                    # A new mono triangle would be (u,v,w) where (u,w)=new_c and (v,w)=new_c
                    # Count how many such w exist
                    new_conflicts = (adj[new_c][u] & adj[new_c][v]).bit_count()
                    # Remove the triangle we're breaking
                    # The old triangles involving (u,v) with color old_c
                    old_conflicts = (adj[old_c][u] & adj[old_c][v]).bit_count()
                    # Net change in triangle count
                    delta = new_conflicts - old_conflicts
                    # We prefer changes that reduce total triangles
                    candidates.append((delta, idx, old_c, new_c, u, v))
            
            if not candidates:
                # No valid move, pick random
                idx, u, v = random.choice([(edge_ij, i, j), (edge_ik, i, k), (edge_jk, j, k)])
                old_c = colors[idx]
                new_c = random.choice([c2 for c2 in range(4) if c2 != old_c])
                delta = (adj[new_c][u] & adj[new_c][v]).bit_count() - (adj[old_c][u] & adj[old_c][v]).bit_count()
            else:
                # Sort by delta (prefer negative)
                candidates.sort(key=lambda x: x[0])
                # With some probability, accept a worse move (simulated annealing)
                if T > 0.5 and random.random() < 0.3:
                    # Pick a random candidate
                    delta, idx, old_c, new_c, u, v = random.choice(candidates)
                else:
                    delta, idx, old_c, new_c, u, v = candidates[0]
            
            # Apply the change
            colors[idx] = new_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            tri_count += delta
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_result = colors[:]
                if tri_count == 0:
                    return colors, 0
            
            T *= 0.998
            if T < 0.2:
                T = 0.2
    
    return best_result, best_tri

def construct():
    # Guaranteed floor: K32 valid coloring.
    floor_colors = _seed_32()
    floor_n = 32

    # Try to improve via forward-checking search
    for n in (33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50):
        if _time_left() < 12.0:
            break
        budget = min(6.0, _time_left() - 8.0)
        if budget <= 0:
            break
        
        colors, tri = _forward_checking_search(n, time_budget=budget, seed_colors_32=floor_colors)
        if colors is not None and tri == 0:
            return {'n': n, 'colors': colors}

    return {'n': floor_n, 'colors': floor_colors}
