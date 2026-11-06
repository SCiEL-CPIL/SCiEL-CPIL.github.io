import time
import random
import math
import sys

_START_TIME = time.time()
# Safety margin: evaluator kills at 150s. We aim to finish by 135s to be safe.
_DEADLINE = 135.0 

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
            # Only consider neighbors j > i to avoid double counting
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

def _geometric_init(n):
    """
    Geometric initialization: Place vertices on a 4-dimensional hypercube (or similar structure)
    and assign colors based on the Hamming distance or coordinate differences.
    This creates a structured starting point that is often better than pure random or simple greedy.
    """
    m = n * (n - 1) // 2
    colors = [0] * m
    
    # Map vertices to points in {0, 1, 2, 3}^4 (hypercube-like)
    # We use a simple mapping for n up to 62.
    # For n=50, we can map to a subset of the 4D hypercube or just use coordinate-wise logic.
    # Let's use a strategy: color based on the first coordinate where the two vertices differ.
    # Vertices are represented by their index in a specific ordering.
    
    # To make it work for arbitrary n, we can embed vertices into a 4D space.
    # If n <= 16, we can use a 4-bit representation.
    # If n > 16, we can use a 3-bit representation for the first 8, etc.
    # Simpler: Just use the binary representation of the index.
    # Color = (i_bit_xor_j_bit) mod 4? No, that's 1 bit per color.
    
    # Better Geometric Heuristic:
    # Partition vertices into 4 groups based on i % 4.
    # If i % 4 == j % 4: color = (i // 4) ^ (j // 4) mod 3 (using 3 colors for intra-group)
    # Else: color = (i % 4 + j % 4) % 4 (using 4 colors for inter-group)
    
    for i in range(n):
        for j in range(i + 1, n):
            if i % 4 == j % 4:
                # Intra-group: use a 3-coloring logic on the reduced indices
                c = ((i // 4) ^ (j // 4)) % 3
            else:
                # Inter-group: use the group indices
                c = (i % 4 + j % 4) % 4
            idx = _compute_index(n, i, j)
            colors[idx] = c
            
    return colors

def _greedy_init(n):
    """
    Greedy initialization: build vertex by vertex, choosing edge colors
    to minimize immediate monochromatic triangle creation.
    """
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                # Count triangles formed if we add edge (u,v) with color c
                # This is the number of common neighbors w < u of u and v in color c
                cost = (adj[c][u] & ((1 << u) - 1)).bit_count()
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
    
    return colors

def _sa_search(n, time_budget, init_strategy='greedy'):
    """
    Simulated annealing to find a valid 4-coloring of K_n.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    # Precompute index-to-edge mapping for speed
    idx_to_edge = [None] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge[idx] = (i, j)
            idx += 1
    
    best_result = None
    best_tri = float('inf')
    
    # Multiple restarts
    for restart in range(1000):
        if time.time() - start_time > time_budget:
            break
        
        # Alternate initialization strategies
        if init_strategy == 'mixed' or restart % 2 == 0:
            if restart % 4 == 0:
                colors = _geometric_init(n)
            else:
                colors = _greedy_init(n)
        else:
            colors = [random.randint(0, 3) for _ in range(m)]
        
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_result = colors[:]
        
        if tri_count == 0:
            return colors
        
        # Simulated Annealing Parameters
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        accepted = 0
        total_steps = 0
        steps_since_improve = 0
        
        while True:
            if time.time() - start_time > time_budget:
                break
            
            # Pick a random edge
            ri = random.randint(0, m - 1)
            old_c = colors[ri]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            u, v = idx_to_edge[ri]
            
            # Compute delta in triangle count
            removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
            added = (adj[new_c][u] & adj[new_c][v]).bit_count()
            delta = added - removed
            
            total_steps += 1
            
            if delta <= 0:
                # Accept if improvement or equal
                colors[ri] = new_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                tri_count += delta
                accepted += 1
                steps_since_improve = 0
                
                if tri_count < best_tri:
                    best_tri = tri_count
                    best_result = colors[:]
                    if tri_count == 0:
                        return colors
            else:
                # Accept with probability exp(-delta/T)
                if T > T_min:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        colors[ri] = new_c
                        adj[old_c][u] &= ~(1 << v)
                        adj[old_c][v] &= ~(1 << u)
                        adj[new_c][u] |= (1 << v)
                        adj[new_c][v] |= (1 << u)
                        tri_count += delta
                        accepted += 1
                        steps_since_improve = 0
                else:
                    steps_since_improve += 1
            
            T *= cooling
            
            # Restart condition: if we've taken many steps but very few were accepted
            if total_steps > 1000 and accepted < total_steps * 0.01:
                break
            
            # Also restart if stuck for too long without improvement
            if steps_since_improve > 5000:
                break
                
    return best_result if best_tri == 0 else None

def _island_sa(n, time_budget, seed):
    """
    Island model SA with geometric initialization mixed in.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    rng = random.Random(seed)
    
    idx_to_edge = [None] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge[idx] = (i, j)
            idx += 1

    num_islands = 4
    islands = []
    for k in range(num_islands):
        if k % 2 == 0:
            colors = _geometric_init(n)
        else:
            colors = _greedy_init(n)
        
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)
        
        islands.append({
            'colors': colors,
            'adj': adj,
            'tri_count': tri_count,
            'best_colors': colors[:],
            'best_tri': tri_count
        })
    
    best_overall_tri = float('inf')
    best_overall_colors = None
    
    migration_interval = 500
    step_count = 0
    
    while True:
        if time.time() - start_time > time_budget:
            break
        
        island_idx = rng.randint(0, num_islands - 1)
        island = islands[island_idx]
        
        colors = island['colors']
        adj = island['adj']
        tri_count = island['tri_count']
        
        T = 2.0
        T_min = 0.01
        cooling = 0.999
        
        ri = rng.randint(0, m - 1)
        old_c = colors[ri]
        new_c = rng.randint(0, 3)
        if new_c == old_c:
            continue
        
        u, v = idx_to_edge[ri]
        
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
            
            if tri_count < island['best_tri']:
                island['best_tri'] = tri_count
                island['best_colors'] = colors[:]
                
                if tri_count < best_overall_tri:
                    best_overall_tri = tri_count
                    best_overall_colors = colors[:]
                    if tri_count == 0:
                        return colors
        else:
            if T > T_min:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    colors[ri] = new_c
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    tri_count += delta
            
            T *= cooling
        
        step_count += 1
        
        if step_count % migration_interval == 0:
            best_island = min(islands, key=lambda x: x['best_tri'])
            worst_island = max(islands, key=lambda x: x['best_tri'])
            
            if best_island is not worst_island:
                worst_island['colors'] = best_island['best_colors'][:]
                worst_island['adj'] = _build_adj(worst_island['colors'], n)
                worst_island['tri_count'] = worst_island['best_tri']
                
                if worst_island['best_tri'] < best_overall_tri:
                    best_overall_tri = worst_island['best_tri']
                    best_overall_colors = worst_island['best_colors'][:]
                    if best_overall_tri == 0:
                        return best_overall_colors

    return best_overall_colors if best_overall_tri == 0 else None

def construct():
    try:
        best = _seed_32()
        best_n = 32
        
        elapsed = time.time() - _START_TIME
        remaining = _DEADLINE - elapsed
        if remaining < 5:
            return best
        
        # Targets: Try 50 first (record), then fall back
        targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 40, 38, 36, 34]
        
        for target_n in targets:
            if target_n <= best_n:
                continue
            
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining < 3:
                break
            
            # Allocate budget
            if target_n == 50:
                sa_budget = min(remaining - 20, 50)
                island_budget = min(remaining - 20, 40)
            elif target_n >= 45:
                sa_budget = min(remaining - 10, 25)
                island_budget = min(remaining - 10, 15)
            else:
                sa_budget = min(remaining - 5, 10)
                island_budget = min(remaining - 5, 5)
            
            if sa_budget < 2 and island_budget < 2:
                continue
                
            # 1. Try Simulated Annealing with mixed initialization (Geometric + Greedy)
            result = _sa_search(target_n, sa_budget, init_strategy='mixed')
            if result is not None:
                return {'n': target_n, 'colors': result}
            
            # 2. Try Island SA (Structural change: parallel islands with migration)
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining > 5 and island_budget > 1:
                result = _island_sa(target_n, island_budget, seed=target_n)
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
