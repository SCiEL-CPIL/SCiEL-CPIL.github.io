import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 115.0

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
    to minimize immediate monochromatic triangle creation.
    """
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    
    for v in range(1, n):
        # For each u < v, choose color for edge (u, v)
        # Cost of color c for edge (u, v): number of w < u with (u, w) color c
        # This is a greedy heuristic that ignores coupling between edges (u,v) and (w,v)
        best_colors = {}
        for u in range(v):
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                # Count neighbors of u in color c that are < u (i.e., already colored)
                cost = (adj[c][u] & ((1 << u) - 1)).bit_count()
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            best_colors[u] = best_c
        
        # Assign colors
        for u in range(v):
            c = best_colors[u]
            idx = _compute_index(n, u, v)
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
    
    return colors

def _local_search(n, time_budget):
    """
    Local search to find a valid 4-coloring of K_n.
    Uses greedy initialization and optimizes total triangle count.
    Returns colors list if successful, None otherwise.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    max_restarts = 500
    best_result = None
    best_tri_count = float('inf')
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Initialize with greedy coloring
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
        max_no_improve = 100
        
        while True:
            if time.time() - start_time > time_budget:
                return None
            
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
                    # Estimate delta in triangle count
                    # Remove triangles involving edge (u,v) in color c
                    removed = (adj[c][u] & adj[c][v]).bit_count()
                    # Add triangles involving edge (u,v) in color new_c
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
            
            # Update triangle count estimate
            removed = (adj[c][u] & adj[c][v]).bit_count()
            added = (adj[best_new_c][u] & adj[best_new_c][v]).bit_count()
            tri_count += added - removed
            
            # Recolor
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
        
        # If stuck, add some randomness to escape local minima
        if no_improve > max_no_improve:
            # Perturb a random edge
            if m > 0:
                ri = random.randint(0, m - 1)
                old_c = colors[ri]
                new_c = random.randint(0, 3)
                if new_c != old_c:
                    # Need to find which edge this is - this is expensive
                    # Skip for now
                    pass
    
    return best_result if best_tri_count == 0 else None

def _simulated_annealing(n, time_budget):
    """
    Simulated annealing approach for finding a valid 4-coloring of K_n.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    # Start with greedy initialization
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
    
    while True:
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random edge to change
        ri = random.randint(0, m - 1)
        old_c = colors[ri]
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
        
        # Find which edge (u, v) this is
        # We need to map index to (u, v)
        # This is O(n) per step, which is acceptable for n=50
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
        
        # Compute delta in triangle count
        removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
        added = (adj[new_c][u] & adj[new_c][v]).bit_count()
        delta = added - removed
        
        if delta <= 0:
            # Accept
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
            # Accept with probability exp(-delta / T)
            import math
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
        
        # Try local search for larger n.
        # Targets: 50, 45, 40, 35.
        
        # Time budget allocation
        elapsed = time.time() - _START_TIME
        remaining = _DEADLINE - elapsed
        if remaining < 5:
            return best
            
        targets = [50, 45, 40, 35]
        
        # We have ~110s.
        # Let's try 50 with 50s using simulated annealing, then 45 with 20s, etc.
        
        for target_n in targets:
            if target_n <= best_n:
                continue
            
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining < 5:
                break
                
            # Allocate budget
            if target_n == 50:
                budget = min(remaining - 5, 50)
            elif target_n == 45:
                budget = min(remaining - 5, 20)
            else:
                budget = min(remaining - 5, 10)
                
            if budget < 2:
                break
                
            # Try simulated annealing first
            result = _simulated_annealing(target_n, budget)
            if result is not None:
                return {'n': target_n, 'colors': result}
            
            # Try local search if time remains
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining > 5:
                result = _local_search(target_n, min(remaining - 5, 10))
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
