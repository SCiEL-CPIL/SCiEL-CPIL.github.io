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
    Greedy vertex-by-vertex initialization.
    Add vertices 0..n-1 one at a time. When adding vertex v,
    assign colors to edges (u, v) for u < v to minimize new triangles.
    Returns colors list in lexicographic order.
    """
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    
    # For each new vertex v, determine colors for edges (u, v)
    for v in range(1, n):
        # For each u < v, try all 4 colors and pick the one that minimizes new triangles
        # A new triangle (u, w, v) is formed if color(u,v) == color(u,w) == color(w,v)
        # But we're setting all (u,v) simultaneously, so we need to consider interactions
        
        # Simpler approach: for each u < v, independently choose color that minimizes
        # the number of w < u such that color(u,w) == c
        # This is a heuristic that ignores coupling between different u's
        
        # Better: use a greedy assignment that considers the full interaction
        # For simplicity, let's do independent assignment first
        
        for u in range(v):
            # Try each color and count how many w < u have color(u,w) == c
            best_c = 0
            min_tris = float('inf')
            for c in range(4):
                # Count w < u such that edge (u,w) has color c
                # adj[c][u] has bits for all neighbors of u with color c
                # We want w < u, so mask out w >= u
                w_mask = adj[c][u] & ((1 << u) - 1)
                tris = w_mask.bit_count()
                if tris < min_tris:
                    min_tris = tris
                    best_c = c
            
            # Assign color best_c to edge (u, v)
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
    
    return colors

def _simulated_annealing(n, time_budget):
    """
    Simulated annealing to find a valid 4-coloring of K_n.
    Uses greedy initialization and then refines with SA.
    Returns colors list if successful, None otherwise.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    # Start with greedy initialization
    colors = _greedy_init(n)
    adj = _build_adj(colors, n)
    
    # Count initial triangles
    current_tris = _count_mono_triangles(adj, n)
    
    if current_tris == 0:
        return colors
    
    # Simulated annealing parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute edge list for random selection
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random edge
        edge_idx = random.randint(0, m - 1)
        # Find which edge this is
        # We need to map edge_idx to (i, j)
        # edge_idx = i*(n-1) - i*(i+1)//2 + (j-i-1)
        # Let's just iterate to find it (slow but simple)
        # Better: precompute mapping
        # For now, let's just pick random i, j
        i = random.randint(0, n - 1)
        j = random.randint(i + 1, n - 1)
        
        old_c = colors[_compute_index(n, i, j)]
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
        
        # Compute delta in triangle count
        # Triangles involving edge (i, j): for each k != i, j
        # Before: triangle (i,j,k) is mono if color(i,k)==color(j,k)==old_c
        # After: triangle (i,j,k) is mono if color(i,k)==color(j,k)==new_c
        
        delta = 0
        # Count k where (i,k) and (j,k) are both old_c
        common_old = adj[old_c][i] & adj[old_c][j]
        common_old &= ~((1 << i) | (1 << j))
        delta -= common_old.bit_count()  # These triangles disappear
        
        # Count k where (i,k) and (j,k) are both new_c
        common_new = adj[new_c][i] & adj[new_c][j]
        common_new &= ~((1 << i) | (1 << j))
        delta += common_new.bit_count()  # These triangles appear
        
        # Accept or reject
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            # Accept the move
            idx = _compute_index(n, i, j)
            colors[idx] = new_c
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            current_tris += delta
            
            if current_tris == 0:
                return colors
        
        # Cool down
        T *= cooling_rate
    
    return None

def _local_search(n, time_budget):
    """
    Local search to find a valid 4-coloring of K_n.
    Returns colors list if successful, None otherwise.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Initialize with greedy
        colors = _greedy_init(n)
        adj = _build_adj(colors, n)
        
        # Local search loop
        while True:
            if time.time() - start_time > time_budget:
                return None
            
            tri = _find_one_mono_triangle(adj, n)
            if tri is None:
                return colors
            
            i, j, k, c = tri
            
            # Choose an edge to recolor. Randomly pick one of the 3 edges.
            edges = [(i, j), (i, k), (j, k)]
            edge = edges[random.randint(0, 2)]
            u, v = edge
            
            # Determine best new color
            best_c = -1
            min_new_tris = float('inf')
            
            for new_c in range(4):
                if new_c == c:
                    continue
                # Count new triangles formed by (u, v) with color new_c
                new_tris = (adj[new_c][u] & adj[new_c][v]).bit_count()
                
                if new_tris < min_new_tris:
                    min_new_tris = new_tris
                    best_c = new_c
                    
            if best_c == -1:
                best_c = (c + 1) % 4
                
            # Recolor
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            
            # Update adj
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)

def construct():
    try:
        best = _seed_32()
        best_n = 32
        
        # Try simulated annealing for larger n.
        # Targets: 50, 45, 40, 35.
        
        # Time budget allocation
        elapsed = time.time() - _START_TIME
        remaining = _DEADLINE - elapsed
        if remaining < 5:
            return best
            
        targets = [50, 45, 40, 35]
        
        for target_n in targets:
            if target_n <= best_n:
                continue
            
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining < 5:
                break
                
            # Allocate budget
            if target_n == 50:
                budget = min(remaining - 5, 60)
            elif target_n == 45:
                budget = min(remaining - 5, 30)
            else:
                budget = min(remaining - 5, 10)
                
            if budget < 2:
                break
                
            # Try simulated annealing first
            result = _simulated_annealing(target_n, budget * 0.6)
            if result is not None:
                return {'n': target_n, 'colors': result}
            
            # Try local search with remaining time
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining < 5:
                break
            budget2 = min(remaining - 5, budget * 0.4)
            if budget2 < 2:
                break
            result = _local_search(target_n, budget2)
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
