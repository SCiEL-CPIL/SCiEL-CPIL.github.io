import time
import random

def _precompute_k32():
    """Precompute the known valid 32-vertex 4-coloring (seed construction)."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

# Precompute at module load
_K32_COLORS = _precompute_k32()

def _edge_idx(i, j, n):
    """Get the index of edge (i,j) in the flat colors list, assuming i<j and n is current vertex count."""
    if i > j:
        i, j = j, i
    # Number of edges before vertex i
    prev = i * (i + 1) // 2
    # In vertex i's row, j is at position (j - i - 1)
    return prev + (j - i - 1)

def _validate_fast(colors, n):
    """Fast validation using bitsets or sets. Returns True if valid."""
    if len(colors) != n * (n - 1) // 2:
        return False
    
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 3:
                return False
            adj[c].add((i, j))
            idx += 1

    for c in range(4):
        adj_set = adj[c]
        # For each vertex, check neighbors
        adj_list = [[] for _ in range(n)]
        for (u, v) in adj_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        for u in range(n):
            nbrs = adj_list[u]
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    a, b = nbrs[i], nbrs[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in adj_set:
                        return False
    return True

def _local_search_optimized(n, deadline):
    """
    Optimized local search for n-vertex 4-coloring.
    Uses incremental cost update.
    """
    num_edges = n * (n - 1) // 2
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Precompute edge indices for (u, v)
    edge_to_uv = {}
    uv_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_uv[idx] = (i, j)
            uv_to_idx[(i, j)] = idx
            uv_to_idx[(j, i)] = idx
            idx += 1
            
    # Initial cost calculation
    def calc_cost():
        cost = 0
        for u in range(n):
            for v in range(u + 1, n):
                c_uv = colors[uv_to_idx[(u, v)]]
                for w in range(v + 1, n):
                    c_vw = colors[uv_to_idx[(v, w)]]
                    c_wu = colors[uv_to_idx[(u, w)]]
                    if c_uv == c_vw == c_wu:
                        cost += 1
        return cost

    current_cost = calc_cost()
    best_cost = current_cost
    best_colors = colors[:]
    
    start_time = time.time()
    temp = 1.0
    cooling = 0.999
    last_improve = start_time
    
    while time.time() < deadline:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        u, v = edge_to_uv[e_idx]
        old_color = colors[e_idx]
        new_color = random.randint(0, 3)
        if old_color == new_color:
            continue
        
        # Calculate delta cost
        delta = 0
        for w in range(n):
            if w == u or w == v:
                continue
            c_vw = colors[uv_to_idx[(v, w)]]
            c_wu = colors[uv_to_idx[(u, w)]]
            
            if old_color == c_vw == c_wu:
                delta -= 1
            if new_color == c_vw == c_wu:
                delta += 1
                
        new_cost = current_cost + delta
        
        if new_cost < current_cost:
            colors[e_idx] = new_color
            current_cost = new_cost
            last_improve = time.time()
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                if best_cost == 0:
                    return best_colors
        else:
            if random.random() < pow(2.71828, -delta / temp):
                colors[e_idx] = new_color
                current_cost = new_cost
        
        temp *= cooling
        if time.time() - last_improve > 5.0:
            temp = 1.0
            last_improve = time.time()
            
    return best_colors if best_cost == 0 else None

def construct():
    start_time = time.time()
    deadline = start_time + 110

    best_n = 32
    best_colors = _K32_COLORS

    # Try local search for n=50
    if time.time() < deadline - 10:
        result_50 = _local_search_optimized(50, deadline - 5)
        if result_50 is not None:
            if _validate_fast(result_50, 50):
                best_n = 50
                best_colors = result_50
                return {'n': best_n, 'colors': best_colors}
                
    # If 50 failed, try 40? Or just return 32?
    # Let's try 40 if we have time.
    if time.time() < deadline - 10 and best_n == 32:
        result_40 = _local_search_optimized(40, deadline - 5)
        if result_40 is not None:
            if _validate_fast(result_40, 40):
                best_n = 40
                best_colors = result_40

    return {'n': best_n, 'colors': best_colors}
