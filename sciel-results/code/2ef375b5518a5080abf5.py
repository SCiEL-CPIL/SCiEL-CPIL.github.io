import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Step 1: Find a valid n=26 coloring
    # We try a few times to find a valid n=26 coloring
    coloring_26 = None
    attempts = 0
    while time.time() < deadline - 60.0 and coloring_26 is None and attempts < 5:
        coloring_26 = local_search_n(26, time_limit=10.0)
        attempts += 1
        
    if coloring_26 is None:
        # Fallback: return a small valid coloring if search fails completely
        # But n=26 should be easy to find.
        # If it fails, try a longer search
        coloring_26 = local_search_n(26, time_limit=60.0)
        
    if coloring_26 is None:
        # Absolute fallback
        return {'n': 1, 'colors': []}

    # Step 2: Try to extend to n=27
    n_27_colors = try_extend(coloring_26, 26, deadline)
    
    if n_27_colors is not None:
        return {'n': 27, 'colors': n_27_colors}
        
    # Step 3: If n=27 failed, try to find a different n=26 coloring that might extend?
    # Or just return n=26.
    # Given the time, returning n=26 is a safe baseline.
    # But we can try a few more n=26 searches and extensions in the remaining time.
    
    # Let's try a few more n=26 colorings and extensions
    while time.time() < deadline - 10.0:
        c26 = local_search_n(26, time_limit=5.0)
        if c26 is not None:
            n27 = try_extend(c26, 26, deadline)
            if n27 is not None:
                return {'n': 27, 'colors': n27}
                
    return coloring_26

def local_search_n(n, time_limit=10.0):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    def get_violations(colors):
        # Build adjacency lists for each color
        # adj[c][v] = set of neighbors
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[c][u] & adj[c][v]
                    k = len(common)
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    current_violations = get_violations(colors)
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        colors[e_idx] = new_color
        new_violations = get_violations(colors)
        
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            colors[e_idx] = old_color
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None

def try_extend(base_coloring, n_base, deadline):
    # base_coloring is a dict with 'n' and 'colors'
    # We want to extend to n_base + 1
    n_new = n_base + 1
    base_colors = base_coloring['colors']
    
    # Precompute adjacency sets for base graph
    adj = [ [set() for _ in range(n_base)] for _ in range(5) ]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
            
    # Build the conflict graphs K_c for the extension
    # K_c has vertices 0..n_base-1
    # Edge (u, v) in K_c if u and v have >= 1 common neighbor in G_c
    conflict = [ [set() for _ in range(n_base)] for _ in range(5) ]
    for c in range(5):
        for u in range(n_base):
            for v in range(u + 1, n_base):
                if adj[c][u] & adj[c][v]:
                    conflict[c][u].add(v)
                    conflict[c][v].add(u)
                    
    # Now we need to assign colors 0..4 to vertices 0..n_base-1
    # such that for each c, no edge in conflict[c] has both endpoints colored c.
    
    # Local search on vertex colors
    v_colors = [random.randint(0, 4) for _ in range(n_base)]
    
    def get_violations_ext(v_colors):
        violations = 0
        for c in range(5):
            # Check all edges in conflict[c]
            for u in conflict[c]:
                for v in conflict[c][u]:
                    if v > u: # avoid double counting
                        if v_colors[u] == c and v_colors[v] == c:
                            violations += 1
        return violations

    current_violations = get_violations_ext(v_colors)
    if current_violations == 0:
        # Construct final colors
        return build_final_colors(base_colors, v_colors, n_base)
        
    best_violations = current_violations
    best_v_colors = v_colors[:]
    
    start_time = time.time()
    time_limit = min(20.0, deadline - start_time - 5.0)
    if time_limit <= 0: return None
    
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        v_idx = random.randint(0, n_base - 1)
        old_color = v_colors[v_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        v_colors[v_idx] = new_color
        new_violations = get_violations_ext(v_colors)
        
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_v_colors = v_colors[:]
                if best_violations == 0:
                    return build_final_colors(base_colors, best_v_colors, n_base)
        else:
            v_colors[v_idx] = old_color
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None

def build_final_colors(base_colors, v_colors, n_base):
    n_new = n_base + 1
    num_edges_new = n_new * (n_new - 1) // 2
    final_colors = [0] * num_edges_new
    
    idx_new = 0
    idx_base = 0
    
    for i in range(n_new):
        for j in range(i + 1, n_new):
            if j < n_base:
                # Edge (i, j) is in base graph
                # Need to map (i, j) to index in base_colors
                # base_colors is for K_n_base
                # Index in K_n_base for (i, j) where i < j < n_base
                # Formula: sum_{k=0}^{i-1} (n_base - 1 - k) + (j - i - 1)
                # sum_{k=0}^{i-1} (n_base - 1 - k) = i * (n_base - 1) - i * (i - 1) / 2
                base_idx = i * (n_base - 1) - i * (i - 1) // 2 + (j - i - 1)
                final_colors[idx_new] = base_colors[base_idx]
            else:
                # j == n_base
                # Edge (i, n_base)
                # Color is v_colors[i]
                final_colors[idx_new] = v_colors[i]
            idx_new += 1
            
    return final_colors
