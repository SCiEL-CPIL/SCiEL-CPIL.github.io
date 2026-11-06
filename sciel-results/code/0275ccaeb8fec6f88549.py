import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin for 150s limit
    
    # Try to find a valid n=26 coloring
    coloring_26 = None
    attempts = 0
    while time.time() < deadline - 30.0 and coloring_26 is None and attempts < 20:
        coloring_26 = local_search_fast(26, time_limit=min(10.0, deadline - time.time() - 30.0))
        attempts += 1
        
    if coloring_26 is None:
        # Fallback: try one more time with more time
        if time.time() < deadline - 5.0:
            coloring_26 = local_search_fast(26, time_limit=deadline - time.time() - 5.0)
        
    if coloring_26 is None:
        # Absolute fallback
        return {'n': 1, 'colors': []}
    
    # Try to extend to n=27
    if time.time() < deadline - 5.0:
        n_27_colors = try_extend_fast(coloring_26, 26, deadline)
        if n_27_colors is not None:
            return {'n': 27, 'colors': n_27_colors}
    
    # Try a few more n=26 colorings and extensions in remaining time
    while time.time() < deadline - 5.0:
        c26 = local_search_fast(26, time_limit=min(5.0, deadline - time.time() - 5.0))
        if c26 is not None:
            n27 = try_extend_fast(c26, 26, deadline)
            if n27 is not None:
                return {'n': 27, 'colors': n27}
    
    return coloring_26

def local_search_fast(n, time_limit=10.0):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial coloring: try to be somewhat structured
    # Use a pattern based on vertex indices
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Simple pattern: color based on (i+j) mod 5
            colors.append((i + j) % 5)
    
    # Compute initial violations and adjacency
    adj = [[set() for _ in range(n)] for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj[c][i].add(j)
        adj[c][j].add(i)
    
    # Count violations: for each color, for each pair of vertices, count pairs of common neighbors
    def count_violations(adj):
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[c][u] & adj[c][v]
                    k = len(common)
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations
    
    current_violations = count_violations(adj)
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        # Pick a random edge to change
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        # Try a random new color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
        
        i, j = edges[e_idx]
        
        # Update adj: remove old color, add new color
        adj[old_color][i].discard(j)
        adj[old_color][j].discard(i)
        adj[new_color][i].add(j)
        adj[new_color][j].add(i)
        
        colors[e_idx] = new_color
        new_violations = count_violations(adj)
        
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            colors[e_idx] = old_color
            adj[new_color][i].discard(j)
            adj[new_color][j].discard(i)
            adj[old_color][i].add(j)
            adj[old_color][j].add(i)
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    return None

def try_extend_fast(base_coloring, n_base, deadline):
    n_new = n_base + 1
    base_colors = base_coloring['colors']
    
    # Precompute adjacency sets for base graph
    adj = [[set() for _ in range(n_base)] for _ in range(5)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
            
    # Build conflict graphs: conflict[c][u] contains v if u and v share a common neighbor in color c
    conflict = [[set() for _ in range(n_base)] for _ in range(5)]
    for c in range(5):
        for u in range(n_base):
            for v in range(u + 1, n_base):
                if adj[c][u] & adj[c][v]:
                    conflict[c][u].add(v)
                    conflict[c][v].add(u)
                    
    # We need to assign colors 0..4 to vertices 0..n_base-1
    # such that for each c, no edge in conflict[c] has both endpoints colored c.
    
    # Initial assignment: try to avoid conflicts
    v_colors = [0] * n_base
    for v in range(n_base):
        # Find a color that doesn't conflict
        best_c = 0
        best_conflicts = float('inf')
        for c in range(5):
            conflicts = 0
            for u in conflict[c][v]:
                if v_colors[u] == c:
                    conflicts += 1
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_c = c
        v_colors[v] = best_c
    
    def count_violations_ext(v_colors):
        violations = 0
        for c in range(5):
            for u in range(n_base):
                for v in conflict[c][u]:
                    if v > u:
                        if v_colors[u] == c and v_colors[v] == c:
                            violations += 1
        return violations
    
    current_violations = count_violations_ext(v_colors)
    if current_violations == 0:
        return build_final_colors(base_colors, v_colors, n_base)
    
    best_violations = current_violations
    best_v_colors = v_colors[:]
    
    start_time = time.time()
    time_limit = min(20.0, deadline - start_time - 5.0)
    if time_limit <= 0:
        return None
    
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        v_idx = random.randint(0, n_base - 1)
        old_color = v_colors[v_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        v_colors[v_idx] = new_color
        new_violations = count_violations_ext(v_colors)
        
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
            
    if best_violations == 0:
        return build_final_colors(base_colors, best_v_colors, n_base)
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
                base_idx = i * (n_base - 1) - i * (i - 1) // 2 + (j - i - 1)
                final_colors[idx_new] = base_colors[base_idx]
            else:
                # j == n_base
                final_colors[idx_new] = v_colors[i]
            idx_new += 1
            
    return final_colors
