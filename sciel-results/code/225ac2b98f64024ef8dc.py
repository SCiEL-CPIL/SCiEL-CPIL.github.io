import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Find a valid coloring for n=26
    # This is known to exist and is findable quickly
    result_26 = local_search(26, time_limit=10.0)
    
    if result_26 is None:
        # Fallback if search fails (should not happen)
        # Try a bit longer or use a hardcoded one if we had one.
        # For now, try again with more time
        result_26 = local_search(26, time_limit=30.0)
        if result_26 is None:
            # Absolute fallback
            return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

    # 2. Try to extend to n=27
    # The new vertex is 26.
    # We need to assign colors to edges (26, i) for i in 0..25.
    # Let's represent the new colors as a list of 26 variables.
    # We will use a local search on these 26 variables to minimize violations involving vertex 26.
    
    base_colors = result_26['colors']
    n_base = 26
    
    # Precompute adjacency lists for the base graph (n=26) for each color
    # adj[c][v] = set/list of neighbors of v in color c
    adj = [[set() for _ in range(n_base)] for _ in range(5)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
            
    # Function to calculate violations involving the new vertex 26
    # new_colors is a list of 26 ints, new_colors[i] is color of (26, i)
    # We want to minimize the number of C4s formed.
    # A C4 is formed if we pick two vertices u, v from the neighbors of 26 in color c,
    # and they have a common neighbor w in G_c.
    # So for each color c, let N_c be the set of vertices i such that new_colors[i] == c.
    # The number of violations in color c is the number of pairs {u, v} in N_c such that |N_c(u) ∩ N_c(v)| >= 1.
    # Actually, the condition is "no two vertices with two common neighbours".
    # So a violation is a pair {u, v} such that |N_c(u) ∩ N_c(v)| >= 2.
    # Let's refine the violation count:
    # For each c, for each pair u, v in N_c, count = |N_c(u) ∩ N_c(v)|.
    # If count >= 2, we have a violation. Specifically, each pair of common neighbors forms a C4.
    # The number of C4s is sum_{u<v in N_c} C(|N_c(u) ∩ N_c(v)|, 2).
    
    def get_violations_27(new_colors):
        violations = 0
        # Group neighbors of 26 by color
        neighbors_by_color = [[] for _ in range(5)]
        for i in range(n_base):
            c = new_colors[i]
            neighbors_by_color[c].append(i)
            
        for c in range(5):
            neighbors = neighbors_by_color[c]
            # For each pair of neighbors u, v in this color, check if they share a common neighbor in G_c
            for u_idx in range(len(neighbors)):
                u = neighbors[u_idx]
                for v_idx in range(u_idx + 1, len(neighbors)):
                    v = neighbors[v_idx]
                    # Check intersection of adj[c][u] and adj[c][v]
                    common = adj[c][u] & adj[c][v]
                    k = len(common)
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    # Local search on new_colors
    best_new_colors = [random.randint(0, 4) for _ in range(n_base)]
    best_violations = get_violations_27(best_new_colors)
    
    # Optimize new_colors
    start_opt = time.time()
    limit_opt = 20.0 # Use 20s to try to extend
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() - start_opt < limit_opt:
        if best_violations == 0:
            break
        e_idx = random.randint(0, n_base - 1)
        old_c = best_new_colors[e_idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
            
        best_new_colors[e_idx] = new_c
        new_violations = get_violations_27(best_new_colors)
        
        delta = new_violations - best_violations
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            best_violations = new_violations
        else:
            best_new_colors[e_idx] = old_c
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    if best_violations == 0:
        # Construct the full coloring for n=27
        final_colors = [0] * (27 * 26 // 2)
        
        # Map for base edges
        # Iterate through n=27 edges
        idx = 0
        for i in range(27):
            for j in range(i + 1, 27):
                if j < 26:
                    # This edge (i, j) exists in K26
                    # We need its index in the K26 coloring
                    # K26 index for (i, j):
                    # Number of edges starting with k < i is sum_{k=0}^{i-1} (25-k)
                    # Edges starting with i and ending with < j is (j - i - 1)
                    # sum_{k=0}^{i-1} (25-k) = i*25 - i*(i-1)/2
                    base_idx = i * 25 - i * (i - 1) // 2 + (j - i - 1)
                    final_colors[idx] = base_colors[base_idx]
                else:
                    # j == 26
                    # Edge is (i, 26)
                    # This corresponds to best_new_colors[i]
                    final_colors[idx] = best_new_colors[i]
                idx += 1
                
        return {'n': 27, 'colors': final_colors}
        
    # If extension failed, return n=26
    return result_26

def local_search(n, time_limit=10.0):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Precompute edge indices for fast lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    def get_violations(colors):
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
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
