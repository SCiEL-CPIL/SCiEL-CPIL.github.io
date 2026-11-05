import random
import time
import math
import itertools

def construct():
    n = 16
    num_edges = n * (n - 1) // 2
    # Precompute all 4-sets
    all_4sets = list(itertools.combinations(range(n), 4))
    
    # Precompute which 4-sets contain which edge
    # edge_index[i][j] -> index in the flat list of edges?
    # It's easier to just iterate through the 91 sets for a given edge.
    # But we need to map (u,v) to a list of 4-set indices.
    
    # Let's create a mapping: edge (u,v) -> list of 4-set indices
    edge_to_sets = {}
    for idx, s in enumerate(all_4sets):
        for i in range(4):
            for j in range(i+1, 4):
                u, v = s[i], s[j]
                if u > v: u, v = v, u
                key = (u, v)
                if key not in edge_to_sets:
                    edge_to_sets[key] = []
                edge_to_sets[key].append(idx)
                
    def get_edge_color(colors, u, v):
        if u > v: u, v = v, u
        # Flat index
        # i * (n-1) - i*(i+1)//2 + (j-i-1) ... complicated
        # Just use a 2D array or dict?
        # For speed, use a 2D list colors[u][v]
        return colors[u][v]
        
    def count_violations(colors):
        violations = 0
        for s in all_4sets:
            # Get colors of the 6 edges
            c01 = colors[s[0]][s[1]]
            c02 = colors[s[0]][s[2]]
            c03 = colors[s[0]][s[3]]
            c12 = colors[s[1]][s[2]]
            c13 = colors[s[1]][s[3]]
            c23 = colors[s[2]][s[3]]
            
            distinct = len({c01, c02, c03, c12, c13, c23})
            if distinct < 5:
                violations += 1
        return violations

    def calculate_delta_cost(colors, edge, old_color, new_color):
        # Returns change in violation count
        if old_color == new_color:
            return 0
        u, v = edge
        if u > v: u, v = v, u
        # Get the 91 sets
        set_indices = edge_to_sets.get((u, v), [])
        delta = 0
        for idx in set_indices:
            s = all_4sets[idx]
            # Find the other 2 vertices
            others = [x for x in s if x != u and x != v]
            w, z = others[0], others[1]
            
            # Old colors
            c_uv_old = old_color
            c_uw = colors[u][w] if u < w else colors[w][u]
            c_uz = colors[u][z] if u < z else colors[z][u]
            c_vw = colors[v][w] if v < w else colors[w][v]
            c_vz = colors[v][z] if v < z else colors[z][v]
            c_wz = colors[w][z] if w < z else colors[z][w]
            
            old_distinct = len({c_uv_old, c_uw, c_uz, c_vw, c_vz, c_wz})
            
            # New colors
            c_uv_new = new_color
            new_distinct = len({c_uv_new, c_uw, c_uz, c_vw, c_vz, c_wz})
            
            old_bad = 1 if old_distinct < 5 else 0
            new_bad = 1 if new_distinct < 5 else 0
            delta += new_bad - old_bad
        return delta

    def run_sa(K, time_limit):
        # Initialize
        colors = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = random.randint(0, K-1)
                colors[j][i] = colors[i][j]
        
        current_cost = count_violations(colors)
        best_cost = current_cost
        best_colors = [row[:] for row in colors]
        
        if current_cost == 0:
            return best_colors
        
        start_time = time.time()
        T = 1.0
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            
            old_color = colors[i][j]
            new_color = random.randint(0, K-1)
            if old_color == new_color: continue
            
            delta = calculate_delta_cost(colors, (i, j), old_color, new_color)
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[i][j] = new_color
                colors[j][i] = new_color
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = [row[:] for row in colors]
                    if best_cost == 0:
                        return best_colors
            
            T *= cooling
            if T < 1e-3:
                T = 1e-3
        
        return best_colors if best_cost == 0 else None

    # Try to find a solution with 13 colors
    # Time budget: 100 seconds total.
    # We can try 13, then 14, then 15.
    # If we find a valid one, return it.
    
    # Seed solution
    def seed_solution():
        return [[(i + j) % 17 for j in range(n)] for i in range(n)]
    
    best_valid = None
    best_K = 17 # Seed uses 17
    
    # Try 13
    start = time.time()
    res = run_sa(13, 40)
    if res:
        return [res[i][j] for i in range(n) for j in range(i+1, n)]
        
    # Try 14
    if time.time() - start < 80:
        res = run_sa(14, 30)
        if res:
            return [res[i][j] for i in range(n) for j in range(i+1, n)]
            
    # Try 15
    if time.time() - start < 110:
        res = run_sa(15, 20)
        if res:
            return [res[i][j] for i in range(n) for j in range(i+1, n)]
            
    # Fallback to seed
    seed = seed_solution()
    return [seed[i][j] for i in range(n) for j in range(i+1, n)]
