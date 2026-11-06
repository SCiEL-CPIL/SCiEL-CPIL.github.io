import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try n=26 first with multiple attempts, then 25, etc.
    for n in [26, 25, 24]:
        if time.time() > deadline - 10:
            break
        # Multiple attempts for each n
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            remaining = deadline - time.time()
            if remaining < 5:
                break
            result = local_search_optimized(n, time_limit=min(remaining, 45))
            if result is not None:
                return result
            
    # Fallback to seed
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def local_search_optimized(n, time_limit=45):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize colors with a structured approach based on distance
    # This gives a better starting point than pure random
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            # Use a structured coloring based on distance
            # This helps avoid immediate violations
            c = d % 5
            colors.append(c)
    
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    def update_neighbors(idx, old_color, new_color):
        i, j = edges[idx]
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    # Precompute common neighbor counts for each color and pair of vertices
    k = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    def calc_k_for_color(c):
        for u in range(n):
            for v in range(u + 1, n):
                common = neighbors[c][u] & neighbors[c][v]
                cnt = common.bit_count()
                k[c][u][v] = cnt
                k[c][v][u] = cnt
                
    for c in range(5):
        calc_k_for_color(c)
        
    def total_violations():
        v = 0
        for c in range(5):
            for u in range(n):
                for v_idx in range(u + 1, n):
                    kc = k[c][u][v_idx]
                    if kc >= 2:
                        v += kc * (kc - 1) // 2
        return v
        
    current_violations = total_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 2.0
    cooling_rate = 0.9995
    min_temperature = 0.1
    
    # Local search loop with simulated annealing
    no_improve_count = 0
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        i, j = edges[e_idx]
        
        # Calculate delta violations
        delta = 0
        
        # Effect on old_color
        for w in range(n):
            if w == i or w == j:
                continue
            if (neighbors[old_color][w] >> j) & 1:
                old_k = k[old_color][i][w]
                new_k = old_k - 1
                delta -= (old_k * (old_k - 1) // 2) - (new_k * (new_k - 1) // 2)
                k[old_color][i][w] = new_k
                k[old_color][w][i] = new_k
                
            if (neighbors[old_color][w] >> i) & 1:
                old_k = k[old_color][j][w]
                new_k = old_k - 1
                delta -= (old_k * (old_k - 1) // 2) - (new_k * (new_k - 1) // 2)
                k[old_color][j][w] = new_k
                k[old_color][w][j] = new_k
                
        # Effect on new_color
        for w in range(n):
            if w == i or w == j:
                continue
            if (neighbors[new_color][w] >> j) & 1:
                old_k = k[new_color][i][w]
                new_k = old_k + 1
                delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                k[new_color][i][w] = new_k
                k[new_color][w][i] = new_k
                
            if (neighbors[new_color][w] >> i) & 1:
                old_k = k[new_color][j][w]
                new_k = old_k + 1
                delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                k[new_color][j][w] = new_k
                k[new_color][w][j] = new_k
                
        # Apply change if accepted
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations += delta
            colors[e_idx] = new_color
            update_neighbors(e_idx, old_color, new_color)
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                no_improve_count = 0
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
            else:
                no_improve_count += 1
        else:
            # Revert k changes
            for w in range(n):
                if w == i or w == j:
                    continue
                common = neighbors[old_color][i] & neighbors[old_color][w]
                k[old_color][i][w] = common.bit_count()
                k[old_color][w][i] = common.bit_count()
                
                common = neighbors[old_color][j] & neighbors[old_color][w]
                k[old_color][j][w] = common.bit_count()
                k[old_color][w][j] = common.bit_count()
                
                common = neighbors[new_color][i] & neighbors[new_color][w]
                k[new_color][i][w] = common.bit_count()
                k[new_color][w][i] = common.bit_count()
                
                common = neighbors[new_color][j] & neighbors[new_color][w]
                k[new_color][j][w] = common.bit_count()
                k[new_color][w][j] = common.bit_count()
            
            # Revert colors and neighbors
            colors[e_idx] = old_color
            update_neighbors(e_idx, new_color, old_color)
            
            no_improve_count += 1
            
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
            
        # If stuck for too long, restart with random initialization
        if no_improve_count > 5000:
            no_improve_count = 0
            # Random restart
            for idx in range(num_edges):
                old_c = colors[idx]
                new_c = random.randint(0, 4)
                colors[idx] = new_c
                i, j = edges[idx]
                neighbors[old_c][i] &= ~(1 << j)
                neighbors[old_c][j] &= ~(1 << i)
                neighbors[new_c][i] |= (1 << j)
                neighbors[new_c][j] |= (1 << i)
            for c in range(5):
                calc_k_for_color(c)
            current_violations = total_violations()
            if current_violations == 0:
                return {'n': n, 'colors': colors}
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
            
    return None
