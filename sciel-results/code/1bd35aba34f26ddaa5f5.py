import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    # Helper to multiply in GF(16)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Log table for GF(16)
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    # Build the base 32-vertex coloring
    # Two copies of the 16-vertex Clebsch-based coloring, cross edges color 3
    n_base = 32
    base_colors = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                # Within the same block (0-15 or 16-31)
                # Use log[i^j] % 3
                # i^j is non-zero since i != j
                base_colors.append(log[i ^ j] % 3)
            else:
                # Between blocks
                base_colors.append(3)
    
    best_n = n_base
    best_colors = base_colors[:]

    # Target n=50
    n_target = 50
    if time.time() - start_time > deadline - safety_margin:
        return {'n': best_n, 'colors': best_colors}

    # Precompute edge indices for n=50
    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Precompute edge pairs for fast lookup
    edge_pairs = []
    for i in range(n_target):
        for j in range(i + 1, n_target):
            edge_pairs.append((i, j))
    
    n_edges = len(edge_pairs)

    # Initialize n=50 coloring
    colors = [0] * n_edges
    
    # Fill base part
    for i in range(n_base):
        for j in range(i + 1, n_base):
            idx = get_edge_index(i, j, n_target)
            colors[idx] = base_colors[i * (n_base - 1) - i * (i - 1) // 2 + (j - i - 1)]

    # Fill new edges (involving vertices >= 32) with random colors
    for i in range(n_base):
        for j in range(i + 1, n_target):
            if j >= n_base:
                idx = get_edge_index(i, j, n_target)
                colors[idx] = random.randint(0, 3)
    for i in range(n_base, n_target):
        for j in range(i + 1, n_target):
            idx = get_edge_index(i, j, n_target)
            colors[idx] = random.randint(0, 3)

    # Efficient Energy Calculation and Update
    def initial_energy_and_adj(colors, n):
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_edge_index(i, j, n)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        energy = 0
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[c][i] >> j) & 1:
                        common = adj[c][i] & adj[c][j]
                        energy += common.bit_count()
        energy //= 3
        return energy, adj

    # Simulated annealing search
    found = False
    max_restarts = 15
    max_iters_per_restart = 30000
    T_initial = 5.0
    T_min = 0.1
    cooling_rate = 0.999
    
    for restart in range(max_restarts):
        if time.time() - start_time > deadline - safety_margin:
            break
            
        # Re-initialize random part for new restart
        if restart > 0:
            for i in range(n_base):
                for j in range(i + 1, n_target):
                    if j >= n_base:
                        idx = get_edge_index(i, j, n_target)
                        colors[idx] = random.randint(0, 3)
            for i in range(n_base, n_target):
                for j in range(i + 1, n_target):
                    idx = get_edge_index(i, j, n_target)
                    colors[idx] = random.randint(0, 3)
        
        energy, adj = initial_energy_and_adj(colors, n_target)
        
        if energy == 0:
            found = True
            break
            
        T = T_initial
        best_energy_this_restart = energy
        best_colors_this_restart = colors[:]
        
        for it in range(max_iters_per_restart):
            if time.time() - start_time > deadline - safety_margin:
                break
                
            # Pick a random edge
            idx = random.randint(0, n_edges - 1)
            u, v = edge_pairs[idx]
            
            old_c = colors[idx]
            
            # Try changing to a new color (choose from other 3 colors)
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Calculate Delta Energy
            common_old = adj[old_c][u] & adj[old_c][v]
            D_old = common_old.bit_count()
            
            common_new = adj[new_c][u] & adj[new_c][v]
            D_new = common_new.bit_count()
            
            delta_e = D_new - D_old
            
            if delta_e <= 0 or random.random() < math.exp(-delta_e / T):
                # Accept
                colors[idx] = new_c
                energy += delta_e
                
                # Update adj
                adj[old_c][u] ^= (1 << v)
                adj[old_c][v] ^= (1 << u)
                adj[new_c][u] ^= (1 << v)
                adj[new_c][v] ^= (1 << u)
                
                if energy < best_energy_this_restart:
                    best_energy_this_restart = energy
                    best_colors_this_restart = colors[:]
                
                if energy == 0:
                    found = True
                    break
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # If we found a solution, we're done
        if found:
            break
            
        # Keep the best solution from this restart
        if best_energy_this_restart < energy:
            colors = best_colors_this_restart[:]
    
    # After all restarts, check if we have a valid solution
    final_energy, _ = initial_energy_and_adj(colors, n_target)
    if final_energy == 0:
        found = True
    
    if found:
        best_n = n_target
        best_colors = colors[:]
        
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
