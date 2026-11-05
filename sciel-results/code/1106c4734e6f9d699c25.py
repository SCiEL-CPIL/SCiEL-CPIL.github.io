import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Seed program's result for fallback (n=32)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def seed_coloring():
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    best_result = seed_coloring()
    best_n = 32

    def build_masks(n, colors):
        """Build adjacency masks from color matrix."""
        masks = [[0] * 4 for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i][j]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
        return masks

    def count_triangles(n, colors, masks):
        """Count monochromatic triangles."""
        total_tri = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i][j]
                total_tri += (masks[i][c] & masks[j][c]).bit_count()
        return total_tri // 3

    def init_random(n):
        """Initialize with random coloring."""
        colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j]
        return colors

    def init_algebraic(n):
        """Initialize based on algebraic structure (perturbed Chung-like construction)."""
        colors = [[0] * n for _ in range(n)]
        # Use a simple algebraic initialization: color based on (i+j) mod 4 or similar
        # This provides structure that may be easier to optimize
        for i in range(n):
            for j in range(i + 1, n):
                # Use a mix of algebraic rules
                val = (i * j + i + j) % 4
                colors[i][j] = val
                colors[j][i] = val
        # Add some randomness to break symmetry
        for _ in range(n * n // 10):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                c = random.randint(0, 3)
                colors[i][j] = c
                colors[j][i] = c
        return colors

    def init_greedy(n):
        """Initialize with a greedy approach to minimize triangles."""
        colors = [[0] * n for _ in range(n)]
        # Start with all zeros, then greedily change to reduce triangles
        for i in range(n):
            for j in range(i + 1, n):
                # Try each color and pick the one that minimizes local triangles
                best_c = 0
                best_delta = float('inf')
                for c in range(4):
                    # Count triangles involving edge (i,j) with color c
                    # This is approximate; we'll refine in SA
                    delta = random.randint(0, 10)  # Placeholder
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c
                colors[i][j] = best_c
                colors[j][i] = best_c
        # Add randomness
        for _ in range(n * n // 5):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                c = random.randint(0, 3)
                colors[i][j] = c
                colors[j][i] = c
        return colors

    def solve_sa_multi(n, max_time):
        """Multi-start simulated annealing with different initializations."""
        if n <= best_n:
            return None
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        num_edges = len(edges)
        
        best_total_tri = float('inf')
        best_colors = None
        
        # Different initialization strategies
        init_strategies = [init_random, init_algebraic, init_greedy]
        
        # Time per attempt
        time_per_attempt = max_time / 10  # Try ~10 attempts
        
        for attempt in range(10):
            if time.time() >= max_time:
                break
            
            # Choose initialization strategy
            init_fn = init_strategies[attempt % len(init_strategies)]
            colors = init_fn(n)
            masks = build_masks(n, colors)
            total_tri = count_triangles(n, colors, masks)
            
            if total_tri == 0:
                return n, colors
            
            if total_tri < best_total_tri:
                best_total_tri = total_tri
                best_colors = [row[:] for row in colors]
            
            # Simulated annealing
            T = 30.0
            cooling = 0.9998
            steps_since_improvement = 0
            local_best_tri = total_tri
            
            max_steps = 50000
            for step in range(max_steps):
                if time.time() >= max_time:
                    break
                
                # Pick random edge
                idx = random.randint(0, num_edges - 1)
                i, j = edges[idx]
                
                c_old = colors[i][j]
                c_new = random.randint(0, 3)
                if c_new == c_old:
                    continue
                
                # Calculate delta triangles
                tri_old = (masks[i][c_old] & masks[j][c_old]).bit_count()
                tri_new = (masks[i][c_new] & masks[j][c_new]).bit_count()
                
                delta_tri = tri_new - tri_old
                
                if delta_tri <= 0 or random.random() < math.exp(-delta_tri / max(T, 0.01)):
                    # Accept
                    colors[i][j] = c_new
                    colors[j][i] = c_new
                    
                    masks[i][c_old] &= ~(1 << j)
                    masks[j][c_old] &= ~(1 << i)
                    masks[i][c_new] |= (1 << j)
                    masks[j][c_new] |= (1 << i)
                    
                    total_tri += delta_tri
                    
                    if total_tri < local_best_tri:
                        local_best_tri = total_tri
                        steps_since_improvement = 0
                        if total_tri < best_total_tri:
                            best_total_tri = total_tri
                            best_colors = [row[:] for row in colors]
                    else:
                        steps_since_improvement += 1

                    if total_tri == 0:
                        return n, colors

                    # Early restart if stuck too long
                    if steps_since_improvement > 50000:
                        break

                T *= cooling
                if T < 0.01:
                    T = 0.01
            
            # If we broke out early due to being stuck, continue to next attempt
        
        # Return best found
        if best_colors is not None and best_total_tri == 0:
            return n, best_colors
        return None

    # Try to solve for descending n
    # Prioritize 50, then 49, etc.
    targets = [50, 49, 48, 45, 42]
    
    for target_n in targets:
        if time.time() > deadline - 10:
            break
            
        # Allocate time: give 50s to 50, less to others
        if target_n == 50:
            time_limit = min(deadline - 5, time.time() + 50)
        elif target_n == 49:
            time_limit = min(deadline - 5, time.time() + 30)
        elif target_n == 48:
            time_limit = min(deadline - 5, time.time() + 15)
        else:
            time_limit = min(deadline - 5, time.time() + 5)

        result = solve_sa_multi(target_n, time_limit)
        if result:
            n, colors = result
            out_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    out_colors.append(colors[i][j])
            return {'n': n, 'colors': out_colors}

    return best_result
