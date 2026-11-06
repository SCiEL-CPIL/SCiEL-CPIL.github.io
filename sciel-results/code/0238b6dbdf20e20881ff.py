import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Seed solution for fallback
    def seed_solution():
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
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    best_result = seed_solution()
    
    # Local search function
    def try_find(n, time_limit):
        if time.time() > deadline:
            return None
            
        num_edges = n * (n - 1) // 2
        # colors is a list of length num_edges
        # Index mapping:
        # edge (i, j) where i < j
        # index = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        # Simpler: use a 2D list or a dictionary?
        # For speed, use a flat list and precomputed indices.
        
        # Precompute triangle neighbors
        # For each edge (u, v), list of w such that w != u, v
        # We can just iterate w from 0 to n-1
        
        def get_idx(u, v):
            if u > v:
                u, v = v, u
            return u * (n - 1) - u * (i + 1) // 2 + (v - u - 1) # This formula is tricky, let's use a mapping dict or list of lists.
        
        # Let's use a 2D list for simplicity and speed of access, though flat is better for memory.
        # n=50 is small. 50x50 list is fine.
        colors = [[-1] * n for _ in range(n)]
        
        # Initialize randomly
        for i in range(n):
            for j in range(i + 1, n):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j]
                
        # Count mono triangles
        tri_counts = [0, 0, 0, 0]
        def update_triangles(u, v, old_c, new_c):
            # Triangles involving (u, v) are (u, v, w)
            for w in range(n):
                if w == u or w == v:
                    continue
                c_uw = colors[u][w]
                c_vw = colors[v][w]
                
                # Check if (old_c, c_uw, c_vw) was mono
                if old_c == c_uw and old_c == c_vw:
                    tri_counts[old_c] -= 1
                    
                # Check if (new_c, c_uw, c_vw) is mono
                if new_c == c_uw and new_c == c_vw:
                    tri_counts[new_c] += 1

        # Initial count
        for u in range(n):
            for v in range(u + 1, n):
                c_uv = colors[u][v]
                for w in range(v + 1, n):
                    c_uw = colors[u][w]
                    c_vw = colors[v][w]
                    if c_uv == c_uw == c_vw:
                        tri_counts[c_uv] += 1
                        
        total_mono = sum(tri_counts)
        
        if total_mono == 0:
            # Flatten and return
            flat_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    flat_colors.append(colors[i][j])
            return {'n': n, 'colors': flat_colors}

        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        start_loop = time.time()
        
        while time.time() - start_loop < time_limit and T > min_T:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            old_c = colors[u][v]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
                
            # Calculate delta in mono triangles
            delta = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                c_uw = colors[u][w]
                c_vw = colors[v][w]
                
                if old_c == c_uw == c_vw:
                    delta -= 1
                if new_c == c_uw == c_vw:
                    delta += 1
                    
            if delta < 0:
                # Accept
                colors[u][v] = new_c
                colors[v][u] = new_c
                total_mono += delta
            else:
                # Accept with probability exp(-delta / T)
                # delta is >= 0
                if delta == 0:
                    # Accept neutral moves to explore
                    if random.random() < 0.5:
                        colors[u][v] = new_c
                        colors[v][u] = new_c
                else:
                    # Calculate probability
                    # Use math.exp
                    import math
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        colors[u][v] = new_c
                        colors[v][u] = new_c
                        total_mono += delta
                        
            T *= cooling
            
            if total_mono == 0:
                flat_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        flat_colors.append(colors[i][j])
                return {'n': n, 'colors': flat_colors}
                
        return None

    # Try n=50, 49, 48...
    # Allocate more time to larger n
    for n in [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]:
        if time.time() > deadline:
            break
            
        # Time allocation: 50 gets 30s, others get less
        # But we have ~110s total.
        # Let's give 50 a chance of 60s. If it fails, try 49 with 20s, etc.
        time_alloc = 10
        if n == 50:
            time_alloc = 60
        elif n == 49:
            time_alloc = 20
        elif n == 48:
            time_alloc = 10
        else:
            time_alloc = 5
            
        if time.time() + time_alloc > deadline:
            time_alloc = deadline - time.time()
            if time_alloc < 1:
                break
                
        result = try_find(n, time_alloc)
        if result:
            best_result = result
            break # Found a valid one, stop (since we iterate from high to low)
            
    return best_result
