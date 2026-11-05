import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    def count_triangles(adj):
        # adj is a list of 4 lists of bitmasks
        # adj[c][u] is bitmask of neighbors of u with color c
        total = 0
        for c in range(4):
            for u in range(n):
                # Count pairs (v, w) in adj[c][u] with v < w
                # This counts each triangle 3 times (once for each vertex)
                # So total / 3 is the number of triangles
                mask = adj[c][u]
                # We need to count edges within the set of neighbors
                # Iterating over set bits is slow if dense, but n=50 is small
                # However, we can just use the property that we will use delta updates
                # For initialization, we can just do a full count
                # Or simpler: just sum popcount(adj[c][u] & adj[c][v]) for u < v and divide by 3?
                # No, popcount(adj[c][u] & adj[c][v]) counts common neighbors.
                # Sum over u < v of popcount(adj[c][u] & adj[c][v]) counts each triangle 3 times?
                # Triangle (a,b,c). 
                # u=a, v=b: counts c.
                # u=a, v=c: counts b.
                # u=b, v=c: counts a.
                # Yes, sum / 3 is the number of mono triangles for color c.
                pass
        # We will just initialize E by running the delta logic on a dummy? 
        # No, let's just calculate E directly for the initial state.
        # For n=50, O(n^3) is 125,000 ops. Very fast.
        return 0 # Placeholder

    def calculate_initial_energy(colors, n):
        # colors is a list of length n*(n-1)//2
        # Reconstruct adj
        adj = [[0]*n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i] |= 1 << j
                adj[c][j] |= 1 << i
                idx += 1
        
        E = 0
        for c in range(4):
            for u in range(n):
                for v in range(u+1, n):
                    # Count common neighbors
                    common = adj[c][u] & adj[c][v]
                    E += common.bit_count()
        return E // 3, adj

    def try_find(n, time_limit):
        # Returns (n_found, colors) or (None, None)
        best_n = 0
        best_colors = None
        
        # Try to find a valid coloring for size n
        # Use multiple restarts
        
        # Random initialization
        # It is better to start from a valid smaller coloring and extend?
        # Or just random?
        # Random for n=50 has ~1000+ mono triangles.
        # SA should be able to fix it.
        
        while time.time() < time_limit:
            # Initialize
            colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            E, adj = calculate_initial_energy(colors, n)
            
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            
            # Local search
            # We can also do a greedy pass first
            # Greedy: pick edge with highest contribution, change to best color
            # But SA is more robust.
            
            # To speed up, we can limit steps per restart
            max_steps = 50000
            steps = 0
            
            while steps < max_steps and time.time() < time_limit:
                steps += 1
                
                # Pick random edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                # Current color
                # We need to map (u, v) to index in colors array to get current color?
                # Or we can maintain a 2D array for colors for O(1) lookup?
                # The return format requires a flat list.
                # Let's maintain a 2D list `col` for speed, and flatten at the end.
                
                # Refactor: maintain `col` 2D array and `adj` bitmasks.
                # Update both on move.
                
                # Re-initializing inside the loop is wasteful.
                # Let's restructure try_find to use local variables.
                pass
            
            # This structure is getting messy. Let's write a cleaner version.
            return best_n, best_colors

    # Clean implementation
    def search(n, deadline):
        # Returns colors list if valid found, else None
        while time.time() < deadline:
            # Init
            col = [[0]*n for _ in range(n)]
            adj = [[0]*n for _ in range(4)]
            E = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    c = random.randint(0, 3)
                    col[i][j] = c
                    col[j][i] = c
                    adj[c][i] |= 1 << j
                    adj[c][j] |= 1 << i
            
            # Calculate initial E
            for c in range(4):
                for u in range(n):
                    for v in range(u+1, n):
                        E += (adj[c][u] & adj[c][v]).bit_count()
            E //= 3
            
            if E == 0:
                return flatten(col, n)
            
            T = 1.0
            T_min = 0.05
            cooling = 0.9995
            
            # Pre-compute edge list? No, random sampling is fine.
            
            while time.time() < deadline:
                # Pick edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v: continue
                if u > v: u, v = v, u
                
                c_old = col[u][v]
                
                # Calculate delta for each new color
                # We want to change to c_new != c_old
                # Delta = K_new - K_old
                # K_old = popcount(adj[c_old][u] & adj[c_old][v])
                # Note: adj[c_old][u] contains v. adj[c_old][v] contains u.
                # The intersection does not include u or v?
                # adj[c_old][u] has bit v set.
                # adj[c_old][v] has bit u set.
                # Intersection: bit v is in adj[c_old][u] but is it in adj[c_old][v]?
                # adj[c_old][v] has bits for neighbors of v. u is a neighbor.
                # Is v a neighbor of v? No.
                # So bit v is NOT in adj[c_old][v].
                # So v is not in the intersection.
                # Similarly u is not in the intersection.
                # So K_old counts triangles (u, v, w) correctly.
                
                k_old = (adj[c_old][u] & adj[c_old][v]).bit_count()
                
                best_delta = 10**9
                best_c = -1
                # Try all other colors
                for c_new in range(4):
                    if c_new == c_old: continue
                    k_new = (adj[c_new][u] & adj[c_new][v]).bit_count()
                    delta = k_new - k_old
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c_new
                
                # Acceptance
                if best_delta < 0 or (best_delta > 0 and random.random() < pow(2.71828, -best_delta / T)):
                    # Apply move
                    # Update col
                    col[u][v] = best_c
                    col[v][u] = best_c
                    
                    # Update adj
                    # Remove from c_old
                    adj[c_old][u] &= ~(1 << v)
                    adj[c_old][v] &= ~(1 << u)
                    # Add to best_c
                    adj[best_c][u] |= 1 << v
                    adj[best_c][v] |= 1 << u
                    
                    E += best_delta
                    
                    if E == 0:
                        return flatten(col, n)
                
                # Cool down
                T *= cooling
                if T < T_min:
                    T = T_min
            
            # Restart with new random init if stuck
    
    def flatten(col, n):
        res = []
        for i in range(n):
            for j in range(i+1, n):
                res.append(col[i][j])
        return res

    # Strategy:
    # 1. Try n=50.
    # 2. If success, try n=51.
    # 3. If n=51 success, return n=51.
    # 4. If n=50 success, return n=50.
    # 5. If neither, return best found (maybe 32 from seed logic? No, seed is 32. If we fail 50, we should probably return the seed 32 to be safe? Or a smaller valid one? The seed 32 is guaranteed valid. But we should try to beat it. If we can't reach 50, we can fall back to 32.)
    
    # Actually, we can just run search(50, deadline).
    # If it returns None, we try search(49)?
    # But 32 is a known good baseline.
    
    # Let's try 50 first.
    colors_50 = search(50, deadline)
    if colors_50 is not None:
        # Try to extend to 51?
        # We can initialize search(51) using the 50-coloring extended randomly?
        # Or just random.
        # Let's try a few restarts for 51 with remaining time.
        if time.time() < deadline - 5.0:
            colors_51 = search(51, deadline)
            if colors_51 is not None:
                return {'n': 51, 'colors': colors_51}
        return {'n': 50, 'colors': colors_50}
    
    # If 50 failed, try 49?
    if time.time() < deadline - 5.0:
        colors_49 = search(49, deadline)
        if colors_49 is not None:
            return {'n': 49, 'colors': colors_49}
            
    # Fallback to seed 32
    # Re-implement seed 32 logic as fallback
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
