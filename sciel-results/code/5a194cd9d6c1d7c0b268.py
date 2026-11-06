import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    def check_time():
        return time.time() - start_time < deadline

    def count_triangles(colors, n):
        # colors is a list of length n*(n-1)/2
        # We need to map index to (i, j)
        # This is slow for large n if done naively, but we use incremental updates in SA.
        # For initial check, we can just count.
        # Actually, for the greedy method, we track it incrementally.
        # For SA, we track it incrementally.
        # This function is only for final verification or initial random setup.
        pass

    def get_edge_index(i, j, n):
        # i < j
        # Number of edges before row i: i * (n - 1) - i * (i - 1) / 2 ?
        # Let's use a precomputed map or formula.
        # Index of (i, j) in lexicographic order (i, i+1), (i, i+2)...(i, n-1), (i+1, i+2)...
        # Count edges for rows 0 to i-1:
        # Row 0: n-1 edges
        # Row 1: n-2 edges
        # ...
        # Row i-1: n-1-(i-1) = n-i edges
        # Total before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
        # In row i, the edges are (i, i+1), (i, i+2), ..., (i, j).
        # Position of j in row i: j - (i+1)
        # Total index: i*(n-1) - i*(i-1)/2 + (j - i - 1)
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def greedy_add_vertex(n_target, order, initial_colors=None, initial_n=0):
        # order: list of vertex indices to add, or a permutation for the first n_target
        # We build up to n_target.
        # colors is a list. We will expand it.
        # This implementation is simplified: we build from scratch for a given permutation.
        # We maintain the color matrix for the current set of vertices.
        # For efficiency, we use a 2D list for colors of the current subgraph.
        
        # We will try to build a coloring for n_target vertices.
        # We use a greedy approach: add vertices one by one.
        # To make it work better, we can randomize the order of vertices.
        
        # For simplicity and speed, we will just try to construct a coloring for n_target
        # using a greedy algorithm that picks the best color for each new edge.
        
        # Let's use a different strategy: 
        # Start with n=1.
        # Add vertices in the given order.
        
        # To avoid O(N^3) overhead in Python for N=50 (which is fine, 125k ops),
        # we can implement the greedy logic directly.
        
        # However, a full greedy from scratch for every restart might be slow if we do many restarts.
        # Let's try a few restarts.
        
        # This function returns a list of colors if successful, else None.
        pass

    # We will implement a Simulated Annealing approach on a fixed N=50.
    # Initialization: Random.
    # Objective: Minimize number of monochromatic triangles.
    
    N = 50
    num_edges = N * (N - 1) // 2
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Precompute triangle neighbors for fast delta calculation
    # For each edge (u, v), we need to know for each w, the colors of (u, w) and (v, w).
    # This is memory heavy if we store all, but N=50 is small.
    # We can just compute on the fly in O(N) per move.
    
    def get_delta(edge_idx, new_color, colors_list, N):
        # Recover u, v from edge_idx
        # Inverse of get_edge_index
        # i*(N-1) - i*(i-1)/2 + (j-i-1) = idx
        # Let's just iterate i from 0 to N-1 and check if idx falls in that row.
        u = -1
        v = -1
        prev = 0
        for i in range(N):
            row_len = N - 1 - i
            if idx := edge_idx: # syntax error, fix below
                pass
        # Better inverse:
        # i is the largest integer such that i*(N-1) - i*(i-1)/2 <= edge_idx
        # Approx i ~ sqrt(2*edge_idx)
        # Let's just loop, N=50 is small.
        u = 0
        while u < N:
            start = u * (N - 1) - u * (u - 1) // 2
            end = start + (N - 1 - u)
            if start <= edge_idx < end:
                u = u
                v = u + 1 + (edge_idx - start)
                break
            u += 1
        
        if u == -1: return 0 # Should not happen

        delta = 0
        old_color = colors_list[edge_idx]
        if old_color == new_color:
            return 0
            
        for w in range(N):
            if w == u or w == v:
                continue
            # Get colors of (u, w) and (v, w)
            # Ensure u < w and v < w for index calculation, or swap
            if u < w:
                idx_uw = get_edge_index(u, w, N)
            else:
                idx_uw = get_edge_index(w, u, N)
                
            if v < w:
                idx_vw = get_edge_index(v, w, N)
            else:
                idx_vw = get_edge_index(w, v, N)
                
            c_uw = colors_list[idx_uw]
            c_vw = colors_list[idx_vw]
            
            # Triangle (u, v, w) is mono if c_uw == c_vw == c_uv
            # Before: c_uv = old_color
            # After: c_uv = new_color
            
            if c_uw == c_vw:
                if c_uw == old_color:
                    # Was mono, now not (unless new_color == c_uw, but old != new)
                    delta -= 1
                elif c_uw == new_color:
                    # Was not mono, now is
                    delta += 1
        return delta

    # Initial triangle count
    T = 0
    # Calculating initial T is O(N^3). N=50 -> 125k. Fast.
    for i in range(N):
        for j in range(i+1, N):
            idx_ij = get_edge_index(i, j, N)
            c_ij = colors[idx_ij]
            for k in range(j+1, N):
                idx_ik = get_edge_index(i, k, N)
                idx_jk = get_edge_index(j, k, N)
                if colors[idx_ik] == c_ij and colors[idx_jk] == c_ij:
                    T += 1

    best_T = T
    best_colors = colors[:]
    
    if T == 0:
        return {'n': N, 'colors': colors}

    # Simulated Annealing
    current_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    steps_per_temp = 1000
    
    # We need to be careful with time.
    # Estimate: 150s / (time per step).
    # Time per step is O(N) = 50 ops.
    # Python is slow. 1M steps might take 1-2s?
    # Let's run until deadline.
    
    while check_time():
        if current_temp < min_temp:
            # Restart with a new random coloring or perturb best
            # Perturb best is better
            colors = best_colors[:]
            # Randomly flip 10% of edges
            for _ in range(num_edges // 10):
                idx = random.randint(0, num_edges - 1)
                colors[idx] = random.randint(0, 3)
            # Recalculate T
            T = 0
            for i in range(N):
                for j in range(i+1, N):
                    idx_ij = get_edge_index(i, j, N)
                    c_ij = colors[idx_ij]
                    for k in range(j+1, N):
                        idx_ik = get_edge_index(i, k, N)
                        idx_jk = get_edge_index(j, k, N)
                        if colors[idx_ik] == c_ij and colors[idx_jk] == c_ij:
                            T += 1
            current_temp = 1.0
            continue
            
        for _ in range(steps_per_temp):
            if not check_time():
                break
                
            edge_idx = random.randint(0, num_edges - 1)
            old_color = colors[edge_idx]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
                
            delta = get_delta(edge_idx, new_color, colors, N)
            
            if delta < 0:
                # Accept
                colors[edge_idx] = new_color
                T += delta
                if T < best_T:
                    best_T = T
                    best_colors = colors[:]
                    if T == 0:
                        return {'n': N, 'colors': colors}
            else:
                # Reject or accept with probability
                if random.random() < math.exp(-delta / current_temp):
                    colors[edge_idx] = new_color
                    T += delta
                    if T < best_T:
                        best_T = T
                        best_colors = colors[:]
                        if T == 0:
                            return {'n': N, 'colors': colors}
        
        current_temp *= cooling_rate

    # If we didn't find T=0, try smaller N
    # We can try to extract a valid subgraph from best_colors
    # Or just run SA for N=49, 48...
    
    # For simplicity, if N=50 failed, return the best valid smaller one we might have found?
    # We didn't track smaller ones.
    # Let's try N=49 quickly if time permits.
    
    if check_time():
        N = 49
        num_edges = N * (N - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        # Re-run SA for N=49
        # This is a lot of code duplication.
        # Let's just return the best we have for N=50 if it's valid? No, it's not valid if T>0.
        # We must return a VALID coloring.
        # If T>0, the coloring is invalid.
        # So we must find a valid one.
        
        # Fallback: Use the seed program's logic for N=32?
        # The seed program returns a valid N=32.
        # If we can't find N=50, we should return N=32 to ensure score > 0.
        # But we want to beat 1.0.
        
        # Let's try one more thing: Greedy for N=50.
        # If Greedy finds a valid N=50, great.
        
    # If we fail to find a valid N=50, we return the seed result.
    # This ensures we don't score 0.
    
    # Re-implement seed logic for safety
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
    colors_seed = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_seed.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors_seed}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0
    
    def check_time():
        return time.time() - start_time < deadline

    def get_edge_index(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def get_delta(edge_idx, new_color, colors_list, N):
        u = -1
        v = -1
        for i in range(N):
            start = i * (N - 1) - i * (i - 1) // 2
            end = start + (N - 1 - i)
            if start <= edge_idx < end:
                u = i
                v = i + 1 + (edge_idx - start)
                break
        
        if u == -1: return 0

        delta = 0
        old_color = colors_list[edge_idx]
        if old_color == new_color:
            return 0
            
        for w in range(N):
            if w == u or w == v:
                continue
            
            idx_uw = get_edge_index(u, w, N) if u < w else get_edge_index(w, u, N)
            idx_vw = get_edge_index(v, w, N) if v < w else get_edge_index(w, v, N)
            
            c_uw = colors_list[idx_uw]
            c_vw = colors_list[idx_vw]
            
            if c_uw == c_vw:
                if c_uw == old_color:
                    delta -= 1
                elif c_uw == new_color:
                    delta += 1
        return delta

    def run_sa(N, time_limit):
        num_edges = N * (N - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Initial T
        T = 0
        for i in range(N):
            for j in range(i+1, N):
                idx_ij = get_edge_index(i, j, N)
                c_ij = colors[idx_ij]
                for k in range(j+1, N):
                    idx_ik = get_edge_index(i, k, N)
                    idx_jk = get_edge_index(j, k, N)
                    if colors[idx_ik] == c_ij and colors[idx_jk] == c_ij:
                        T += 1

        best_T = T
        best_colors = colors[:]
        
        if T == 0:
            return colors

        current_temp = 1.0
        cooling_rate = 0.999
        min_temp = 0.01
        steps_per_temp = 1000
        
        while time.time() - start_time < time_limit:
            if current_temp < min_temp:
                colors = best_colors[:]
                for _ in range(num_edges // 10):
                    idx = random.randint(0, num_edges - 1)
                    colors[idx] = random.randint(0, 3)
                T = 0
                for i in range(N):
                    for j in range(i+1, N):
                        idx_ij = get_edge_index(i, j, N)
                        c_ij = colors[idx_ij]
                        for k in range(j+1, N):
                            idx_ik = get_edge_index(i, k, N)
                            idx_jk = get_edge_index(j, k, N)
                            if colors[idx_ik] == c_ij and colors[idx_jk] == c_ij:
                                T += 1
                current_temp = 1.0
                continue
                
            for _ in range(steps_per_temp):
                if time.time() - start_time >= time_limit:
                    break
                    
                edge_idx = random.randint(0, num_edges - 1)
                old_color = colors[edge_idx]
                new_color = random.randint(0, 3)
                if new_color == old_color:
                    continue
                    
                delta = get_delta(edge_idx, new_color, colors, N)
                
                if delta < 0:
                    colors[edge_idx] = new_color
                    T += delta
                    if T < best_T:
                        best_T = T
                        best_colors = colors[:]
                        if T == 0:
                            return colors
                else:
                    if random.random() < math.exp(-delta / current_temp):
                        colors[edge_idx] = new_color
                        T += delta
                        if T < best_T:
                            best_T = T
                            best_colors = colors[:]
                            if T == 0:
                                return colors
            current_temp *= cooling_rate
        return None

    # Try N=50
    res = run_sa(50, deadline - 5)
    if res:
        return {'n': 50, 'colors': res}
        
    # Try N=49
    if check_time():
        res = run_sa(49, deadline - 5)
        if res:
            return {'n': 49, 'colors': res}

    # Fallback to seed
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
    colors_seed = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_seed.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors_seed}
# EVOLVE-BLOCK-END
