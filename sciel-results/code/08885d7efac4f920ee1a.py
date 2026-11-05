import time
import random
from collections import Counter

DEADLINE = 145  # seconds, safety margin before 150s kill

def _time_left():
    return time.time() - _START_TIME < DEADLINE

_START_TIME = time.time()

def construct():
    global _START_TIME
    _START_TIME = time.time()

    def _validate(colors, n):
        """Check no monochromatic triangle."""
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        for c in range(4):
            edges = adj[c]
            adj_list = {i: set() for i in range(n)}
            for (i, j) in edges:
                adj_list[i].add(j)
                adj_list[j].add(i)
            for i in range(n):
                for j in adj_list[i]:
                    if j > i:
                        common = adj_list[i] & adj_list[j]
                        if common:
                            return False
        return True

    # Fallback seed: n=32 Greenwood-Gleason
    def _make_seed():
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
        n_seed = 32
        colors_seed = []
        for i in range(n_seed):
            for j in range(i + 1, n_seed):
                if i // 16 == j // 16:
                    colors_seed.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors_seed.append(3)
        return n_seed, colors_seed

    n_seed, colors_seed = _make_seed()

    # Local search: start with random coloring of K_50, repair mono-triangles
    def _local_search(n_target, max_time):
        """Try to find valid 4-coloring of K_n_target via local search."""
        random.seed(42)
        start_t = time.time()
        
        # Initialize with random coloring
        colors = [random.randint(0, 3) for _ in range(n_target * (n_target - 1) // 2)]
        
        # Precompute edge index lookup
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            return i * n_target - i * (i + 1) // 2 + (j - i - 1)
        
        # Build triangle count function
        def count_triangles(colors_arr):
            """Count monochromatic triangles."""
            count = 0
            # For each pair (i,j), check all k > j
            # More efficient: for each color c, for each edge (i,j) of color c,
            # count common neighbors of i and j in color c
            # Build adjacency per color
            adj = [set() for _ in range(4)]
            idx = 0
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    c = colors_arr[idx]
                    adj[c].add((i, j))
                    idx += 1
            
            # For each color, count triangles
            for c in range(4):
                adj_list = {i: set() for i in range(n_target)}
                for (i, j) in adj[c]:
                    adj_list[i].add(j)
                    adj_list[j].add(i)
                for i in range(n_target):
                    for j in adj_list[i]:
                        if j > i:
                            count += len(adj_list[i] & adj_list[j])
            return count // 3  # each triangle counted 3 times
        
        # Find all mono-triangles for repair
        def find_mono_triangles(colors_arr):
            """Return list of (i, j, k, c) for each mono-triangle."""
            triangles = []
            adj = [set() for _ in range(4)]
            idx = 0
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    c = colors_arr[idx]
                    adj[c].add((i, j))
                    idx += 1
            
            for c in range(4):
                adj_list = {i: set() for i in range(n_target)}
                for (i, j) in adj[c]:
                    adj_list[i].add(j)
                    adj_list[j].add(i)
                for i in range(n_target):
                    for j in adj_list[i]:
                        if j > i:
                            for k in adj_list[i] & adj_list[j]:
                                if k > j:
                                    triangles.append((i, j, k, c))
            return triangles
        
        # Current triangle count
        current_tc = count_triangles(colors)
        
        if current_tc == 0:
            return colors
        
        best_colors = colors[:]
        best_tc = current_tc
        
        # Simulated annealing style local search
        temperature = 1.0
        cooling = 0.9995
        
        while time.time() - start_t < max_time:
            # Find a random mono-triangle
            triangles = find_mono_triangles(colors)
            if not triangles:
                break
            
            tri = random.choice(triangles)
            i, j, k, c = tri
            
            # Try recoloring one of the three edges to break the triangle
            # Pick a random edge from the triangle
            edge_choices = [(i, j), (i, k), (j, k)]
            ei, ej = random.choice(edge_choices)
            
            old_color = colors[edge_idx(ei, ej)]
            new_color = random.choice([0, 1, 2, 3])
            
            if new_color == old_color:
                continue
            
            colors[edge_idx(ei, ej)] = new_color
            
            # Count new triangles (only need to check triangles involving this edge)
            # For efficiency, just count all triangles (n=50, ~19600 triangles, manageable)
            new_tc = count_triangles(colors)
            
            delta = new_tc - current_tc
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < (2.71828 ** (-delta / max(temperature, 0.01)))):
                current_tc = new_tc
                if new_tc < best_tc:
                    best_tc = new_tc
                    best_colors = colors[:]
                    if best_tc == 0:
                        break
            else:
                colors[edge_idx(ei, ej)] = old_color
            
            temperature *= cooling
        
        if best_tc == 0:
            return best_colors
        return None
    
    # Try local search for n=50
    best_n = n_seed
    best_colors = colors_seed
    
    if _time_left():
        result = _local_search(50, 60)
        if result is not None and _validate(result, 50):
            best_n = 50
            best_colors = result
            return {'n': best_n, 'colors': best_colors}
    
    # If 50 failed, try smaller targets
    for target in [45, 40, 35, 33]:
        if not _time_left() or target <= best_n:
            continue
        result = _local_search(target, 30)
        if result is not None and _validate(result, target):
            best_n = target
            best_colors = result
    
    # Fallback to seed
    if best_n < n_seed:
        best_n = n_seed
        best_colors = colors_seed
    
    return {'n': best_n, 'colors': best_colors}
