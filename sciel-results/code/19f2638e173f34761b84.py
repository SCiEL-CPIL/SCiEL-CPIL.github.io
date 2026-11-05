import random
import time
import numpy as np

def construct():
    # Target n. The known record is 50. We aim for 50.
    # If we can't find 50 in time, we might fall back to 32 or lower, 
    # but the heuristic is strong enough to likely find 50.
    n = 50
    num_colors = 4
    
    # Precompute triangle indices for fast access
    # For each edge (i, j), store the list of vertices k such that (i,j,k) is a triangle.
    # Actually, for efficiency in counting monochromatic triangles, 
    # it's better to iterate through edges and check the common neighbors.
    
    def count_mono_triangles(colors):
        """
        Count monochromatic triangles.
        colors is a flat array of length n*(n-1)//2.
        """
        # Build adjacency matrix or list of edges per color
        # To be fast, we can use numpy.
        # But simple loops in Python might be slow for 50 vertices (1225 edges).
        # Let's use a slightly optimized approach.
        
        # Map edge to index
        # i < j
        # idx = i*n - i*(i+1)//2 + j - i - 1
        # This formula is correct for 0-indexed i, j with i<j.
        
        # Alternative: just iterate all triples? 50^3 / 6 = ~20k checks.
        # 20k checks * 1000 iterations = 20M ops. Might be too slow in pure Python.
        
        # Let's use the edge-neighbor approach.
        # For each color c, build a set of edges? No, that's expensive to update.
        
        # Let's stick to a simple but optimized counting method using precomputed triangle lists.
        # For each edge (i,j), we know the potential third vertices k.
        # A triangle (i,j,k) is monochromatic if color(i,j) == color(i,k) == color(j,k).
        
        # Precompute for each edge (i,j) the list of k's.
        # Actually, precomputing the index of (i,k) and (j,k) for each k is useful.
        
        pass # Placeholder, we will implement a faster counter below.

    # Faster counting using numpy?
    # Let's represent the coloring as an n x n matrix (symmetric, 0 on diag).
    # Then a triangle (i,j,k) is mono if M[i,j] == M[i,k] == M[j,k].
    # We can vectorize this?
    
    # Let's try a pure Python approach with precomputed indices for speed.
    # n=50, edges=1225.
    # Precompute for each edge index e (corresponding to (i,j)):
    #   list of (k, idx_ik, idx_jk)
    
    edge_to_ij = []
    idx_map = {}
    for i in range(n):
        for j in range(i+1, n):
            idx = len(edge_to_ij)
            edge_to_ij.append((i, j))
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx # For quick lookup if needed, though we keep i<j
    
    # Precompute triangle neighbors for each edge
    # tri_neighbors[e] = list of (k, idx_ik, idx_jk)
    tri_neighbors = [[] for _ in range(len(edge_to_ij))]
    
    for e, (i, j) in enumerate(edge_to_ij):
        for k in range(n):
            if k == i or k == j:
                continue
            # Get indices for (i,k) and (j,k)
            if i < k:
                idx_ik = idx_map[(i, k)]
            else:
                idx_ik = idx_map[(k, i)]
            if j < k:
                idx_jk = idx_map[(j, k)]
            else:
                idx_jk = idx_map[(k, j)]
            tri_neighbors[e].append((k, idx_ik, idx_jk))

    def count_mono(colors):
        count = 0
        # For each edge (i,j), check if it forms a mono triangle with any k
        # To avoid double counting, we can just sum over all edges and divide?
        # No, each triangle has 3 edges. If we count for each edge if it's part of a mono triangle,
        # each mono triangle is counted 3 times.
        # So total = sum / 3.
        # But we just need to minimize the count, so minimizing the sum is equivalent.
        
        s = 0
        for e in range(len(colors)):
            c_e = colors[e]
            for k, idx_ik, idx_jk in tri_neighbors[e]:
                if colors[idx_ik] == c_e and colors[idx_jk] == c_e:
                    s += 1
        return s // 3

    def get_mono_triangles(colors):
        """Return list of (e, k, idx_ik, idx_jk) for each mono triangle instance (counted 3x)."""
        res = []
        for e in range(len(colors)):
            c_e = colors[e]
            for k, idx_ik, idx_jk in tri_neighbors[e]:
                if colors[idx_ik] == c_e and colors[idx_jk] == c_e:
                    res.append((e, k, idx_ik, idx_jk))
        return res

    def recolor_impact(colors, e, new_color):
        """
        Calculate change in mono triangle count if we change edge e to new_color.
        Returns (delta, new_mono_count)
        """
        old_color = colors[e]
        if old_color == new_color:
            return 0, count_mono(colors)
        
        # We need to check all k in tri_neighbors[e]
        # Current state: c_e = old_color
        # New state: c_e = new_color
        
        # For a specific k:
        # Old: triangle (i,j,k) is mono if colors[idx_ik]==old and colors[idx_jk]==old
        # New: triangle (i,j,k) is mono if colors[idx_ik]==new and colors[idx_jk]==new
        
        delta = 0
        for k, idx_ik, idx_jk in tri_neighbors[e]:
            c_ik = colors[idx_ik]
            c_jk = colors[idx_jk]
            
            was_mono = (c_ik == old_color and c_jk == old_color)
            is_mono = (c_ik == new_color and c_jk == new_color)
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        
        return delta

    # Local Search
    start_time = time.time()
    time_limit = 115 # Leave 5s buffer
    
    best_n_found = 32
    best_colors_32 = None # We can just return the seed if we fail, but let's try to beat it.
    
    # We will try to find a valid 50-vertex coloring.
    # If successful, return n=50.
    # If not, we should probably return a valid smaller one to ensure score > 1.0?
    # The prompt says "Maximize n".
    # If we can't find 50, maybe we can find 33, 34...?
    # The seed is 32.
    # Let's try to find the max n we can solve within time.
    
    # Strategy:
    # 1. Try n=50. If solved, return 50.
    # 2. If not solved in time, try n=40, 35, etc?
    #    Actually, random restarts for n=50 are most likely to succeed if it's possible.
    #    If n=50 is impossible (which it isn't, record is 50), we'd fail.
    #    But since record is 50, it is possible.
    #    Let's just focus on n=50.
    
    # However, if the heuristic gets stuck, we might not find it.
    # To be safe, we can run for a while, and if we find a valid coloring, return it.
    # If we don't, we can fall back to the known 32-vertex construction to guarantee score 1.0.
    # But we want to beat 1.0.
    
    # Let's try n=50.
    
    for attempt in range(200): # Max 200 restarts
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
        
        # Local search loop
        max_iters = 5000
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            mono_tris = get_mono_triangles(colors)
            if not mono_tris:
                # Found a valid coloring!
                return {'n': n, 'colors': colors}
            
            # Pick a random mono triangle from the list
            # Or pick the edge involved in the most mono triangles?
            # Random is simpler and often works well with restarts.
            # Let's pick a random one.
            e, k, idx_ik, idx_jk = random.choice(mono_tris)
            
            current_color = colors[e]
            
            # Try all other colors
            best_delta = float('inf')
            best_new_color = -1
            
            for c in range(4):
                if c == current_color:
                    continue
                delta = recolor_impact(colors, e, c)
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = c
            
            # If best_delta is still positive or zero, we might be in a local minimum
            # or stuck. 
            # If best_delta >= 0, we should probably restart or jump.
            # But let's accept if delta <= 0.
            # If all deltas are positive, we are stuck.
            
            if best_new_color != -1 and best_delta <= 0:
                colors[e] = best_new_color
            else:
                # Stuck or no improvement.
                # Occasional jump: recolor a few random edges
                if random.random() < 0.1:
                    for _ in range(5):
                        re = random.randint(0, len(colors)-1)
                        colors[re] = random.randint(0, 3)
                else:
                    # Restart
                    break
        
        # If loop finished without break (unlikely) or restart, continue outer loop

    # If we failed to find 50 in time, fall back to 32 to ensure validity and score 1.0
    # We can reconstruct the 32-vertex solution from the seed code logic
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
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors_32}

# The construct function is the entry point
