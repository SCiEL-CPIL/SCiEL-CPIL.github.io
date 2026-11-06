import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def get_edge_index(i, j, n):
        # Map (i, j) with i < j to index in 0..n*(n-1)/2 - 1
        # Lexicographic order: (0,1), (0,2)...(0,n-1), (1,2)...
        # Index = i * (n - 1 - i) + (j - i - 1) is not quite right for simple mapping.
        # Let's use a standard mapping.
        # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*n - i*(i+1)/2
        # Offset in row i: j - i - 1
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        
    # Precompute neighbors for faster delta calculation? 
    # For a given edge (u, v), the third vertices are all w != u, v.
    # We need col(v, w) and col(u, w).
    
    def solve_sa(n, time_limit):
        if n < 1:
            return None
        if n == 1:
            return {'n': 1, 'colors': []}
            
        num_edges = n * (n - 1) // 2
        # Initialize colors
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Precompute adjacency lists for vertices to speed up triangle counting?
        # Actually, for delta calculation of edge (u, v), we iterate w.
        # We need to quickly get color of (u, w) and (v, w).
        
        # Helper to get color of (a, b)
        def get_col(a, b):
            if a > b:
                a, b = b, a
            idx = get_edge_index(a, b, n)
            return colors[idx]
            
        # Calculate initial energy
        # O(n^3) is too slow for large n. 
        # For n=50, 125k triangles. 125k * 3 checks. ~400k ops. Fast enough for init.
        energy = 0
        for u in range(n):
            for v in range(u + 1, n):
                cu = get_col(u, v)
                for w in range(v + 1, n):
                    cv = get_col(v, w)
                    cuw = get_col(u, w)
                    if cu == cv == cuw:
                        energy += 1
                    # Also need to check triangles where u is not the smallest?
                    # The loop u < v < w covers all triangles exactly once.
        
        best_energy = energy
        best_colors = colors[:]
        
        if energy == 0:
            return {'n': n, 'colors': colors}
            
        T = 1.0
        cool_rate = 0.9999
        # Estimate steps per second
        # Roughly 10k-50k steps/sec in Python depending on overhead.
        # Let's just run until time_limit.
        
        while time.time() < time_limit:
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            # Decode e_idx to (u, v)
            # This is the inverse of get_edge_index.
            # We can precompute a list of (u, v) pairs for each index.
            # For n=50, 1225 pairs. List of tuples is fine.
            
            # Optimization: Precompute edge pairs
            # But n changes in outer loop. So we do it inside solve_sa.
            
            # We didn't precompute pairs yet. Let's add that.
            
            # Actually, let's restructure to precompute pairs at start of solve_sa.
            
            # Since we are inside the while loop, we need pairs.
            # Let's move pair generation to the start of solve_sa.
            
            pass # Placeholder, will rewrite properly below

    # Rewriting solve_sa with precomputation
    
    def solve_sa_optimized(n, time_limit):
        if n < 1:
            return None
        if n == 1:
            return {'n': 1, 'colors': []}
            
        num_edges = n * (n - 1) // 2
        # Precompute (u, v) for each edge index
        edge_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_pairs.append((i, j))
                
        # Precompute index for (u, v)
        # We already have get_edge_index, but accessing a 2D array or dict might be faster?
        # A 2D list `idx[u][v]` is fast.
        idx = [[0]*n for _ in range(n)]
        for k, (u, v) in enumerate(edge_pairs):
            idx[u][v] = k
            idx[v][u] = k # Symmetric
            
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Calculate initial energy
        energy = 0
        # Iterate over all triangles (u, v, w) with u < v < w
        for u in range(n):
            for v in range(u + 1, n):
                cu = colors[idx[u][v]]
                for w in range(v + 1, n):
                    cv = colors[idx[v][w]]
                    cuw = colors[idx[u][w]]
                    if cu == cv == cuw:
                        energy += 1
                    
        best_energy = energy
        best_colors = colors[:]
        
        if energy == 0:
            return {'n': n, 'colors': colors}
            
        T = 1.0
        # Adjust cooling rate based on expected steps? 
        # Let's use a fixed cooling rate and hope it works.
        # Or use a simple hill climbing with random restarts?
        # SA is better for escaping local minima.
        
        # Estimate steps:
        # Each step involves:
        # 1. Random edge
        # 2. Delta calculation (loop over n-2 vertices)
        # 3. Acceptance decision
        # For n=40, n-2=38. ~40 ops.
        # Python ~5-10M simple ops/sec. So ~100k-200k steps/sec.
        # In 10s, we can do ~1M-2M steps.
        
        while time.time() < time_limit:
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            u, v = edge_pairs[e_idx]
            
            old_c = colors[e_idx]
            # Pick new color different from old
            new_c = random.randint(0, 3)
            while new_c == old_c:
                new_c = random.randint(0, 3)
                
            # Calculate delta energy
            # Delta = sum_{w != u, v} (I(new_c, c(v,w), c(u,w)) - I(old_c, c(v,w), c(u,w)))
            delta = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                # Get colors
                cvw = colors[idx[v][w]]
                cuw = colors[idx[u][w]]
                
                # Contribution before
                if old_c == cvw == cuw:
                    delta -= 1
                # Contribution after
                if new_c == cvw == cuw:
                    delta += 1
                    
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[e_idx] = new_c
                energy += delta
                if energy < best_energy:
                    best_energy = energy
                    best_colors = colors[:]
                    if best_energy == 0:
                        return {'n': n, 'colors': colors}
                        
            T *= 0.9999
            if T < 0.01:
                T = 0.01
                
        if best_energy == 0:
            return {'n': n, 'colors': best_colors}
        return None

    # Main search loop
    # We know n=32 is valid. 
    # Let's try to find valid colorings for n > 32.
    # We will iterate n from 33 upwards.
    # For each n, we allocate a time slice.
    
    best_n = 32
    best_result = None
    
    # Re-construct n=32 result to have a fallback
    # Using the seed logic
    def construct_seed_32():
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
        
    best_result = construct_seed_32()
    
    # Try to improve n
    # Strategy: Try n=33, 34, ... with increasing time budgets?
    # Or just try a fixed set of n values with a time budget each?
    # Since we don't know how hard it is, let's try sequential.
    # If we solve n, try n+1.
    # If we fail n, stop? Or try n+1 anyway? 
    # Usually if you can't solve n, you can't solve n+1.
    # But random initialization might be bad.
    
    # Let's try n from 33 to 50.
    # Time budget per n: 
    # Total time ~ 140s.
    # Let's give 10s per n. 10 * 10s = 100s.
    
    current_n = 33
    while current_n <= 62 and time.time() < deadline - 5:
        # Allocate time slice
        slice_end = min(time.time() + 10, deadline - 5)
        result = solve_sa_optimized(current_n, slice_end)
        if result:
            best_n = current_n
            best_result = result
            current_n += 1
        else:
            # If we fail, maybe try once more with a different seed?
            # Or just break?
            # Let's try once more with a longer slice if we have time?
            # No, let's break to avoid wasting time.
            # However, random failure is possible.
            # Let's try 2 attempts per n.
            pass # We'll implement 2 attempts in the loop logic if needed, 
                 # but for simplicity, let's just break on first fail 
                 # to move to the next higher n? No, if n fails, n+1 likely fails.
            # Actually, let's just break.
            break

    return best_result

# Test locally? No, just output.

# One issue: The seed n=32 is constructed in construct_seed_32. 
# The SA for n=33 might be slow to converge if it starts random.
# Is there a better initialization for SA?
# "Blow-up" of n=32?
# If we have a valid coloring of K_32, we can create K_33 by adding a vertex.
# But SA from random is robust.

# Let's refine the time management.
# If we solve n=33 quickly, we should try n=34.
# If we solve n=40, great.

# The code above looks correct.
# One detail: `math.exp` can be slow. `2 ** (-delta / T)` is faster? Or precompute?
# Delta is integer. T is float.
# We can use `random.random() < 2 ** (-delta / T)`.

# Let's replace math.exp with a faster approximation or just use it.
# For small delta/T, exp is fine.

# Final check on constraints:
# 1 <= n <= 62.
# Colors in {0,1,2,3}.
# No monochromatic triangles.
# Return dict.

# The code seems solid.
