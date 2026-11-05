"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    def count_mono_triangles(adj, n):
        """Count monochromatic triangles using bitmask intersections."""
        # adj is a list of lists of ints (0-3) or a numpy array. 
        # For speed, we use bitmasks for neighbors of each color.
        # masks[c][i] is a bitmask of vertices j such that adj[i][j] == c
        masks = [[0] * n for _ in range(4)]
        
        # Build masks
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
                
        total = 0
        for c in range(4):
            m = masks[c]
            for i in range(n):
                for j in range(i + 1, n):
                    if m[i] & (1 << j):
                        # Edge (i,j) is color c. Count common neighbors k > j
                        common = m[i] & m[j]
                        # Mask out i and j and vertices <= j to avoid double counting
                        # We only care about k > j
                        # common has bits set for all common neighbors.
                        # We want count of bits > j.
                        # Since i < j, bit i is not in m[j] (no self loops in mask logic? 
                        # wait, masks[c][j] contains i if adj[j][i]==c. Yes.
                        # So common has bit i and bit j? No, m[i] doesn't have bit i. m[j] doesn't have bit j.
                        # But m[i] has bit j if adj[i][j]==c. m[j] has bit i if adj[j][i]==c.
                        # So common has bit i and bit j? 
                        # m[i] & m[j]: 
                        # bit i: m[i] has 0 at i. So bit i is 0 in common.
                        # bit j: m[j] has 0 at j. So bit j is 0 in common.
                        # So common contains k != i, j.
                        # We need k > j to count each triangle once (i<j<k).
                        mask_gt_j = ((1 << (n - j - 1)) - 1) << (j + 1)
                        count = (common & mask_gt_j).bit_count()
                        total += count
        return total

    def count_mono_triangles_fast(adj, n):
        """
        Optimized count. 
        adj is a flat array of length n*(n-1)//2 or similar? 
        Let's stick to a 2D numpy array for simplicity but optimize the loop.
        Actually, for n up to 50, n^3/6 is ~20k ops. Python loops are slow.
        We can use vectorized numpy operations for the triangle count if we store adjacency matrices per color.
        """
        # Create binary adjacency matrices for each color
        # A_c[i,j] = 1 if color(i,j) == c
        # Triangles in color c = trace(A_c @ A_c @ A_c) / 6
        # This is O(4 * n^3) which is fast with numpy.
        
        # adj is 2D numpy array
        total = 0
        for c in range(4):
            A = (adj == c).astype(np.float32)
            # Count triangles: sum over i<j<k of A[i,j]*A[j,k]*A[i,k]
            # trace(A^3) = 6 * (# triangles)
            # A^3 = A @ A @ A
            # We can compute A @ A first (matrix mult), then element-wise multiply with A, then sum.
            # (A @ A)[i,k] = sum_j A[i,j] A[j,k]
            # (A @ A * A)[i,k] = sum_j A[i,j] A[j,k] A[i,k]
            # Sum over all i,k gives sum_{i,j,k} A[i,j]A[j,k]A[k,i] = 6 * Triangles
            A2 = A @ A
            total += np.sum(A2 * A)
        return total // 6

    def generate_greedy(n):
        """Generate a greedy coloring to start SA."""
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                # Pick color that minimizes immediate triangles
                best_c = 0
                best_tri = float('inf')
                for c in range(4):
                    # Count triangles formed by (i,j) with color c
                    # Triangles are (k,i,j) where k<i.
                    # Need adj[k,i]==c and adj[k,j]==c
                    count = 0
                    for k in range(i):
                        if adj[k, i] == c and adj[k, j] == c:
                            count += 1
                    if count < best_tri:
                        best_tri = count
                        best_c = c
                adj[i, j] = best_c
                adj[j, i] = best_c
        return adj

    def generate_random(n):
        adj = np.random.randint(0, 4, size=(n, n), dtype=np.int8)
        for i in range(n):
            adj[i, i] = 0
            for j in range(i + 1, n):
                adj[j, i] = adj[i, j]
        return adj

    best_score = float('inf')
    best_adj = None
    
    # Try different starting sizes and initializations
    # We want to maximize n such that score == 0.
    # Strategy: Start with small n, find a 0-score coloring, then extend?
    # Or just run SA for n=50 directly?
    # SA for n=50 is 1225 edges. 
    # Time limit 115s.
    
    # Let's try to find a valid coloring for n=50.
    # If we fail, try n=49, etc.
    # But the task asks to maximize n. 
    # The seed program returns 32. 
    # If we can reliably get 50, that's a huge jump.
    # If we get stuck, we should return the best valid n found (score 0) or the one with fewest triangles if no valid one found?
    # The evaluator measures "raw value" which is n. 
    # "A valid 4-colouring... certifies R_4(3) > n."
    # "Maximize n."
    # If the coloring is invalid, does it score 0? 
    # "An invalid result... scores 0."
    # So we MUST return a valid coloring (0 triangles) to score > 0.
    # If we can't find a valid one for n, we must drop n until we do.
    
    # Let's run SA for n=50. If it finds 0 triangles, return 50.
    # If not, try n=49, etc.
    # To save time, we can run SA in parallel? No, single process.
    # We can run SA for n=50 for 110s. If it fails, we have no time for smaller n.
    # Better: Run SA for n=50. If it reaches 0, great. 
    # If not, we need a fallback. 
    # The greedy construction for n=32 is known to be valid? 
    # The seed program uses a specific construction for 32.
    # Let's use the seed's 32-vertex coloring as a fallback.
    
    # Re-implement seed's 32-vertex coloring for fallback
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

    n_start = 32
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    adj_32 = np.zeros((32, 32), dtype=np.int8)
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            c = colors_32[idx]
            adj_32[i, j] = c
            adj_32[j, i] = c
            idx += 1
            
    # Verify adj_32 is valid
    if count_mono_triangles_fast(adj_32, 32) != 0:
        # Should not happen, but just in case
        pass
        
    best_adj = adj_32
    best_n = 32
    
    # Try to improve n using SA
    # We will try n=50 down to 33.
    # If we find a valid coloring for any n > 32, we update best.
    # To maximize n, we should try higher n first.
    # But if n=50 takes all time and fails, we might miss n=49 which is easier?
    # Unlikely. If 50 is hard, 49 is easier.
    # However, SA is stochastic. 
    # Let's allocate time based on n.
    # Time per n: 115 / (50 - 32 + 1) ~ 6s per n? That's very short for SA to solve n=50.
    # Solving n=50 from scratch with SA might take longer.
    # Maybe we should focus all time on n=50?
    # If n=50 fails, we return 32.
    # Is there a way to extend 32 to 50?
    # The seed program tries to add vertices one by one. It stalls.
    # SA on the whole graph is a different approach.
    
    # Let's try n=50 with a generous time budget.
    # If it doesn't reach 0, we can try to "repair" or just accept failure for that n.
    # But we need a valid result.
    
    # Strategy:
    # 1. Try n=50 for 100s.
    # 2. If success (0 triangles), return 50.
    # 3. If fail, try n=49 for 10s? No, 10s is too short.
    # 4. If fail, return 32.
    
    # To increase chances, we can use multiple restarts in the time budget.
    
    target_n = 50
    current_adj = generate_greedy(target_n)
    current_score = count_mono_triangles_fast(current_adj, target_n)
    
    if current_score == 0:
        return {'n': target_n, 'colors': [int(current_adj[i,j]) for i in range(target_n) for j in range(i+1, target_n)]}
        
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    # Precompute indices for faster access? 
    # Numpy matrix mult is fast enough for n=50.
    
    # To speed up, we can update the triangle count incrementally?
    # Changing one edge (i,j) affects triangles involving (i,j).
    # Number of triangles involving (i,j) is at most n-2.
    # Recalculating total from scratch is O(n^3) ~ 125k ops. 
    # Incremental is O(n) ~ 50 ops. 
    # Huge speedup!
    
    def count_triangles_involving_edge(adj, i, j, c_new, n):
        """Count triangles containing edge (i,j) if it were color c_new."""
        # Triangles are (k,i,j) where k != i,j.
        # Need adj[k,i] == c_new and adj[k,j] == c_new.
        count = 0
        for k in range(n):
            if k == i or k == j:
                continue
            if adj[k, i] == c_new and adj[k, j] == c_new:
                count += 1
        return count

    # We need to maintain the total triangle count.
    # When we change edge (i,j) from c_old to c_new:
    # 1. Remove triangles that used c_old.
    #    These are triangles (k,i,j) where adj[k,i]==c_old and adj[k,j]==c_old.
    # 2. Add triangles that use c_new.
    #    These are triangles (k,i,j) where adj[k,i]==c_new and adj[k,j]==c_new.
    #    BUT, we must be careful: the edge (i,j) itself is changing.
    #    The triangles counted in step 1 and 2 are exactly those involving edge (i,j).
    #    Other triangles are unaffected.
    
    total_tris = current_score
    
    # Randomly pick edges to swap
    # We need a list of edges
    edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
    
    for step in range(10000000): # Large cap, controlled by time
        if time.time() > deadline - 5:
            break
            
        # Pick random edge
        i, j = random.choice(edges)
        c_old = current_adj[i, j]
        
        if c_old == 0:
            continue # Or pick a different one
        
        c_new = random.randint(0, 3)
        if c_new == c_old:
            continue
            
        # Calculate delta
        # Triangles removed: those with c_old
        # Triangles added: those with c_new
        
        # Optimize: Precompute neighbor color counts?
        # For each vertex k, we have adj[k,i] and adj[k,j].
        # We can just loop k.
        
        delta = 0
        # Remove c_old triangles
        for k in range(target_n):
            if k == i or k == j:
                continue
            if current_adj[k, i] == c_old and current_adj[k, j] == c_old:
                delta += 1
        # Add c_new triangles
        for k in range(target_n):
            if k == i or k == j:
                continue
            if current_adj[k, i] == c_new and current_adj[k, j] == c_new:
                delta -= 1
                
        # delta is change in total triangles. (added - removed)
        # Wait, my loop above:
        # First loop adds to delta for each c_old triangle. So delta = #removed.
        # Second loop subtracts from delta for each c_new triangle. So delta = #removed - #added.
        # We want new_total = old_total - #removed + #added.
        # So delta_cost = #added - #removed.
        # My delta variable is #removed - #added.
        # So cost_change = -delta.
        
        cost_change = -delta
        
        if cost_change < 0:
            accept = True
        else:
            if random.random() < np.exp(-cost_change / T):
                accept = True
            else:
                accept = False
                
        if accept:
            current_adj[i, j] = c_new
            current_adj[j, i] = c_new
            total_tris += cost_change
            
            if total_tris == 0:
                # Found valid coloring!
                return {'n': target_n, 'colors': [int(current_adj[i,j]) for i in range(target_n) for j in range(i+1, target_n)]}
                
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart? 
            # If we restart, we lose the current state. 
            # Maybe we should keep the best state found so far?
            # For now, just continue with low T.
            
    # If we didn't find 0 triangles for n=50, try n=49?
    # Time is likely up. 
    # We can check time.
    if time.time() < deadline - 20:
        target_n = 49
        current_adj = generate_greedy(target_n)
        total_tris = count_mono_triangles_fast(current_adj, target_n)
        
        if total_tris == 0:
             return {'n': target_n, 'colors': [int(current_adj[i,j]) for i in range(target_n) for j in range(i+1, target_n)]}
             
        T = 1.0
        edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
        
        while time.time() < deadline - 5:
            i, j = random.choice(edges)
            c_old = current_adj[i, j]
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
                
            delta = 0
            for k in range(target_n):
                if k == i or k == j:
                    continue
                if current_adj[k, i] == c_old and current_adj[k, j] == c_old:
                    delta += 1
            for k in range(target_n):
                if k == i or k == j:
                    continue
                if current_adj[k, i] == c_new and current_adj[k, j] == c_new:
                    delta -= 1
                    
            cost_change = -delta
            
            if cost_change < 0 or random.random() < np.exp(-cost_change / T):
                current_adj[i, j] = c_new
                current_adj[j, i] = c_new
                total_tris += cost_change
                
                if total_tris == 0:
                    return {'n': target_n, 'colors': [int(current_adj[i,j]) for i in range(target_n) for j in range(i+1, target_n)]}
                    
            T *= 0.999
            if T < 0.001:
                T = 0.001

    # Fallback to 32
    return {'n': 32, 'colors': [int(adj_32[i,j]) for i in range(32) for j in range(i+1, 32)]}
# EVOLVE-BLOCK-END
