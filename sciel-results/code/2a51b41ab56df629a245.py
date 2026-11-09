import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256

    # Precompute all 4-term APs
    # a, a+d, a+2d, a+3d with 1 <= a, a+3d <= 256
    # 0-indexed: 0..255
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    M = len(aps)
    if M == 0:
        return [0] * n

    aps_arr = np.array(aps, dtype=np.int32)
    
    # Precompute for each position, the indices of the APs it belongs to
    # and for each such AP, the indices of the other 3 elements
    # We will build a structure to allow vectorized delta calculation
    
    # For vectorized delta calculation:
    # We iterate over all APs. For each AP, we check if the 3 "other" elements 
    # are equal. If so, we update the delta for the 4th element.
    
    # To make this fast, we precompute:
    # 1. For each AP index j, the 4 indices.
    # 2. We need to accumulate deltas for each of the 4 positions.
    
    # Let's create arrays for the "other" elements for each position in an AP.
    # This is tricky to vectorize directly for "all positions at once" without 
    # duplicating data heavily, but we can do it efficiently.
    
    # Alternative: 
    # For each position i, store list of APs it's in.
    # But vectorizing the "check if others are equal" part is key.
    
    # Let's try a different vectorization strategy:
    # Create a 4 x M array `aps_arr` where aps_arr[k, j] is the k-th element of AP j.
    
    # To compute delta for ALL positions simultaneously:
    # We need to sum contributions from all APs.
    # Contribution of AP j to position i (where i is one of the 4 indices):
    # Let others be the other 3 indices.
    # If x[others] are all equal (say to c), then:
    #   if x[i] == c, delta -= 1
    #   else delta += 1
    # If x[others] are not all equal, delta += 0.
    
    # We can compute a mask for "others are equal" for each AP.
    # For AP j, let indices be i0, i1, i2, i3.
    # others for i0 are i1, i2, i3.
    # Are x[i1], x[i2], x[i3] all equal?
    
    # We can precompute the "other" indices for each of the 4 roles.
    # role 0: others are cols 1, 2, 3
    # role 1: others are cols 0, 2, 3
    # role 2: others are cols 0, 1, 3
    # role 3: others are cols 0, 1, 2
    
    # Let's create 4 arrays of shape (4, M) or similar?
    # Actually, we can just compute the delta for the 4 positions of each AP and scatter-add.
    
    # Precompute "other" indices for each role
    # others[role, j] is a tuple of 3 indices? No, numpy arrays.
    # Let's create 4 arrays `others_arr` of shape (4, 3, M)
    # others_arr[role, k, j] is the index of the k-th other element in AP j for role 'role'
    
    # This is memory heavy but M is ~10000-20000. 4*3*20000*4 bytes = ~1MB. Fine.
    
    # Let's build it
    # aps_arr is (M, 4)
    # We want for each AP j and role r, the 3 indices that are NOT r.
    
    # Instead of complex indexing, let's just use the fact that M is small enough
    # to do a loop over APs in Python if the inner part is very fast, 
    # BUT vectorizing over APs is better.
    
    # Let's try to vectorize the delta calculation for ALL positions in one go.
    # We create a `delta` array of zeros of length n.
    # For each AP j:
    #   i0, i1, i2, i3 = aps_arr[j]
    #   c0 = x[i1] == x[i2] == x[i3]
    #   if c0:
    #      if x[i0] == x[i1]: delta[i0] -= 1 else delta[i0] += 1
    #   ... and so on for i1, i2, i3
    
    # This loop over M (~13k) is fast in Python if we use numpy scalars? 
    # No, Python loop over 13k items is ~1ms. 
    # If we do 1000 iterations, that's 1s. 
    # With 150s limit, we can do ~100-200 full greedy passes per start.
    # And we can do many starts.
    
    # Let's optimize the inner loop.
    
    def count_mono(x):
        x0 = x[aps_arr[:, 0]]
        x1 = x[aps_arr[:, 1]]
        x2 = x[aps_arr[:, 2]]
        x3 = x[aps_arr[:, 3]]
        return int(np.sum((x0 == x1) & (x1 == x2) & (x2 == x3)))
    
    def greedy_polish(x, max_iter=1000):
        x = x.copy()
        cur_count = count_mono(x)
        
        # Precompute others indices for vectorization? 
        # Let's just use the loop, it's likely fast enough for M~13000.
        
        for _ in range(max_iter):
            if (time.time() - start_time) > deadline - 5.0:
                break
                
            delta = np.zeros(n, dtype=np.int32)
            
            # Vectorized check for each AP
            # Get colors for all 4 positions of all APs
            x0 = x[aps_arr[:, 0]]
            x1 = x[aps_arr[:, 1]]
            x2 = x[aps_arr[:, 2]]
            x3 = x[aps_arr[:, 3]]
            
            # Check if others are equal for each role
            # Role 0 (pos 0): others 1,2,3. Equal if x1==x2 and x2==x3
            eq_123 = (x1 == x2) & (x2 == x3)
            # If eq_123, delta[0] changes.
            # If x0 == x1, delta -= 1, else += 1
            # delta_change_0 = np.where(eq_123, np.where(x0 == x1, -1, 1), 0)
            
            # We need to accumulate these into the `delta` array at the correct indices.
            # np.add.at is the way to do scatter-add.
            
            # Role 0
            d0 = np.where(eq_123, np.where(x0 == x1, -1, 1), 0)
            np.add.at(delta, aps_arr[:, 0], d0)
            
            # Role 1 (pos 1): others 0,2,3. Equal if x0==x2 and x2==x3
            eq_023 = (x0 == x2) & (x2 == x3)
            d1 = np.where(eq_023, np.where(x1 == x0, -1, 1), 0)
            np.add.at(delta, aps_arr[:, 1], d1)
            
            # Role 2 (pos 2): others 0,1,3. Equal if x0==x1 and x1==x3
            eq_013 = (x0 == x1) & (x1 == x3)
            d2 = np.where(eq_013, np.where(x2 == x0, -1, 1), 0)
            np.add.at(delta, aps_arr[:, 2], d2)
            
            # Role 3 (pos 3): others 0,1,2. Equal if x0==x1 and x1==x2
            eq_012 = (x0 == x1) & (x1 == x2)
            d3 = np.where(eq_012, np.where(x3 == x0, -1, 1), 0)
            np.add.at(delta, aps_arr[:, 3], d3)
            
            # Find the best flip
            min_delta = np.min(delta)
            if min_delta >= 0:
                break
            
            # Flip all positions with min_delta (ties are fine, or just one)
            # Flipping multiple is safer to escape plateaus
            flip_mask = delta == min_delta
            x[flip_mask] = 1 - x[flip_mask]
            
            # Update count
            # Instead of recounting, we can sum the deltas applied
            # But since we flipped multiple, and deltas were calculated independently,
            # the sum of deltas is the change IF the flips don't interact in the delta calculation.
            # Wait, the delta calculation assumes only ONE flip at a time.
            # If we flip multiple, the deltas might be invalid for the combined move.
            # However, for local search, flipping the single best is standard.
            # Flipping all ties is a heuristic. Let's just recount to be safe, 
            # or only flip one.
            # Let's flip only ONE to be safe and accurate with delta.
            # Actually, if we flip multiple, we MUST recount.
            # Recount is O(M) vectorized, which is fast.
            
            cur_count = count_mono(x)
            
        return x, cur_count

    def simulated_annealing(x, cur_count, T_init, cooling, min_T, max_steps):
        x = x.copy()
        best_x = x.copy()
        best_count = cur_count
        T = T_init
        
        for step in range(max_steps):
            if (time.time() - start_time) > deadline - 5.0:
                break
                
            # Calculate delta for all positions
            x0 = x[aps_arr[:, 0]]
            x1 = x[aps_arr[:, 1]]
            x2 = x[aps_arr[:, 2]]
            x3 = x[aps_arr[:, 3]]
            
            eq_123 = (x1 == x2) & (x2 == x3)
            d0 = np.where(eq_123, np.where(x0 == x1, -1, 1), 0)
            delta = np.zeros(n, dtype=np.int32)
            np.add.at(delta, aps_arr[:, 0], d0)
            
            eq_023 = (x0 == x2) & (x2 == x3)
            d1 = np.where(eq_023, np.where(x1 == x0, -1, 1), 0)
            np.add.at(delta, aps_arr[:, 1], d1)
            
            eq_013 = (x0 == x1) & (x1 == x3)
            d2 = np.where(eq_013, np.where(x2 == x0, -1, 1), 0)
            np.add.at(delta, aps_arr[:, 2], d2)
            
            eq_012 = (x0 == x1) & (x1 == x2)
            d3 = np.where(eq_012, np.where(x3 == x0, -1, 1), 0)
            np.add.at(delta, aps_arr[:, 3], d3)
            
            # Pick a random position
            pos = random.randint(0, n - 1)
            d = delta[pos]
            
            accept = False
            if d < 0:
                accept = True
            elif d == 0:
                accept = True
            else:
                if T > min_T:
                    prob = math.exp(-d / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                x[pos] = 1 - x[pos]
                cur_count += d
                if cur_count < best_count:
                    best_x = x.copy()
                    best_count = cur_count
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return best_x, best_count

    # Initializations
    rng = np.random.default_rng(42)
    random.seed(42)
    
    initializations = []
    
    # 1. Thue-Morse
    def thue_morse(n):
        return bin(n).count('1') % 2
    tm_colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    initializations.append(tm_colors.copy())
    
    # 2. Random
    initializations.append(rng.integers(0, 2, size=n, dtype=np.int8))
    
    # 3. Block patterns
    for block_size in [2, 3, 4, 5, 8, 16]:
        block_colors = np.array([(i // block_size) % 2 for i in range(n)], dtype=np.int8)
        initializations.append(block_colors)
        
    # 4. Thue-Morse with noise
    for _ in range(5):
        flipped_tm = tm_colors.copy()
        mask = rng.random(n) < 0.1
        flipped_tm[mask] = 1 - flipped_tm[mask]
        initializations.append(flipped_tm)
        
    best_c = None
    best_count = float('inf')
    
    # Phase 1: Greedy Polish
    for init_colors in initializations:
        if (time.time() - start_time) > deadline - 20.0:
            break
            
        c, count = greedy_polish(init_colors, max_iter=500)
        if count < best_count:
            best_c = c.copy()
            best_count = count
            
    # Phase 2: Simulated Annealing from Best
    if (time.time() - start_time) < deadline - 10.0:
        sa_col, sa_count = simulated_annealing(
            best_c, best_count,
            T_init=2.0, cooling=0.9999, min_T=0.05,
            max_steps=20000
        )
        if sa_count < best_count:
            best_c = sa_col.copy()
            best_count = sa_count
            
        # Polish after SA
        if (time.time() - start_time) < deadline - 5.0:
            final_c, final_count = greedy_polish(best_c, max_iter=200)
            if final_count < best_count:
                best_c = final_c
                best_count = final_count
                
    # Phase 3: More SA runs from perturbations
    while (time.time() - start_time) < deadline - 10.0:
        # Perturb best solution
        perturbed = best_c.copy()
        num_flips = random.randint(5, 15)
        for _ in range(num_flips):
            p = random.randint(0, n - 1)
            perturbed[p] = 1 - perturbed[p]
            
        sa_col, sa_count = simulated_annealing(
            perturbed, count_mono(perturbed),
            T_init=1.5, cooling=0.9999, min_T=0.05,
            max_steps=10000
        )
        
        # Polish
        if (time.time() - start_time) < deadline - 5.0:
            final_c, final_count = greedy_polish(sa_col, max_iter=100)
            if final_count < best_count:
                best_c = final_c
                best_count = final_count
                
    # Final Polish
    if (time.time() - start_time) < deadline - 2.0:
        final_c, final_count = greedy_polish(best_c, max_iter=100)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count
            
    return [int(c) for c in best_c]
