import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Precompute all 4-term APs as index tuples (0-based)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            aps.append((i0, i1, i2, i3))
    
    num_aps = len(aps)
    aps_array = np.array(aps, dtype=np.int32)
    
    # For each position, precompute which APs it participates in
    # pos_to_aps[pos] = list of (ap_idx, role) where role is 0,1,2,3
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_to_aps[i0].append((ap_idx, 0))
        pos_to_aps[i1].append((ap_idx, 1))
        pos_to_aps[i2].append((ap_idx, 2))
        pos_to_aps[i3].append((ap_idx, 3))
    
    def count_mono(colors):
        """Count monochromatic 4-term APs. colors is a numpy array of 0s and 1s."""
        c0 = colors[aps_array[:, 0]]
        c1 = colors[aps_array[:, 1]]
        c2 = colors[aps_array[:, 2]]
        c3 = colors[aps_array[:, 3]]
        mono0 = (c0 == 0) & (c1 == 0) & (c2 == 0) & (c3 == 0)
        mono1 = (c0 == 1) & (c1 == 1) & (c2 == 1) & (c3 == 1)
        return int(np.sum(mono0) + np.sum(mono1))
    
    def compute_deltas(colors):
        """
        Compute the change in mono count if we flip each position.
        Returns an array of shape (N,) where deltas[i] = change if we flip position i.
        
        For each AP, if we flip position p, the AP's mono status changes based on
        how many of the other 3 positions match the new color.
        
        Let's think about this differently. For a given AP (i0,i1,i2,i3) and current colors,
        the AP is mono if all 4 are same. If we flip one position, say i0, then:
        - Before: mono if all same
        - After: i0 is flipped. The AP is mono if i1,i2,i3 are all same AND i0_flipped == i1
        
        Actually, let's compute it per-AP contribution change.
        
        For AP j with positions p0,p1,p2,p3 and current colors c0,c1,c2,c3:
        Let s = c0+c1+c2+c3 (number of 1s)
        mono_before = 1 if s==0 or s==4 else 0
        
        If we flip position pk (k in 0..3), new color at pk is 1-ck.
        New sum s' = s - ck + (1-ck) = s + 1 - 2*ck
        mono_after = 1 if s'==0 or s'==4 else 0
        
        delta_j = mono_after - mono_before
        
        We need to sum delta_j over all APs containing position p.
        
        Let's vectorize: for each position p, find all APs containing p, compute delta for each.
        """
        deltas = np.zeros(N, dtype=np.int32)
        
        for p in range(N):
            # Get all APs containing position p
            if not pos_to_aps[p]:
                continue
            ap_indices = [x[0] for x in pos_to_aps[p]]
            roles = [x[1] for x in pos_to_aps[p]]
            
            if not ap_indices:
                continue
            
            ap_arr = np.array(ap_indices, dtype=np.int32)
            role_arr = np.array(roles, dtype=np.int32)
            
            # Get current colors for all 4 positions of these APs
            c0 = colors[aps_array[ap_arr, 0]]
            c1 = colors[aps_array[ap_arr, 1]]
            c2 = colors[aps_array[ap_arr, 2]]
            c3 = colors[aps_array[ap_arr, 3]]
            
            # Current sum
            s = c0 + c1 + c2 + c3
            
            # Mono before
            mono_before = ((s == 0) | (s == 4)).astype(np.int32)
            
            # Color at the flipped position
            # We need to extract the color at the role position
            all_colors = np.stack([c0, c1, c2, c3], axis=1)  # shape (len, 4)
            flipped_color = all_colors[np.arange(len(ap_arr)), role_arr]
            
            # New sum after flip
            s_new = s - flipped_color + (1 - flipped_color)
            s_new = s + 1 - 2 * flipped_color
            
            # Mono after
            mono_after = ((s_new == 0) | (s_new == 4)).astype(np.int32)
            
            # Delta for each AP
            delta_ap = mono_after - mono_before
            
            # Sum over all APs containing p
            deltas[p] = np.sum(delta_ap)
        
        return deltas
    
    def greedy_polish_vectorized(colors, max_iters=5000):
        """Local search: flip bits to reduce mono count using vectorized deltas."""
        colors = colors.copy()
        current = count_mono(colors)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            deltas = compute_deltas(colors)
            
            # Find the best flip (most negative delta)
            min_delta = np.min(deltas)
            if min_delta >= 0:
                break
            
            # Flip all positions with the most negative delta? Or just the best one?
            # Let's flip the single best position
            best_pos = np.argmin(deltas)
            colors[best_pos] = 1 - colors[best_pos]
            current += min_delta
            improved = True
        
        return colors, current
    
    def simulated_annealing_vectorized(initial_colors, start_temp=1.0, end_temp=0.01, 
                                       cooling_rate=0.999, max_steps=50000):
        """Simulated annealing with vectorized delta computation."""
        colors = initial_colors.copy()
        current = count_mono(colors)
        best_colors = colors.copy()
        best_count = current
        temp = start_temp
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick random position to flip
            pos = random.randint(0, N - 1)
            
            # Compute delta for this specific flip
            # We can reuse compute_deltas but that's expensive. Let's compute just for this pos.
            if not pos_to_aps[pos]:
                continue
            
            ap_indices = [x[0] for x in pos_to_aps[pos]]
            roles = [x[1] for x in pos_to_aps[pos]]
            
            ap_arr = np.array(ap_indices, dtype=np.int32)
            role_arr = np.array(roles, dtype=np.int32)
            
            c0 = colors[aps_array[ap_arr, 0]]
            c1 = colors[aps_array[ap_arr, 1]]
            c2 = colors[aps_array[ap_arr, 2]]
            c3 = colors[aps_array[ap_arr, 3]]
            
            s = c0 + c1 + c2 + c3
            mono_before = ((s == 0) | (s == 4)).astype(np.int32)
            
            all_colors = np.stack([c0, c1, c2, c3], axis=1)
            flipped_color = all_colors[np.arange(len(ap_arr)), role_arr]
            s_new = s + 1 - 2 * flipped_color
            mono_after = ((s_new == 0) | (s_new == 4)).astype(np.int32)
            
            delta = int(np.sum(mono_after - mono_before))
            
            old_c = colors[pos]
            new_c = 1 - old_c
            colors[pos] = new_c
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                current += delta
                if current < best_count:
                    best_count = current
                    best_colors = colors.copy()
            else:
                colors[pos] = old_c
            
            temp *= cooling_rate
            if temp < end_temp:
                temp = end_temp
        
        return best_colors, best_count
    
    def iterated_local_search(initial_colors, max_rounds=20):
        """
        Iterated Local Search: 
        1. Polish to local optimum
        2. Perturb (flip k random bits)
        3. Polish again
        4. Keep best
        """
        colors = initial_colors.copy()
        current = count_mono(colors)
        best_colors = colors.copy()
        best_count = current
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Polish
            polished, score = greedy_polish_vectorized(colors, max_iters=2000)
            if score < best_count:
                best_count = score
                best_colors = polished.copy()
            
            # Perturb: flip 2-5 random bits
            num_flips = random.randint(2, 5)
            flip_positions = random.sample(range(N), num_flips)
            for p in flip_positions:
                polished[p] = 1 - polished[p]
            
            colors = polished
        
        return best_colors, best_count
    
    # Strategy: Multiple restarts with random initial colorings,
    # each followed by iterated local search.
    
    best_colors = None
    best_score = float('inf')
    
    # First, try a structured initial coloring
    # Pattern: alternate in blocks to break APs
    colors_init = np.zeros(N, dtype=np.int8)
    for i in range(N):
        # Use a pattern that avoids 4-term APs
        # e.g., based on base-4 digits or similar
        # Simple: i % 4 < 2 -> 0, else 1
        colors_init[i] = 0 if (i % 4) < 2 else 1
    
    polished, score = iterated_local_search(colors_init, max_rounds=10)
    if score < best_score:
        best_score = score
        best_colors = polished.copy()
    
    # Now do multiple random restarts with ILS
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initial coloring
        random_colors = np.random.randint(0, 2, N, dtype=np.int8)
        
        # ILS from this state
        ils_colors, ils_score = iterated_local_search(random_colors, max_rounds=15)
        
        if ils_score < best_score:
            best_score = ils_score
            best_colors = ils_colors.copy()
        
        # If we have time, try simulated annealing from the best found so far
        if time.time() < deadline - 15:
            sa_colors, sa_score = simulated_annealing_vectorized(
                best_colors, start_temp=0.3, end_temp=0.01, 
                cooling_rate=0.9995, max_steps=10000
            )
            if sa_score < best_score:
                best_score = sa_score
                best_colors = sa_colors.copy()
    
    # Final polish on the best solution
    if time.time() < deadline - 3:
        final_colors, final_score = greedy_polish_vectorized(best_colors, max_iters=5000)
        if final_score < best_score:
            best_score = final_score
            best_colors = final_colors.copy()
    
    # Convert to list of ints
    return [int(c) for c in best_colors]
