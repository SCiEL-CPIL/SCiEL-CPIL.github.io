import time
import random
import numpy as np

def construct():
    """
    Constructs a 2-coloring of [1, 256] minimizing monochromatic 4-term APs.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    N = 256
    
    # 1. Pre-compute all 4-APs and affected indices
    # An AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= 256
    # 0-indexed: 0 <= i, i+3d <= 255 => i+3d < 256
    all_aps = []
    affected = [[] for _ in range(N)]
    
    for d in range(1, (N - 3) // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            idx0 = a
            idx1 = a + d
            idx2 = a + 2 * d
            idx3 = a + 3 * d
            
            ap_idx = len(all_aps)
            all_aps.append((idx0, idx1, idx2, idx3))
            
            # Record which APs affect each position
            affected[idx0].append((ap_idx, 0))
            affected[idx1].append((ap_idx, 1))
            affected[idx2].append((ap_idx, 2))
            affected[idx3].append((ap_idx, 3))
            
    num_aps = len(all_aps)
    
    # Convert all_aps to numpy arrays for faster access? 
    # Actually, list of tuples is fine if we access by index.
    # But let's store the 4 indices in separate lists for speed in delta calc?
    # Or just keep the tuple.
    
    def count_mono_4ap(colors):
        """Count monochromatic 4-APs for a given coloring."""
        c = colors
        count = 0
        for ap in all_aps:
            if c[ap[0]] == c[ap[1]] == c[ap[2]] == c[ap[3]]:
                count += 1
        return count

    def delta_mono(c, p):
        """Calculate change in mono count if we flip position p."""
        old_color = c[p]
        new_color = 1 - old_color
        delta = 0
        
        for ap_idx, pos_in_ap in affected[p]:
            ap = all_aps[ap_idx]
            # Check if currently mono
            if c[ap[0]] == c[ap[1]] == c[ap[2]] == c[ap[3]]:
                delta -= 1
                
            # Check if it becomes mono after flip
            # The other 3 positions must be new_color
            # We can just check the 3 positions that are not p
            # Since we know p is at pos_in_ap, we can check the others
            # But it's simpler to just check if all 4 are new_color? No, c[p] is old.
            # So we check if the other 3 are new_color.
            
            # To avoid branching on pos_in_ap, we can just check:
            # Are all 4 positions equal to new_color, assuming c[p] is flipped?
            # c[ap[0]] == new_color AND c[ap[1]] == new_color ... 
            # But c[ap[pos_in_ap]] is old_color, not new_color.
            # So we just check the 3 others.
            
            # Optimization: 
            # If any of the 3 others is not new_color, it's not mono.
            # Let's just check the 3 others explicitly.
            
            # Get the 3 other indices
            # This is a bit slow with list access. 
            # Alternative: store for each affected entry, the indices of the other 3 positions.
            # Let's pre-compute `affected_other_indices`
            
        # This implementation is too slow if done naively in the inner loop.
        # Let's pre-compute the "other 3 indices" for each affected entry.
        pass

    # Re-do pre-computation to store other indices
    affected_detailed = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(all_aps):
        i0, i1, i2, i3 = ap
        # For i0, others are i1, i2, i3
        affected_detailed[i0].append((ap_idx, [i1, i2, i3]))
        # For i1, others are i0, i2, i3
        affected_detailed[i1].append((ap_idx, [i0, i2, i3]))
        # For i2, others are i0, i1, i3
        affected_detailed[i2].append((ap_idx, [i0, i1, i3]))
        # For i3, others are i0, i1, i2
        affected_detailed[i3].append((ap_idx, [i0, i1, i2]))

    def fast_delta_mono(c, p):
        old_color = c[p]
        new_color = 1 - old_color
        delta = 0
        
        for ap_idx, others in affected_detailed[p]:
            # Check if currently mono
            # We can cache the mono status? No, c changes.
            # Just check the 4 values.
            ap = all_aps[ap_idx]
            if c[ap[0]] == c[ap[1]] == c[ap[2]] == c[ap[3]]:
                delta -= 1
                
            # Check if becomes mono
            # All others must be new_color
            if c[others[0]] == new_color and c[others[1]] == new_color and c[others[2]] == new_color:
                delta += 1
                
        return delta

    # Initial Candidates
    def thue_morse(n):
        # Standard Thue-Morse: t(n) = parity of number of 1s in binary representation of n
        # We want t(i) for i in 0..255
        return [bin(i).count('1') % 2 for i in range(n)]

    candidates = []
    tm = thue_morse(N)
    candidates.append(tm)
    
    # Shifted TM
    tm_shifted = [tm[(i+1)%N] for i in range(N)]
    candidates.append(tm_shifted)
    
    # Complement TM
    candidates.append([1-x for x in tm])
    
    # Random
    for _ in range(10):
        candidates.append([random.randint(0, 1) for _ in range(N)])
        
    # Block patterns (from seed)
    candidates.append([(i // 2) % 2 for i in range(N)])
    candidates.append([(i // 3) % 2 for i in range(N)])
    
    # Evaluate and sort candidates
    scored_candidates = []
    for cand in candidates:
        cnt = count_mono_4ap(cand)
        scored_candidates.append((cnt, cand))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    best_count = scored_candidates[0][0]
    best_colors = scored_candidates[0][1]
    
    # Local Search (Hill Climbing)
    def local_search(colors, time_limit):
        c = list(colors)
        cur_count = count_mono_4ap(c)
        improved = True
        start = time.time()
        
        while improved and time.time() < time_limit:
            improved = False
            # Random order to avoid bias
            order = list(range(N))
            random.shuffle(order)
            
            for p in order:
                if time.time() > time_limit:
                    return c, cur_count
                d = fast_delta_mono(c, p)
                if d < 0:
                    c[p] = 1 - c[p]
                    cur_count += d
                    improved = True
                    
        return c, cur_count

    # Simulated Annealing
    def simulated_annealing(colors, time_limit, start_temp=5.0, end_temp=0.1):
        c = list(colors)
        cur_count = count_mono_4ap(c)
        best_c = list(c)
        best_count = cur_count
        
        start = time.time()
        
        # Estimate steps per second
        # Do a few trial steps
        trial_steps = 100
        t0 = time.time()
        for _ in range(trial_steps):
            p = random.randint(0, N - 1)
            _ = fast_delta_mono(c, p)
        t1 = time.time()
        steps_per_sec = trial_steps / (t1 - t0) if t1 > t0 else 10000
        max_steps = int((time_limit - time.time()) * steps_per_sec)
        
        if max_steps < 10:
            return best_c, best_count
            
        cooling_rate = (end_temp / start_temp) ** (1.0 / max_steps)
        temp = start_temp
        
        for i in range(max_steps):
            if time.time() > time_limit:
                break
                
            p = random.randint(0, N - 1)
            d = fast_delta_mono(c, p)
            
            if d < 0 or random.random() < np.exp(-d / max(temp, 1e-9)):
                c[p] = 1 - c[p]
                cur_count += d
                if cur_count < best_count:
                    best_count = cur_count
                    best_c = list(c)
                    
            temp *= cooling_rate
            
        return best_c, best_count

    # Phase 1: Local search on top candidates
    # We have ~140s. 
    # Use 20s for initial local search on top 3 candidates.
    ls_deadline = start_time + 20
    for i, (cnt, cand) in enumerate(scored_candidates[:3]):
        if time.time() > ls_deadline:
            break
        ls_colors, ls_count = local_search(cand, ls_deadline - i * 5)
        if ls_count < best_count:
            best_count = ls_count
            best_colors = ls_colors

    # Phase 2: Simulated Annealing on the best result
    sa_deadline = start_time + 120
    sa_colors, sa_count = simulated_annealing(best_colors, sa_deadline)
    if sa_count < best_count:
        best_count = sa_count
        best_colors = sa_colors

    # Phase 3: Final Local Search Polish
    final_deadline = deadline
    final_colors, final_count = local_search(best_colors, final_deadline)
    if final_count < best_count:
        best_count = final_count
        best_colors = final_colors

    return best_colors

# EVOLVE-BLOCK-END
