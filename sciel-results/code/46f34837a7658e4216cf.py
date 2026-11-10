import time
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precompute all 4-term arithmetic progressions (4-APs)
    # A 4-AP is (a, a+d, a+2d, a+3d) with d >= 1 and a+3d <= N
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            # Convert to 0-based indices
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    aps_np = np.array(aps, dtype=np.int32)
    total_aps = len(aps_np)
    
    # 2. Precompute adjacency: for each index i, which APs does it belong to?
    # We need this to calculate the delta of flipping a single bit efficiently.
    # bit_to_aps[i] will be a list of AP indices containing i.
    bit_to_aps = [[] for _ in range(N)]
    for ap_id in range(total_aps):
        for idx in aps_np[ap_id]:
            bit_to_aps[idx].append(ap_id)
    
    # Convert to numpy arrays for faster access? 
    # Actually, Python lists of ints are fine for small counts, 
    # but we can precompute the "other 3 indices" for each (bit, ap) pair 
    # to speed up the delta calculation.
    
    # Let's create a structure: for each bit i, we store a list of tuples (ap_id, other_indices)
    # But calculating delta requires checking the colors of the other 3 indices.
    # A vectorized approach for single flip delta:
    # If we flip bit i, the cost changes for all APs containing i.
    # Let S_i be the set of APs containing i.
    # For each AP in S_i:
    #   - If it was mono, it becomes non-mono (unless the flip makes it mono? No, flipping one bit in a mono AP makes it non-mono).
    #     Wait: If AP is (c, c, c, c) and we flip one c to 1-c, it becomes (1-c, c, c, c) which is non-mono.
    #     So delta = -1 for each mono AP containing i.
    #   - If it was non-mono, it might become mono.
    #     It becomes mono if the other 3 bits are all equal to 1-c_i (the new color of i).
    #     So if the other 3 bits are all equal to each other AND different from c_i, delta = +1.
    #     If the other 3 bits are all equal to c_i, the AP was already mono (handled above).
    #     If the other 3 bits are not all equal, it stays non-mono.
    
    # Let's precompute for each bit i, the list of (ap_id, other_indices_array)
    # This might be memory heavy or slow to build. 
    # Alternative: Just use the precomputed aps_np and bit_to_aps lists.
    # Since N=256 and total_aps ~ 10k, bit_to_aps[i] has ~150 entries.
    # Checking 150 APs per flip is fast.
    
    # Let's optimize the delta calculation function.
    # We will maintain `colors` as a numpy array of 0s and 1s.
    # We will maintain `current_cost` as an integer.
    
    def get_delta_single_flip(colors, i):
        """
        Calculate the change in cost if we flip bit i.
        colors: numpy array of 0/1
        i: index to flip
        returns: delta (integer)
        """
        delta = 0
        c_i = colors[i]
        
        for ap_id in bit_to_aps[i]:
            ap = aps_np[ap_id]
            # Get colors of the 4 elements
            # We need to check if the AP is currently mono, or if it becomes mono.
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            # Check if currently mono
            # Since i is in ap, one of c0..c3 is c_i.
            # If all are equal, it's mono.
            if c0 == c1 and c1 == c2 and c2 == c3:
                # Currently mono. Flipping i makes it non-mono.
                delta -= 1
            else:
                # Currently non-mono.
                # Does it become mono?
                # This happens if the other 3 bits (excluding i) are all equal to 1 - c_i.
                # Let's find the other 3 indices.
                # Instead of searching, we can just check the condition directly.
                # The other 3 bits must be all equal to each other AND different from c_i.
                
                # Let's get the other 3 indices
                # We can precompute this to avoid the loop inside the loop?
                # For now, let's just extract them.
                others = []
                for j in range(4):
                    if ap[j] != i:
                        others.append(ap[j])
                
                co0 = colors[others[0]]
                co1 = colors[others[1]]
                co2 = colors[others[2]]
                
                if co0 == co1 and co1 == co2:
                    # The other 3 are the same.
                    # If they are different from c_i, flipping i makes them all equal.
                    if co0 != c_i:
                        delta += 1
        return delta

    # 3. Simulated Annealing
    # We use a cooling schedule.
    # Start with high temperature, allow worse moves.
    # End with low temperature, only accept better moves (effectively greedy).
    
    best_count = float('inf')
    best_colors = None
    
    # Multiple restarts? SA is good at escaping local minima, so fewer restarts might be needed.
    # Let's do 5 restarts, each with a portion of the time.
    
    rng = np.random.default_rng()
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        # Initialize randomly
        colors = rng.integers(0, 2, size=N)
        # Calculate initial cost
        # Vectorized count
        c0 = colors[aps_np[:, 0]]
        c1 = colors[aps_np[:, 1]]
        c2 = colors[aps_np[:, 2]]
        c3 = colors[aps_np[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        current_cost = np.sum(mono)
        
        if current_cost < best_count:
            best_count = current_cost
            best_colors = colors.copy()
            
        # SA Parameters
        # T_start: High enough to accept many moves.
        # T_end: Low enough to converge.
        # Max steps: Based on time.
        
        t_start = 1.0
        t_end = 0.001
        # We estimate ~10,000 steps per second? 
        # Each step: 1 delta calc (~150 ops) + random choice + accept/reject.
        # Let's run as many steps as possible.
        
        # Cooling schedule: Geometric cooling is standard.
        # T_k = T_start * (T_end/T_start)^(k/K)
        # We don't know K in advance, so we use a fixed cooling factor per step?
        # Or we can define a total number of steps and use that.
        # Let's aim for 50,000 steps per restart.
        max_steps = 50000
        cool_factor = (t_end / t_start) ** (1.0 / max_steps)
        
        current_t = t_start
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Choose a random index to flip
            i = rng.integers(0, N)
            
            delta = get_delta_single_flip(colors, i)
            
            # Acceptance probability
            if delta < 0:
                # Always accept improving moves
                accept = True
            else:
                # Accept worsening moves with probability exp(-delta / T)
                if current_t > 0:
                    prob = np.exp(-delta / current_t)
                    accept = rng.random() < prob
                else:
                    accept = False
            
            if accept:
                colors[i] = 1 - colors[i]
                current_cost += delta
                
                if current_cost < best_count:
                    best_count = current_cost
                    best_colors = colors.copy()
            
            # Cool down
            current_t *= cool_factor
            
            # If T is very low, we can switch to greedy to save time?
            # Or just break if T < threshold and we've been stuck?
            # For now, let it run.
            
    # Final Polish: Greedy local search on the best solution found
    if time.time() < deadline - 2.0 and best_colors is not None:
        colors = best_colors.copy()
        current_cost = best_count
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Try flipping each bit
            for i in range(N):
                delta = get_delta_single_flip(colors, i)
                if delta < 0:
                    colors[i] = 1 - colors[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_count:
                        best_count = current_cost
                        best_colors = colors.copy()
                    # Restart the loop to check other bits again
                    break # Break inner for loop to restart while loop
    
    return best_colors.tolist()

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
    print(res[:10])
