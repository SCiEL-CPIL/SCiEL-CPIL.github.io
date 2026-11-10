import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precompute all 4-term arithmetic progressions
    # Store as list of tuples (i1, i2, i3, i4) with 0-based indices
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # Precompute for each index, which APs it belongs to
    # index_to_aps[i] = list of indices in 'aps' that contain i
    index_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for i in ap:
            index_to_aps[i].append(idx)

    def score_full(colors):
        """Calculate exact number of monochromatic 4-APs."""
        count = 0
        for ap in aps:
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            if c0 == c1 == c2 == c3:
                count += 1
        return count

    # 2. Greedy Initialization
    # Build the coloring greedily. 
    # To make it effective, we can use a random order or sequential.
    # Sequential greedy often creates long blocks. Random order greedy is more balanced.
    # Let's try a "randomized greedy" where we pick a random uncolored index 
    # and assign the value that minimizes local cost, but we need to handle 
    # the fact that APs are not fully determined until all 4 bits are set.
    # 
    # Alternative Greedy: 
    # Start with all 0s. 
    # Repeatedly find the bit flip that reduces the score the most.
    # This is essentially local search from 0s.
    # 
    # Better Initialization: 
    # Use a random coloring, then run a quick "Hill Climbing" phase 
    # to get out of the worst local minima before starting SA.
    
    # Let's initialize with a random coloring
    colors = np.random.randint(0, 2, size=N, dtype=np.int8)
    
    # Quick Hill Climb (Greedy Descent)
    # Try flipping each bit. If it improves, keep it. Repeat until no improvement.
    # This is fast because we use delta scoring.
    
    # Precompute current state for delta calculation
    # is_mono[ap_idx] = True if the AP is currently monochromatic
    is_mono = np.zeros(total_aps, dtype=bool)
    for idx, ap in enumerate(aps):
        c0 = colors[ap[0]]
        if c0 == colors[ap[1]] and c0 == colors[ap[2]] and c0 == colors[ap[3]]:
            is_mono[idx] = True
            
    current_score = np.sum(is_mono)
    
    def get_delta(i, new_val):
        """
        Calculate the change in score if we set colors[i] to new_val.
        Returns delta = new_score - current_score.
        """
        if colors[i] == new_val:
            return 0
        
        delta = 0
        for ap_idx in index_to_aps[i]:
            ap = aps[ap_idx]
            # Determine the state of the other 3 bits
            other_indices = [j for j in ap if j != i]
            c_other = [colors[j] for j in other_indices]
            
            # Current state of this AP
            was_mono = is_mono[ap_idx]
            
            # New state: check if new_val matches all other_indices
            # Since it's a 4-AP, if the other 3 are all equal to new_val, it becomes mono.
            # If the other 3 are NOT all equal to new_val, it is not mono.
            # Note: The other 3 bits don't change.
            
            # If the other 3 are all equal to each other AND equal to new_val, 
            # then the new AP is mono.
            # Otherwise, it is not mono.
            
            # Actually, simpler:
            # New AP is mono iff colors[o1]==colors[o2]==colors[o3]==new_val
            
            # Let's just check the count of new_val among the other 3.
            # If count == 3, then it's mono.
            # If count < 3, it's not mono.
            
            is_new_mono = (c_other[0] == new_val and c_other[1] == new_val and c_other[2] == new_val)
            
            if was_mono and not is_new_mono:
                delta -= 1
            elif not was_mono and is_new_mono:
                delta += 1
            
            # Update is_mono for this step? 
            # No, get_delta should be pure. We update is_mono in the main loop.
            
        return delta

    # Hill Climb Phase
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Shuffle order to avoid bias
        order = list(range(N))
        random.shuffle(order)
        for i in order:
            # Try flipping bit i
            new_val = 1 - colors[i]
            d = get_delta(i, new_val)
            if d < 0:
                # Apply flip
                colors[i] = new_val
                # Update is_mono for all APs containing i
                for ap_idx in index_to_aps[i]:
                    ap = aps[ap_idx]
                    other_indices = [j for j in ap if j != i]
                    c_other = [colors[j] for j in other_indices]
                    # Recalculate mono status for this AP
                    # It is mono if all 4 are equal.
                    # Since we just flipped i, we check if colors[i] matches others.
                    # If colors[i] == c_other[0] == c_other[1] == c_other[2], it is mono.
                    new_is_mono = (colors[i] == c_other[0] and c_other[0] == c_other[1] and c_other[1] == c_other[2])
                    if new_is_mono != is_mono[ap_idx]:
                        is_mono[ap_idx] = new_is_mono
                        current_score += 1 if new_is_mono else -1
                improved = True
    
    best_score = current_score
    best_colors = colors.copy()

    # 3. Simulated Annealing Phase
    # Parameters
    T_start = 5.0
    T_end = 0.01
    alpha = 0.9995 # Cooling rate
    
    # We need to estimate the number of steps we can do.
    # Each step involves get_delta (loop over ~10-20 APs) and potential update.
    # Let's just run until deadline.
    
    T = T_start
    steps_since_improve = 0
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = random.randint(0, N - 1)
        new_val = 1 - colors[i]
        
        d = get_delta(i, new_val)
        
        # Acceptance probability
        if d < 0 or (T > 0 and random.random() < math.exp(-d / T)):
            # Apply flip
            colors[i] = new_val
            
            # Update is_mono and current_score
            for ap_idx in index_to_aps[i]:
                ap = aps[ap_idx]
                other_indices = [j for j in ap if j != i]
                c_other = [colors[j] for j in other_indices]
                new_is_mono = (colors[i] == c_other[0] and c_other[0] == c_other[1] and c_other[1] == c_other[2])
                if new_is_mono != is_mono[ap_idx]:
                    is_mono[ap_idx] = new_is_mono
                    current_score += 1 if new_is_mono else -1
            
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        else:
            steps_since_improve += 1
            
        # Cool down
        T *= alpha
        
        # If stuck for too long, restart temperature? 
        # Or just let it cool. 
        # A simple restart strategy: if T < T_end, reset T to T_start * 0.5?
        # Let's stick to standard SA cooling for now.
        
        # Check time inside loop
        if time.time() > deadline - 1.0:
            break

    return best_colors.tolist()

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Result length: {len(res)}")
    assert len(res) == 256
    assert all(c in [0, 1] for c in res)
    print("Valid")
