import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 118.0
    
    # 1. Precompute APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            # indices 0-based: a-1, a-1+d, a-1+2d, a-1+3d
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps, dtype=np.int32)
    M = len(aps)
    
    # 2. Precompute neighbors for each index
    # neighbors[i] = list of AP indices that include position i
    neighbors = [[] for _ in range(n)]
    for idx, p in enumerate(aps):
        for pos in p:
            neighbors[pos].append(idx)
    
    # Convert to numpy arrays for faster access if needed, or keep lists
    # Lists are fine for small sizes.
    
    def count_mono(c):
        # Vectorized count
        # c is array of 0/1
        # aps is (M, 4)
        vals = c[aps] # (M, 4)
        # Check if all 4 are equal
        # vals[:, 0] == vals[:, 1] == vals[:, 2] == vals[:, 3]
        mono = (vals[:, 0] == vals[:, 1]) & (vals[:, 1] == vals[:, 2]) & (vals[:, 2] == vals[:, 3])
        return mono.sum()

    def calc_delta(i, c):
        # Calculate change in cost if we flip c[i]
        delta = 0
        for ap_idx in neighbors[i]:
            p = aps[ap_idx]
            # Get current values
            v = c[p]
            # Check if currently mono
            is_mono_before = (v[0] == v[1] == v[2] == v[3])
            
            # Flip i
            # We need to know which position i is in p
            # Find pos of i in p
            # p is (x0, x1, x2, x3)
            # i is one of them.
            # new_v = v.copy()
            # new_v[i == p[0]] = 1 - new_v[i == p[0]] ... this is slow.
            
            # Faster:
            # v is array of 4 ints.
            # Find index k such that p[k] == i
            # We can precompute for each AP and each of its 4 positions, 
            # but it's easier to just check.
            
            # Actually, we can just count how many 1s and 0s are in the AP.
            # If it's mono, it's either 0000 or 1111.
            # If we flip one bit:
            # 0000 -> 1000 (not mono) => delta -1
            # 1111 -> 0111 (not mono) => delta -1
            # If it's not mono:
            # It might become mono?
            # Only if it was 0001 or 1110 (3 same, 1 diff).
            # If we flip the diff one, it becomes 0000 or 1111 => delta +1.
            # If we flip a same one, it becomes 1001 etc (2 same, 2 diff) => delta 0.
            
            # So:
            sum_v = v.sum()
            if sum_v == 0 or sum_v == 4:
                # Currently mono
                # Flipping makes it non-mono (3-1 split)
                delta -= 1
            elif sum_v == 1 or sum_v == 3:
                # 3-1 split
                # If we flip the minority bit, it becomes mono.
                # If we flip a majority bit, it becomes 2-2.
                # We need to know if we are flipping the minority bit.
                if sum_v == 1:
                    # Minority is 1. If c[i] == 1, we flip it to 0.
                    # Then sum becomes 0 => mono.
                    if c[i] == 1:
                        delta += 1
                    else:
                        delta -= 1 # Wait.
                        # If c[i] == 0 (majority), flip to 1.
                        # Sum becomes 2. Not mono.
                        # Old state: not mono. New state: not mono.
                        # Delta 0.
                        # Let's re-evaluate.
                        # Old: 0001 (not mono).
                        # Flip a 0 -> 1001 (not mono). Delta 0.
                        # Flip the 1 -> 0000 (mono). Delta +1.
                else: # sum_v == 3
                    # Majority is 1. If c[i] == 0 (minority), flip to 1 -> 1111 (mono).
                    if c[i] == 0:
                        delta += 1
                    else:
                        delta = 0 # Flip majority 1 to 0 -> 0111 (not mono). Old not mono. Delta 0.
            else:
                # sum_v == 2. 2-2 split.
                # Flip any bit -> 3-1 split.
                # Old not mono. New not mono.
                delta = 0
                
        return delta

    # Initialize Thue-Morse
    c = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    current_cost = count_mono(c)
    best_c = c.copy()
    best_cost = current_cost
    
    # Local Search + SA
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    while time.time() < deadline:
        improved = False
        # Pass of local search
        indices = np.arange(n)
        random.shuffle(indices)
        for i in indices:
            d_cost = calc_delta(i, c)
            if d_cost < 0:
                c[i] = 1 - c[i]
                current_cost += d_cost
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_c = c.copy()
            elif d_cost == 0:
                # Random walk on plateaus? Maybe skip to save time or accept randomly.
                # Accept with small prob to explore?
                if random.random() < 0.1:
                    c[i] = 1 - c[i]
                    # cost unchanged
            else:
                # d_cost > 0
                # Simulated Annealing acceptance
                if T > min_T:
                    prob = np.exp(-d_cost / T)
                    if random.random() < prob:
                        c[i] = 1 - c[i]
                        current_cost += d_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_c = c.copy()
        
        if not improved and T <= min_T:
            break
            
        T *= cooling
        
        # If stuck, maybe restart?
        # For now, just keep cooling.
        
    return best_c.tolist()
