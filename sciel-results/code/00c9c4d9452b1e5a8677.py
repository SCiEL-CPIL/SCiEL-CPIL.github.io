import numpy as np
import time
import math
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precompute all 4-term APs
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # 2. Build index to APs mapping
    # aps_by_index[i] = list of indices in 'aps' that contain i
    aps_by_index = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for i in ap:
            aps_by_index[i].append(idx)

    # Helper to count monochromatic APs
    def count_mono(colors):
        count = 0
        for ap in aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count

    # Helper to calculate delta for flipping bit i
    def calc_delta(i, colors):
        curr_color = colors[i]
        new_color = 1 - curr_color
        delta = 0
        for ap_idx in aps_by_index[i]:
            ap = aps[ap_idx]
            # Check old state
            old_mono = (colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]])
            # Check new state
            # We know colors[i] becomes new_color. 
            # We need to check if all other 3 are new_color AND the flipped one is new_color.
            # Actually, just check if all 4 are new_color in the hypothetical state.
            # Since only i changes, we can check:
            # Are all other 3 elements equal to new_color?
            # If yes, and new_color is the color of i (which it is by definition in new state), then it's mono.
            # Wait, if the other 3 are new_color, then the new AP is mono.
            # If the other 3 are curr_color, the old AP was mono.
            
            # Optimization: 
            # old_mono is True iff all 4 are curr_color.
            # new_mono is True iff all 4 are new_color.
            # Since i is curr_color in old and new_color in new:
            # old_mono == (colors[other1] == curr_color and colors[other2] == curr_color and colors[other3] == curr_color)
            # new_mono == (colors[other1] == new_color and colors[other2] == new_color and colors[other3] == new_color)
            
            # Get the other 3 indices
            others = [j for j in ap if j != i]
            
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            
            if c1 == curr_color and c2 == curr_color and c3 == curr_color:
                delta -= 1 # Old was mono, now it's not (unless new_color == curr_color which is false)
                # Actually, if old was mono, new is NOT mono because i flipped.
                # So contribution is 0 - 1 = -1.
                
            if c1 == new_color and c2 == new_color and c3 == new_color:
                delta += 1 # New is mono, old was not (unless curr_color == new_color which is false)
                # If new is mono, old was NOT mono because i was curr_color != new_color.
                # So contribution is 1 - 0 = +1.
                
            # Note: It's impossible for both to be true simultaneously.
        return delta

    best_score = float('inf')
    best_colors = None

    # Multiple restarts
    num_restarts = 50
    steps_per_restart = 500000 # Adjust based on speed
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize colors
        # Mix of random and Thue-Morse
        if r % 3 == 0:
            # Random
            colors = np.random.randint(0, 2, N)
        elif r % 3 == 1:
            # Thue-Morse: popcount(i) % 2
            colors = np.array([(bin(i).count('1') % 2) for i in range(N)])
        else:
            # Random blocks of size 4
            colors = np.array([(i // 4) % 2 for i in range(N)])
            
        # Ensure colors are int8 for speed
        colors = colors.astype(np.int8)
        
        current_energy = count_mono(colors)
        if current_energy < best_score:
            best_score = current_energy
            best_colors = colors.copy()
            
        # Simulated Annealing
        T = 2.0 # Initial temperature
        cooling = 0.9995
        
        # We will run a fixed number of steps or until time runs out
        # Estimate time per step
        # Let's just run steps until deadline is approached
        
        while time.time() < deadline - 2.0:
            i = np.random.randint(0, N)
            delta = calc_delta(i, colors)
            
            if delta < 0 or (delta > 0 and np.random.rand() < math.exp(-delta / T)):
                colors[i] = 1 - colors[i]
                current_energy += delta
                if current_energy < best_score:
                    best_score = current_energy
                    best_colors = colors.copy()
            
            T *= cooling
            if T < 0.01:
                # Restart temperature or break? 
                # Let's just keep going with low T (greedy) for a bit
                pass

        # Local Search Polish (Greedy)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Shuffle order to avoid bias? Or just random order
            indices = np.random.permutation(N)
            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                delta = calc_delta(i, colors)
                if delta < 0:
                    colors[i] = 1 - colors[i]
                    current_energy += delta
                    improved = True
                    if current_energy < best_score:
                        best_score = current_energy
                        best_colors = colors.copy()

    # Final conversion to list of ints
    if best_colors is None:
        # Fallback to seed
        return [(i // 2) % 2 for i in range(N)]
        
    return [int(c) for c in best_colors]
