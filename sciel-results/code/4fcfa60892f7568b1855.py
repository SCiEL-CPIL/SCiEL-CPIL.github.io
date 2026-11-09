import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Precompute all 4-APs: (a, a+d, a+2d, a+3d) with 0-indexed positions
    # Position i corresponds to integer i+1
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2*d, a + 3*d))
    
    num_aps = len(aps)
    
    # For each position, list of AP indices that contain it
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Count mono APs
    def count_mono(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                count += 1
        return count
    
    # Thue-Morse initialization
    def thue_morse(n):
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = bin(i).count('1') % 2
        return colors
    
    # Try multiple initializations
    best_cost = float('inf')
    best_colors = None
    
    initializations = []
    # Thue-Morse
    initializations.append(thue_morse(n))
    # Random
    for _ in range(3):
        initializations.append(np.random.randint(0, 2, size=n, dtype=np.int8))
    # Alternating
    initializations.append(np.array([(i % 2) for i in range(n)], dtype=np.int8))
    # Blocks of 4
    initializations.append(np.array([(i // 4) % 2 for i in range(n)], dtype=np.int8))
    
    for init_colors in initializations:
        if time.time() > deadline - 10:
            break
        
        colors = init_colors.copy()
        current_cost = count_mono(colors)
        
        # Hill climbing with simulated annealing
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        
        for step in range(50000):
            if time.time() > deadline - 5:
                break
            
            # Pick a random position to flip
            pos = random.randint(0, n - 1)
            
            # Compute delta: change in mono count if we flip pos
            old_color = colors[pos]
            new_color = 1 - old_color
            
            delta = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                # Get colors of the 4 positions
                cs = [colors[p] for p in ap]
                # Count how many are same as old_color
                same_old = sum(1 for c in cs if c == old_color)
                same_new = sum(1 for c in cs if c == new_color)
                
                # Before flip: mono if all 4 same
                mono_before = (same_old == 4)
                # After flip: the flipped position changes
                # If old_color was the common color, after flip it's not mono
                # If new_color was the common color (i.e., 3 were new_color, 1 was old_color), after flip it becomes mono
                mono_after = (same_new == 4)
                
                if mono_before and not mono_after:
                    delta -= 1
                elif mono_after and not mono_before:
                    delta += 1
            
            # Accept based on simulated annealing
            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                colors[pos] = new_color
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
            
            T *= cooling
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors.copy()
    
    # Final polish: pure hill climbing on best solution
    if best_colors is not None and time.time() < deadline - 5:
        colors = best_colors.copy()
        current_cost = count_mono(colors)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for pos in range(n):
                if time.time() > deadline - 3:
                    break
                old_color = colors[pos]
                new_color = 1 - old_color
                
                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    cs = [colors[p] for p in ap]
                    same_old = sum(1 for c in cs if c == old_color)
                    same_new = sum(1 for c in cs if c == new_color)
                    mono_before = (same_old == 4)
                    mono_after = (same_new == 4)
                    if mono_before and not mono_after:
                        delta -= 1
                    elif mono_after and not mono_before:
                        delta += 1
                
                if delta < 0:
                    colors[pos] = new_color
                    current_cost += delta
                    improved = True
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
    
    if best_colors is None:
        best_colors = thue_morse(n)
    
    return [int(c) for c in best_colors]

