import time
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 140.0  # Safety margin before 150s
    
    N = 256
    
    # Helper to count monochromatic 4-APs
    def count_mono_4aps(colors):
        count = 0
        for d in range(1, (N - 3) // 3 + 1):
            for a in range(N - 3 * d):
                if colors[a] == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                    count += 1
        return count

    # Precompute all 4-APs as tuples of indices
    # ap_list[i] is a list of indices of APs that contain position i
    # ap_indices[ap_idx] is the tuple of 4 indices
    ap_indices = []
    ap_list = [[] for _ in range(N)]
    ap_id = 0
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            ap_indices.append(ap)
            for idx in ap:
                ap_list[idx].append(ap_id)
            ap_id += 1
    
    total_aps = len(ap_indices)
    
    # Initialize with Thue-Morse sequence
    def thue_morse(n):
        return bin(n).count('1') % 2
        
    colors = [thue_morse(i) for i in range(N)]
    
    # Initial count
    current_count = count_mono_4aps(colors)
    best_count = current_count
    best_colors = colors[:]
    
    # Precompute the state of each AP: is it mono?
    # ap_mono[ap_id] is 1 if mono, 0 otherwise
    ap_mono = np.zeros(total_aps, dtype=int)
    for i in range(total_aps):
        ap = ap_indices[i]
        if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
            ap_mono[i] = 1
    
    # Local search
    # We will try flipping each bit and see if it improves the count.
    # To be efficient, we can calculate the delta for each flip.
    # Delta for flipping bit i:
    # For each AP containing i, let the other 3 bits be b1, b2, b3.
    # If the AP was mono (all 4 same), flipping i makes it non-mono (delta -1).
    # If the AP was non-mono, flipping i might make it mono (delta +1) if the other 3 are same as new color of i.
    
    # Let's implement a simple greedy step:
    # For each position i, calculate potential gain.
    # Flip the one with max gain.
    # Repeat until no gain or time limit.
    
    # To speed up, we can maintain the ap_mono array and update it incrementally.
    
    improved = True
    while improved and (time.time() - start_time) < TIME_LIMIT:
        improved = False
        best_gain = 0
        best_flip_idx = -1
        
        for i in range(N):
            # Calculate delta if we flip i
            delta = 0
            for ap_idx in ap_list[i]:
                ap = ap_indices[ap_idx]
                # Get the other 3 indices
                others = [j for j in ap if j != i]
                # Check if the other 3 are monochromatic
                if colors[others[0]] == colors[others[1]] == colors[others[2]]:
                    # If they are all the same as current color of i, the AP is currently mono.
                    # Flipping i will make it non-mono. Delta = -1.
                    if colors[i] == colors[others[0]]:
                        delta -= 1
                    # If they are all the same as new color of i (1-colors[i]), the AP will become mono.
                    # Flipping i will make it mono. Delta = +1.
                    elif (1 - colors[i]) == colors[others[0]]:
                        delta += 1
            # If delta is negative, it's an improvement.
            if delta < best_gain:
                best_gain = delta
                best_flip_idx = i
        
        if best_flip_idx != -1:
            # Perform the flip
            i = best_flip_idx
            colors[i] = 1 - colors[i]
            current_count += best_gain
            improved = True
            
            # Update ap_mono for APs containing i
            for ap_idx in ap_list[i]:
                ap = ap_indices[ap_idx]
                if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                    ap_mono[ap_idx] = 1
                else:
                    ap_mono[ap_idx] = 0
            
            if current_count < best_count:
                best_count = current_count
                best_colors = colors[:]
    
    return best_colors
