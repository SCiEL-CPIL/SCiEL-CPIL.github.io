import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 105  # Leave 15s buffer for 120s limit
    
    # Precompute all 4-term arithmetic progressions
    # a, a+d, a+2d, a+3d where 1 <= a, a+3d <= n
    # Using 0-indexed: positions 0..n-1
    # a (0-indexed) from 0 to n-1-3d
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Precompute for each position, which APs it belongs to
    # And for each AP, the other 3 positions
    pos_aps = [[] for _ in range(n)]
    ap_others = []
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_aps[pos].append(idx)
        # For each position in the AP, store the other 3 positions
        for i, pos in enumerate(ap):
            others = [ap[j] for j in range(4) if j != i]
            ap_others.append((idx, pos, others))
    
    # Helper: count monochromatic APs
    def count_mono(colors):
        count = 0
        for ap in ap_list:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count
    
    # Helper: compute delta if we flip position pos
    def compute_delta(colors, pos):
        """Return change in mono count if we flip colors[pos]"""
        delta = 0
        for ap_idx in pos_aps[pos]:
            ap = ap_list[ap_idx]
            # Count how many of the other 3 have same color as pos
            same_count = 0
            for other in ap:
                if other != pos:
                    if colors[other] == colors[pos]:
                        same_count += 1
            
            if colors[pos] == 0:
                # Currently all same means mono. If we flip, we break mono if all 4 were same.
                # Before flip: mono if all 4 same. After flip: mono if other 3 same and new color matches them.
                # Actually, let's think differently.
                # Before: the AP is mono if all 4 have color 0 (since pos has color 0).
                # After flip: pos has color 1. AP is mono if all 4 have color 1.
                # So we need to check if the other 3 all have color 0 (before mono) and if they all have color 1 (after mono).
                others_colors = [colors[other] for other in ap if other != pos]
                before_mono = all(c == 0 for c in others_colors)  # since pos is 0
                after_mono = all(c == 1 for c in others_colors)   # since pos will be 1
                if before_mono and not after_mono:
                    delta -= 1
                elif after_mono and not before_mono:
                    delta += 1
            else:
                others_colors = [colors[other] for other in ap if other != pos]
                before_mono = all(c == 1 for c in others_colors)  # since pos is 1
                after_mono = all(c == 0 for c in others_colors)   # since pos will be 0
                if before_mono and not after_mono:
                    delta -= 1
                elif after_mono and not before_mono:
                    delta += 1
        return delta
    
    # Thue-Morse initial coloring (0-indexed)
    def thue_morse(i):
        # Number of 1-bits in binary representation of i, mod 2
        return bin(i).count('1') % 2
    
    # Local search
    def local_search(colors, max_sweeps=100):
        colors = list(colors)
        current_count = count_mono(colors)
        
        for sweep in range(max_sweeps):
            if time.time() > deadline:
                break
            improved = False
            # Random order for diversity
            order = list(range(n))
            random.shuffle(order)
            for pos in order:
                delta = compute_delta(colors, pos)
                if delta < 0:
                    colors[pos] = 1 - colors[pos]
                    current_count += delta
                    improved = True
            if not improved:
                break
        return colors, current_count
    
    # Start with Thue-Morse
    best_colors = [thue_morse(i) for i in range(n)]
    best_count = count_mono(best_colors)
    
    # Run local search
    best_colors, best_count = local_search(best_colors)
    
    # Try multiple random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initial coloring
        colors = [random.randint(0, 1) for _ in range(n)]
        
        # Run local search
        colors, count = local_search(colors, max_sweeps=50)
        
        if count < best_count:
            best_count = count
            best_colors = colors
    
    # One more local search on best
    best_colors, best_count = local_search(best_colors, max_sweeps=200)
    
    return best_colors
