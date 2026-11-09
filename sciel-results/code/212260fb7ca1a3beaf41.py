"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def _thue_morse(n):
    """Return the Thue-Morse sequence value at index n."""
    return bin(n).count('1') % 2

def _count_mono_4aps(colors):
    """Count monochromatic 4-term APs in colors (0-indexed, for positions 0..255)."""
    n = len(colors)
    count = 0
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            c = colors[a]
            if c == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def _build_ap_index(colors):
    """Build index: for each position, list of (ap_tuple, position_in_ap) for APs containing it."""
    n = len(colors)
    pos_to_aps = [[] for _ in range(n)]
    all_aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_aps)
            all_aps.append(ap)
            for j, pos in enumerate(ap):
                pos_to_aps[pos].append((idx, j))
    return all_aps, pos_to_aps

def construct():
    n = 256
    # Initialize with Thue-Morse
    colors = [_thue_morse(i) for i in range(n)]
    
    # Precompute all 4-APs and index
    all_aps, pos_to_aps = _build_ap_index(colors)
    
    # Current count
    current_count = _count_mono_4aps(colors)
    
    # Local search: try flipping each position if it reduces count
    improved = True
    max_iters = 50
    iteration = 0
    while improved and iteration < max_iters:
        improved = False
        iteration += 1
        for i in range(n):
            # Try flipping position i
            old_color = colors[i]
            new_color = 1 - old_color
            
            # Count change: for each AP containing i, check mono status before and after
            delta = 0
            for (ap_idx, j) in pos_to_aps[i]:
                ap = all_aps[ap_idx]
                # Before flip: is it monochromatic?
                before_mono = all(colors[ap[k]] == old_color for k in range(4) if k != j) and colors[ap[j]] == old_color
                # Actually, simpler: check all 4 positions
                before_mono = (colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]])
                
                # After flip: position i has new_color
                after_colors = [colors[ap[k]] if ap[k] != i else new_color for k in range(4)]
                after_mono = (after_colors[0] == after_colors[1] == after_colors[2] == after_colors[3])
                
                if before_mono and not after_mono:
                    delta -= 1
                elif not before_mono and after_mono:
                    delta += 1
            
            if delta < 0:
                colors[i] = new_color
                current_count += delta
                improved = True
    
    return colors

# EVOLVE-BLOCK-END
