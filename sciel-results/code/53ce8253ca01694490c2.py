import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _get_4taps(n):
    """Generate all 4-term arithmetic progressions as an array of indices."""
    taps = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            taps.append([a, a + d, a + 2 * d, a + 3 * d])
    return np.array(taps, dtype=np.int32)

def _count_violations(colors, taps):
    """Count number of 4-term progressions with < 3 distinct colors."""
    if len(taps) == 0:
        return 0
    tap_colors = colors[taps] # shape (M, 4)
    
    # Sort each row to check for distinct counts
    # Size 1: all equal
    # Size 2: patterns x,x,x,y or x,x,y,y
    sorted_taps = np.sort(tap_colors, axis=1)
    
    size_1 = sorted_taps[:, 0] == sorted_taps[:, 3]
    
    size_2_1 = (sorted_taps[:, 0] == sorted_taps[:, 2]) & (sorted_taps[:, 2] != sorted_taps[:, 3])
    size_2_2 = (sorted_taps[:, 0] == sorted_taps[:, 1]) & (sorted_taps[:, 2] == sorted_taps[:, 3]) & (sorted_taps[:, 1] != sorted_taps[:, 2])
    
    size_2 = size_2_1 | size_2_2
    
    invalid = size_1 | size_2
    return np.sum(invalid)

def _count_distinct(colors):
    """Count the number of distinct colors used."""
    return len(np.unique(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    n = 128
    rng = random.Random(42)
    
    taps = _get_4taps(n)
    
    # 1. Initialize with a valid solution: Random permutation of 0..127
    # This ensures 0 violations and 128 distinct colors.
    colors = np.arange(n, dtype=np.int32)
    rng.shuffle(colors)
    
    current_distinct = _count_distinct(colors)
    current_viol = 0
    
    best_colors = colors.copy()
    best_distinct = current_distinct
    
    # 2. Local Search to minimize distinct colors while maintaining validity
    # We only accept moves that keep violations at 0.
    # This guarantees the result is always valid.
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() < deadline:
        if current_viol != 0:
            # Should not happen if we start valid and only accept valid moves,
            # but just in case, revert to best
            colors = best_colors.copy()
            current_distinct = best_distinct
            current_viol = 0
            
        # Pick a random position
        pos = rng.randint(0, n - 1)
        old_color = colors[pos]
        
        # Pick a random new color
        # To encourage merging, we could pick a color that is already used.
        # But random 0..127 is fine.
        new_color = rng.randint(0, 127)
        
        if new_color == old_color:
            continue
            
        colors[pos] = new_color
        
        new_viol = _count_violations(colors, taps)
        
        if new_viol == 0:
            new_distinct = _count_distinct(colors)
            
            # Accept if it reduces distinct colors
            # Or if it keeps them same, accept with some probability to allow rearrangement
            if new_distinct < current_distinct:
                current_distinct = new_distinct
                current_viol = new_viol
                if new_distinct < best_distinct:
                    best_distinct = new_distinct
                    best_colors = colors.copy()
            elif new_distinct == current_distinct:
                # Accept to explore state space
                # Use a simple probability or just accept
                if rng.random() < 0.5: 
                    current_viol = new_viol
                    # Distinct count same, so best doesn't change
            else:
                # Reject if distinct count increases (we are minimizing)
                colors[pos] = old_color
        else:
            # Reject if violations introduced
            colors[pos] = old_color
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Return the best valid solution found
    # Ensure the returned list is of integers
    return best_colors.tolist()

# EVOLVE-BLOCK-END
