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
    # colors: np array of shape (n,)
    # taps: np array of shape (M, 4)
    tap_colors = colors[taps] # shape (M, 4)
    
    # For each row, we need to check if the number of unique values is < 3.
    # A row has < 3 unique values if:
    # 1. All 4 are same (1 unique)
    # 2. Exactly 2 unique values (e.g., x,x,x,y or x,x,y,y etc.)
    
    # Efficient check:
    # Sort the 4 values in each row? No, sorting M rows of 4 is okay but maybe overkill.
    # Alternative:
    # Unique count < 3 iff:
    # (all equal) OR (two pairs) OR (three same one diff)
    
    # Let's use a trick:
    # If we subtract min from all, and max... no.
    
    # Let's just compute unique counts.
    # np.unique on axis=1 is not directly supported efficiently for counting per row in older numpy?
    # Actually, we can check:
    # viol = 0
    # For each tap, if len(set) < 3.
    
    # Vectorized approach:
    # A tap is valid if it has at least 3 distinct colors.
    # Invalid if it has 1 or 2 distinct colors.
    
    # Let's check 1 distinct: all 4 equal.
    all_equal = np.all(tap_colors == tap_colors[:, 0:1], axis=1)
    
    # Let's check 2 distinct:
    # This is trickier.
    # A set has size 2 if there are exactly 2 distinct values.
    # We can check if there exist i, j such that c_i != c_j, and for all k, c_k is in {c_i, c_j}.
    
    # Simpler: 
    # If we sort each row, size 1 means [x,x,x,x], size 2 means [x,x,x,y] or [x,x,y,y].
    # Sorting M=2600 rows of 4 elements is very fast.
    sorted_taps = np.sort(tap_colors, axis=1)
    
    # Size 1: sorted[:, 0] == sorted[:, 3]
    size_1 = sorted_taps[:, 0] == sorted_taps[:, 3]
    
    # Size 2: 
    # Pattern x,x,x,y: sorted[:, 0] == sorted[:, 2] and sorted[:, 2] != sorted[:, 3]
    # Pattern x,x,y,y: sorted[:, 0] == sorted[:, 1] and sorted[:, 2] == sorted[:, 3] and sorted[:, 1] != sorted[:, 2]
    
    size_2_1 = (sorted_taps[:, 0] == sorted_taps[:, 2]) & (sorted_taps[:, 2] != sorted_taps[:, 3])
    size_2_2 = (sorted_taps[:, 0] == sorted_taps[:, 1]) & (sorted_taps[:, 2] == sorted_taps[:, 3]) & (sorted_taps[:, 1] != sorted_taps[:, 2])
    
    size_2 = size_2_1 | size_2_2
    
    invalid = size_1 | size_2
    return np.sum(invalid)

def _local_search(n, k, taps, time_limit=10.0, rng=None):
    """
    Local search to find a valid k-coloring.
    Returns (colors, violations) or (None, None).
    """
    if rng is None:
        rng = random.Random(42)
        
    start_time = time.time()
    colors = np.array([rng.randint(0, k - 1) for _ in range(n)], dtype=np.int32)
    
    current_viol = _count_violations(colors, taps)
    best_colors = colors.copy()
    best_viol = current_viol
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        if current_viol == 0:
            return colors, 0
            
        # Pick random position
        pos = rng.randint(0, n - 1)
        old_color = colors[pos]
        
        # Try a different color
        if k == 1:
            new_color = old_color
        else:
            new_color = rng.randint(0, k - 1)
            while new_color == old_color:
                new_color = rng.randint(0, k - 1)
                
        colors[pos] = new_color
        new_viol = _count_violations(colors, taps)
        
        delta = new_viol - current_viol
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_viol = new_viol
            if new_viol < best_viol:
                best_viol = new_viol
                best_colors = colors.copy()
        else:
            colors[pos] = old_color
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    if best_viol == 0:
        return best_colors, 0
    return None, best_viol

def _greedy_fallback(n, taps):
    """
    Greedy construction that guarantees validity by using enough colors.
    Assigns colors left to right.
    """
    colors = np.zeros(n, dtype=np.int32)
    next_color = 1
    
    # Precompute for each position i, the list of (a, d) such that a+3d = i
    # and the indices of the other 3 terms.
    ending_taps = [[] for _ in range(n)]
    for a, b, c, d in taps:
        ending_taps[d].append((a, b, c))
        
    for i in range(n):
        forbidden = set()
        impossible = False
        for a, b, c in ending_taps[i]:
            c_a = colors[a]
            c_b = colors[b]
            c_c = colors[c]
            other_set = {c_a, c_b, c_c}
            
            if len(other_set) == 1:
                # Three same colors. Cannot fix with 1 new color.
                # We need to use a color different from the existing one?
                # No, if other 3 are x, x, x, then adding y gives {x, y} size 2. Invalid.
                # Adding x gives {x} size 1. Invalid.
                # So we MUST have violated the constraint in previous steps.
                # This greedy approach is flawed for small k.
                # However, if we allow using MANY colors, we still can't fix {x,x,x}.
                # So we need a different fallback.
                pass
            
            if len(other_set) == 2:
                # Other 3 have 2 colors, say x, y.
                # We need size >= 3.
                # So new color must NOT be x or y.
                forbidden.update(other_set)
            
            # If len(other_set) == 3, any color works.
            
        # If we encounter a case where other_set size 1, we are stuck with this greedy.
        # To make a robust fallback, we can just use distinct colors for all?
        # That uses 128 colors.
        # Let's just use a simple valid pattern if greedy fails.
        
        # For now, let's just pick the smallest color not in forbidden.
        # If forbidden contains all available colors (if we limit k), we fail.
        # But in fallback, we can use up to 128 colors.
        
        chosen = 0
        while chosen in forbidden:
            chosen += 1
            
        colors[i] = chosen
        if chosen + 1 > next_color:
            next_color = chosen + 1
            
    # Verify
    if _count_violations(colors, taps) == 0:
        return colors
    else:
        # If greedy failed (due to the size 1 case), use a known valid coloring.
        # e.g. color i with i % 100? No.
        # Let's use the seed's logic if available, or just return a large color count solution.
        # A trivial valid solution: assign color i to position i. (128 colors)
        return np.arange(n, dtype=np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    n = 128
    rng = random.Random(42)
    
    taps = _get_4taps(n)
    
    # Try to find minimal k
    for k in range(3, 11):
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            break
            
        # Allocate time proportional to k? Or fixed?
        # Give more time to smaller k as they are harder?
        # Let's give 5 seconds for k=3, 3 for k=4, etc.
        t_limit = min(5.0, remaining_time * 0.8)
        
        result, viol = _local_search(n, k, taps, time_limit=t_limit, rng=rng)
        
        if result is not None and viol == 0:
            return result.tolist()
            
    # Fallback
    result = _greedy_fallback(n, taps)
    return result.tolist()

# EVOLVE-BLOCK-END
