"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
import numpy as np

def _precompute_progressions(n=128):
    """Precompute all nonconstant 4-term progressions."""
    progs = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            e = a + 3 * d
            if e < n:
                progs.append((a, a + d, a + 2 * d, e))
    return progs

PROGS = _precompute_progressions(128)

def _count_conflicts(colors):
    """Count the number of 4-term progressions with < 3 distinct colors."""
    conflicts = 0
    for a, b, c, d in PROGS:
        if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
            conflicts += 1
    return conflicts

def _local_search(colors, max_iters=50000, deadline=None):
    """
    Local search to minimize conflicts. 
    Returns the best coloring found.
    """
    colors = np.array(colors)
    n = len(colors)
    current_conflicts = _count_conflicts(colors)
    
    if current_conflicts == 0:
        return colors
    
    best_colors = colors.copy()
    best_conflicts = current_conflicts
    temp = 1.0
    cooling = 0.999
    
    for i in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Pick a random position
        pos = random.randint(0, n - 1)
        old_color = colors[pos]
        
        # Try a new random color
        # We want to minimize the number of colors used eventually, 
        # but first we need a valid solution.
        # Let's restrict colors to 0..3 initially to keep the space small, 
        # then try to reduce.
        new_color = random.randint(0, 3)
        
        if new_color == old_color:
            continue
            
        colors[pos] = new_color
        new_conflicts = _count_conflicts(colors)
        
        delta = new_conflicts - current_conflicts
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_conflicts = new_conflicts
            if new_conflicts < best_conflicts:
                best_conflicts = new_conflicts
                best_colors = colors.copy()
        else:
            colors[pos] = old_color
            
        temp *= cooling
        
        if current_conflicts == 0:
            break
            
    return best_colors

def _reduce_colors(colors, deadline):
    """
    Try to reduce the number of distinct colors in a valid coloring.
    """
    colors = np.array(colors)
    unique_colors = sorted(set(colors))
    
    # Map to 0..k-1
    color_map = {c: i for i, c in enumerate(unique_colors)}
    colors = np.array([color_map[c] for c in colors])
    k = len(unique_colors)
    
    # Try to merge colors
    while k > 2: # At least 2 colors needed? No, 1 color is invalid for N>4.
        merged = False
        for i in range(k):
            for j in range(i + 1, k):
                # Try merging j into i
                trial = colors.copy()
                trial[trial == j] = i
                
                # Check validity
                if _count_conflicts(trial) == 0:
                    colors = trial
                    k -= 1
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
            
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_result = None
    best_score = 128 # Max colors
    
    # Try different numbers of colors
    # Start with small K and increase if we can't find a valid solution quickly
    # However, local search with fixed K might get stuck.
    # Let's try K=3 first, then K=4, etc.
    
    for K in [3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
            
        # Multiple restarts for each K
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Initialize with K random colors
            colors = np.random.randint(0, K, 128)
            
            # Local search to minimize conflicts
            # We use a simple SA approach
            # To make it faster, we can use a greedy hill climbing first
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                current_conflicts = _count_conflicts(colors)
                if current_conflicts == 0:
                    break
                    
                # Find a position to change
                # We can sample random positions
                for _ in range(100):
                    pos = random.randint(0, 127)
                    old_c = colors[pos]
                    
                    # Try all K colors
                    best_new_c = old_c
                    min_conflicts = current_conflicts
                    
                    for c in range(K):
                        if c == old_c:
                            continue
                        colors[pos] = c
                        new_conf = _count_conflicts(colors)
                        if new_conf < min_conflicts:
                            min_conflicts = new_conf
                            best_new_c = c
                    colors[pos] = best_new_c
                    
                    if best_new_c != old_c and min_conflicts < current_conflicts:
                        improved = True
                        break
                        # If we improved, restart the outer while loop to re-evaluate
                    else:
                        # If no improvement found in this batch of 100, break to try SA or next restart
                        # Actually, just break the inner for loop
                        pass
                
                if not improved:
                     # Stuck in local minimum, use SA step or break
                     break
            
            # After hill climbing, if still conflicts, try SA
            if _count_conflicts(colors) > 0 and time.time() < deadline - 5:
                colors = _local_search(colors, max_iters=2000, deadline=deadline - 5)
                
            if _count_conflicts(colors) == 0:
                # Valid solution found with K colors
                # Try to reduce colors
                reduced = _reduce_colors(colors, deadline)
                num_colors = len(set(reduced))
                
                if num_colors < best_score:
                    best_score = num_colors
                    best_result = reduced
                    
                if best_score == 3: # Can't go lower than 3? 
                    # Actually, can we go lower?
                    # 1 color: all 4-APs are monochromatic (1 color) -> Invalid.
                    # 2 colors: Can we avoid 4-APs with <= 2 colors?
                    # If we use 2 colors, any 4-AP will have colors from {0,1}.
                    # The set of colors in the AP will be a subset of {0,1}.
                    # So the number of distinct colors will be <= 2.
                    # Thus, any 4-AP will violate the condition "at least 3 distinct colors".
                    # So 2 colors is impossible for any N >= 4.
                    # So min colors is 3.
                    return list(best_result)
    
    if best_result is not None:
        return list(best_result)
        
    # Fallback: return a random valid-looking thing or just a range
    # If no valid solution found, we must return something.
    # The task says "Minimize the number of labels actually used".
    # If we can't find a valid one, we should probably return the best effort.
    # But the evaluator checks validity. If invalid, score 0.
    # So we must ensure we return a valid one if possible.
    # The hill climbing above is quite aggressive.
    
    # Emergency fallback: Greedy
    colors = [0] * 128
    for x in range(128):
        found = False
        for c in range(3):
            colors[x] = c
            valid = True
            for a, b, d, e in PROGS:
                if e == x or (a==x) or (b==x) or (d==x):
                    if len({colors[a], colors[b], colors[d], colors[e]}) < 3:
                        valid = False
                        break
            if valid:
                found = True
                break
        if not found:
            # If 3 colors fail, try 4
            for c in range(4):
                colors[x] = c
                valid = True
                for a, b, d, e in PROGS:
                    if e == x or (a==x) or (b==x) or (d==x):
                        if len({colors[a], colors[b], colors[d], colors[e]}) < 3:
                            valid = False
                            break
                if valid:
                    found = True
                    break
    return colors
# EVOLVE-BLOCK-END
