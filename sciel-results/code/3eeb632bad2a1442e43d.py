import time
import numpy as np
import random

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d), 1-indexed a,d
# Convert to 0-indexed internally.
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

# Precompute AP indices as numpy array for fast vectorized checks
_AP_ARRAY = np.array(_ALL_APS, dtype=np.int32)  # shape (num_aps, 4)
_NUM_APS = len(_ALL_APS)

# For each position, which APs contain it
_pos_in_aps = [[] for _ in range(128)]
for ap_idx, ap in enumerate(_ALL_APS):
    for i in ap:
        _pos_in_aps[i].append(ap_idx)


def _count_violations(colors_arr):
    """
    Vectorized count of APs that have fewer than 3 distinct colors.
    colors_arr: numpy array of shape (128,) with integer color labels.
    Returns: number of violating APs (0 = valid).
    """
    # Get colors for each AP: shape (num_aps, 4)
    ap_colors = colors_arr[_AP_ARRAY]  # (num_aps, 4)
    
    # For each AP, count distinct colors
    # An AP is valid iff it has >= 3 distinct colors
    # Equivalent: NOT (all same) AND NOT (exactly 2 distinct)
    
    c0 = ap_colors[:, 0]
    c1 = ap_colors[:, 1]
    c2 = ap_colors[:, 2]
    c3 = ap_colors[:, 3]
    
    # All four same
    all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
    
    # Exactly 2 distinct: check all pairs of "which two values"
    # Case 1: c0=c1, c2=c3, c0!=c2
    case1 = (c0 == c1) & (c2 == c3) & (c0 != c2)
    # Case 2: c0=c2, c1=c3, c0!=c1
    case2 = (c0 == c2) & (c1 == c3) & (c0 != c1)
    # Case 3: c0=c3, c1=c2, c0!=c1
    case3 = (c0 == c3) & (c1 == c2) & (c0 != c1)
    # Case 4: c0=c1=c2, c3 different (3 of one, 1 of another)
    case4 = (c0 == c1) & (c1 == c2) & (c0 != c3)
    # Case 5: c0=c1=c3, c2 different
    case5 = (c0 == c1) & (c1 == c3) & (c0 != c2)
    # Case 6: c0=c2=c3, c1 different
    case6 = (c0 == c2) & (c2 == c3) & (c0 != c1)
    # Case 7: c1=c2=c3, c0 different
    case7 = (c1 == c2) & (c2 == c3) & (c1 != c0)
    
    exactly_two = case1 | case2 | case3 | case4 | case5 | case6 | case7
    
    violating = all_same | exactly_two
    return int(np.sum(violating))


def _count_distinct_colors(colors_arr):
    """Count distinct color labels used."""
    return len(set(colors_arr.tolist()))


def _local_search(start_colors, deadline, seed=42):
    """
    Simulated annealing / randomized local search.
    Start with a valid coloring, then try to reduce the number of distinct colors.
    """
    rng = random.Random(seed)
    colors = np.array(start_colors, dtype=np.int32)
    n = 128
    
    # Ensure we start valid
    if _count_violations(colors) != 0:
        return None
    
    best_colors = colors.copy()
    best_num_colors = _count_distinct_colors(colors)
    
    # Strategy: try to merge colors (reduce palette) via hill climbing + restarts
    # At each step, pick a random position and change its color to another used color,
    # or to a new color (to explore). Accept if it improves or maintains validity with fewer colors.
    
    current_violations = 0
    current_colors = _count_distinct_colors(colors)
    
    # Temperature schedule
    t = 1.0
    cooling = 0.995
    min_t = 0.01
    
    max_steps = 500000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        if t < min_t:
            break
        
        # Pick a random position
        pos = rng.randint(0, n - 1)
        old_color = colors[pos]
        
        # Choose a new color: either an existing color or a new one
        used = set(colors.tolist())
        # Try to reduce: pick a color from used (possibly same)
        # To encourage reduction, bias toward colors that are "central"
        if rng.random() < 0.7:
            # Pick from existing colors (bias toward reducing distinct count)
            candidate = rng.choice(list(used))
        else:
            # Try a new color (might help escape local minima)
            candidate = rng.randint(0, 127)
        
        if candidate == old_color:
            continue
        
        colors[pos] = candidate
        new_violations = _count_violations(colors)
        new_colors = _count_distinct_colors(colors)
        
        # Objective: minimize (violations, num_colors) lexicographically
        # We want violations == 0 and minimize num_colors
        
        if new_violations == 0 and new_colors <= current_colors:
            # Improvement or equal
            if new_colors < current_colors:
                best_colors = colors.copy()
                best_num_colors = new_colors
            current_violations = 0
            current_colors = new_colors
            t = max(t * cooling, min_t)
        elif new_violations == 0 and new_colors > current_colors:
            # More colors but still valid: accept with some probability to escape
            if rng.random() < t:
                current_violations = 0
                current_colors = new_colors
                t = max(t * cooling, min_t)
            else:
                colors[pos] = old_color
        elif new_violations > 0:
            # Invalid: accept with small probability to escape
            if rng.random() < t * 0.1:
                current_violations = new_violations
                current_colors = new_colors
                t = max(t * cooling, min_t)
            else:
                colors[pos] = old_color
        else:
            # Shouldn't happen (new_violations < current_violations when current > 0)
            current_violations = new_violations
            current_colors = new_colors
            t = max(t * cooling, min_t)
        
        # Periodic restart if stuck
        if step % 5000 == 0 and t < 0.1:
            # Restart from best
            colors = best_colors.copy()
            current_violations = 0
            current_colors = best_num_colors
            t = 0.5
    
    return best_colors.tolist()


def _greedy_valid_coloring(n=128, max_colors=128):
    """
    Greedy: assign colors left to right, using the smallest color that doesn't create
    an immediately unfixable violation. Returns a valid coloring.
    """
    colors = [-1] * n
    for pos in range(n):
        # Try colors 0, 1, 2, ... up to max_colors
        for c in range(max_colors):
            colors[pos] = c
            # Check all APs containing pos: after assigning pos, can the AP still reach >=3?
            valid = True
            for ap_idx in _pos_in_aps[pos]:
                ap = _ALL_APS[ap_idx]
                present = set()
                unassigned = 0
                for i in ap:
                    if colors[i] == -1:
                        unassigned += 1
                    else:
                        present.add(colors[i])
                # Need len(present) + unassigned >= 3
                if len(present) + unassigned < 3:
                    valid = False
                    break
            if valid:
                break
        else:
            # Shouldn't happen if max_colors is large enough
            colors[pos] = 0
    
    # Verify
    assert _count_violations(np.array(colors, dtype=np.int32)) == 0, "Greedy failed"
    return colors


def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    
    n = 128
    
    # Step 1: Get a valid starting coloring using greedy
    initial = _greedy_valid_coloring(n, 128)
    best_colors = initial
    best_ncolors = _count_distinct_colors(np.array(initial, dtype=np.int32))
    
    # Step 2: Local search to minimize number of distinct colors
    # Multiple restarts with different seeds
    remaining = DEADLINE - (time.time() - start)
    num_restarts = max(1, int(remaining / 5.0))  # ~5s per restart
    
    for restart in range(num_restarts):
        if time.time() > DEADLINE:
            break
        
        # For each restart, start from a perturbed version of the best
        if restart == 0:
            start_colors = best_colors
        else:
            # Perturb: randomly reassign some positions
            perturbed = best_colors[:]
            num_perturb = rng.randint(5, 20) if 'rng' in dir() else random.randint(5, 20)
            for _ in range(num_perturb):
                pos = random.randint(0, n - 1)
                perturbed[pos] = random.randint(0, best_ncolors - 1)
            # Ensure validity; if not, revert
            if _count_violations(np.array(perturbed, dtype=np.int32)) != 0:
                perturbed = best_colors[:]
            start_colors = perturbed
        
        result = _local_search(start_colors, DEADLINE - 1.0, seed=restart * 7 + 13)
        if result is not None:
            num_c = _count_distinct_colors(np.array(result, dtype=np.int32))
            if num_c < best_ncolors:
                best_colors = result
                best_ncolors = num_c
        
        # Early stop if we hit 3 colors (minimum possible)
        if best_ncolors == 3:
            break
    
    return best_colors
