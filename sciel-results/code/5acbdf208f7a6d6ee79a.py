import time
import random

def _seed_fallback():
    """Deterministic seed algorithm - guaranteed valid."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

def _validate(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _backtrack_3color(n=128, seed=0, time_limit=5.0):
    """Backtracking search for a 3-coloring."""
    rng = random.Random(seed)
    start_time = time.time()
    
    colors = [-1] * n  # -1 means unassigned
    
    # Precompute for each position i, which (a, d) pairs form a 4-AP (a, a+d, a+2d, a+3d) where i is one of the four
    # For forward checking, when we assign position i, we need to check all 4-APs containing i
    # and see if the constraint is satisfied or if it eliminates colors for unassigned positions
    
    # For efficiency, precompute APs
    # APs where position i is the last element (a+3d = i): a = i-3d, d>=1, a>=0
    # APs where position i is the third element (a+2d = i): a = i-2d, d>=1, a>=0, and i+d < n
    # APs where position i is the second element (a+d = i): a = i-d, d>=1, a>=0, and i+2d < n
    # APs where position i is the first element (a = i): d>=1, i+3d < n
    
    def get_aps_containing(i):
        """Yield (a, b, c, e, pos_of_i) for all 4-APs containing position i."""
        ap_list = []
        # i is a+3d: a = i-3d, b = i-2d, c = i-d, e = i
        for d in range(1, (i // 3) + 1):
            a = i - 3 * d
            if a >= 0:
                ap_list.append((a, i - 2 * d, i - d, i, 3))
        # i is a+2d: a = i-2d, b = i-d, c = i, e = i+d
        for d in range(1, ((n - 1 - i) // 1) + 1):
            a = i - 2 * d
            e = i + d
            if a >= 0 and e < n:
                ap_list.append((a, i - d, i, e, 2))
        # i is a+d: a = i-d, b = i, c = i+d, e = i+2d
        for d in range(1, ((n - 1 - i) // 2) + 1):
            a = i - d
            c = i + d
            e = i + 2 * d
            if a >= 0 and e < n:
                ap_list.append((a, i, c, e, 1))
        # i is a: a = i, b = i+d, c = i+2d, e = i+3d
        for d in range(1, ((n - 1 - i) // 3) + 1):
            b = i + d
            c = i + 2 * d
            e = i + 3 * d
            if e < n:
                ap_list.append((i, b, c, e, 0))
        return ap_list
    
    # Cache APs
    ap_cache = [get_aps_containing(i) for i in range(n)]
    
    def is_valid_assignment(i, color):
        """Check if assigning color to position i is consistent with all 4-APs containing i 
        where all other positions are already assigned."""
        for a, b, c, e, pos_idx in ap_cache[i]:
            positions = [a, b, c, e]
            # Check if all other positions are assigned
            other_assigned = True
            for j, p in enumerate(positions):
                if p == i:
                    continue
                if colors[p] == -1:
                    other_assigned = False
                    break
            if other_assigned:
                # All other positions assigned, check constraint
                cols = [color]
                for j, p in enumerate(positions):
                    if p != i:
                        cols.append(colors[p])
                if len(set(cols)) < 3:
                    return False
        return True
    
    def forward_check(i, color):
        """After assigning color to position i, check if any unassigned position 
        in a 4-AP with i now has no valid colors. Returns True if consistent."""
        for a, b, c, e, pos_idx in ap_cache[i]:
            positions = [a, b, c, e]
            unassigned = [p for p in positions if p != i and colors[p] == -1]
            if len(unassigned) == 0:
                # All assigned, verify
                cols = [color] + [colors[p] for p in positions if p != i]
                if len(set(cols)) < 3:
                    return False
            elif len(unassigned) == 1:
                # One unassigned, check if it has at least one valid color
                upos = unassigned[0]
                # The other two positions (besides i and upos) are assigned
                other_two = [p for p in positions if p != i and p != upos]
                c1, c2 = colors[other_two[0]], colors[other_two[1]]
                # For color x for upos: {color, c1, c2, x} must have size >= 3
                # x is in {0,1,2}
                valid_x = []
                for x in range(3):
                    if len({color, c1, c2, x}) >= 3:
                        valid_x.append(x)
                if not valid_x:
                    return False
        return True
    
    def backtrack(pos):
        if time.time() - start_time > time_limit:
            return False
        if pos == n:
            return True
        
        # Try colors in random order
        colors_to_try = rng.sample([0, 1, 2], 3)
        for color in colors_to_try:
            if is_valid_assignment(pos, color):
                colors[pos] = color
                if forward_check(pos, color):
                    if backtrack(pos + 1):
                        return True
                colors[pos] = -1
        return False
    
    if backtrack(0):
        return list(colors)
    return None

def _greedy_3color(n=128, seed=0):
    """Randomized greedy with backtracking fallback."""
    rng = random.Random(seed)
    colors = [None] * n
    
    for i in range(n):
        valid_colors = set([0, 1, 2])
        # Check all 4-APs where i is the last element
        for d in range(1, (i) // 3 + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            if a < 0:
                break
            c1, c2, c3 = colors[a], colors[b], colors[c]
            for color in list(valid_colors):
                if len({c1, c2, c3, color}) < 3:
                    valid_colors.discard(color)
        
        if not valid_colors:
            return None
        
        # Prefer colors that are already used to minimize color count
        used = set(colors[:i])
        preferred = valid_colors & used
        if preferred:
            color = rng.choice(list(preferred))
        else:
            color = rng.choice(list(valid_colors))
        
        colors[i] = color
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    
    # First, get the seed fallback (guaranteed valid)
    seed_colors = _seed_fallback()
    best = seed_colors
    best_count = len(set(seed_colors))
    
    # Try to find a 3-coloring
    # First try randomized greedy with many seeds
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        result = _greedy_3color(n, seed)
        if result is not None:
            if _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count == 3:
                        return best
    
    # Try backtracking with various seeds and time limits
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        num_attempts = min(50, int(remaining_time / 2))
        for seed in range(num_attempts):
            if time.time() > deadline - 2:
                break
            time_limit = min(4.0, deadline - time.time() - 2)
            if time_limit < 0.5:
                break
            result = _backtrack_3color(n, seed, time_limit)
            if result is not None:
                if _validate(result):
                    count = len(set(result))
                    if count < best_count:
                        best = result
                        best_count = count
                        if best_count == 3:
                            return best
    
    return best
