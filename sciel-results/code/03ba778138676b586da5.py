import time
import random

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

def _count_colors(colors):
    return len(set(colors))

def _precompute_ap_constraints(n=128):
    """Precompute for each position i, the list of 4-APs containing i.
    Returns a list where ap_constraints[i] is a list of (a, b, c, e) tuples
    where i is one of the positions."""
    ap_constraints = [[] for _ in range(n)]
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                ap = (a, b, c, e)
                ap_constraints[a].append(ap)
                ap_constraints[b].append(ap)
                ap_constraints[c].append(ap)
                ap_constraints[e].append(ap)
    return ap_constraints

def _greedy_with_bitmasks(n=128, seed=0, max_colors=3):
    """Greedy assignment using precomputed constraints and bitmask conflict checks."""
    rng = random.Random(seed)
    ap_constraints = _precompute_ap_constraints(n)
    colors = [-1] * n
    
    for i in range(n):
        # Determine which colors are valid for position i
        valid = [True] * max_colors
        
        for ap in ap_constraints[i]:
            a, b, c, e = ap
            # Get colors of other three positions (if assigned)
            other_positions = []
            for pos in (a, b, c, e):
                if pos != i and colors[pos] != -1:
                    other_positions.append(colors[pos])
            
            if len(other_positions) == 3:
                # All other positions assigned, check each color
                for color in range(max_colors):
                    all_cols = set([color] + other_positions)
                    if len(all_cols) < 3:
                        valid[color] = False
            elif len(other_positions) == 2:
                # Two other positions assigned, one unassigned
                # For each color, check if it's possible for the unassigned
                # position to make the constraint satisfiable
                for color in range(max_colors):
                    current_set = set([color] + other_positions)
                    # The unassigned position can be any color from 0..max_colors-1
                    # We need: exists x such that len(current_set | {x}) >= 3
                    if len(current_set) >= 3:
                        continue  # Already satisfied regardless of x
                    else:
                        # len(current_set) <= 2, need x to add a new color
                        # There exists x not in current_set if max_colors > len(current_set)
                        if max_colors <= len(current_set):
                            valid[color] = False
        
        # Filter valid colors
        valid_colors = [c for c in range(max_colors) if valid[c]]
        
        if not valid_colors:
            return None
        
        # Prefer already-used colors to minimize color count
        used = set(colors[:i])
        preferred = [c for c in valid_colors if c in used]
        if preferred:
            color = rng.choice(preferred)
        else:
            color = rng.choice(valid_colors)
        
        colors[i] = color
    
    return colors

def _local_search(colors, n=128, seed=0, iterations=10000, time_limit=5.0):
    """Local search to reduce number of colors used."""
    rng = random.Random(seed)
    start_time = time.time()
    ap_constraints = _precompute_ap_constraints(n)
    
    current = list(colors)
    current_count = _count_colors(current)
    best = list(current)
    best_count = current_count
    
    for _ in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random position to reassign
        i = rng.randrange(n)
        old_color = current[i]
        
        # Try each color
        for new_color in range(128):
            if new_color == old_color:
                continue
            
            # Check if new_color is valid
            valid = True
            for ap in ap_constraints[i]:
                a, b, c, e = ap
                other_positions = []
                for pos in (a, b, c, e):
                    if pos != i:
                        other_positions.append(current[pos])
                
                all_cols = set([new_color] + other_positions)
                if len(all_cols) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Try the new color
            current[i] = new_color
            new_count = _count_colors(current)
            
            if new_count < best_count:
                best_count = new_count
                best = list(current)
            elif new_count == best_count:
                # Accept with some probability (simulated annealing-like)
                if rng.random() < 0.3:
                    best = list(current)
            else:
                # Revert
                current[i] = old_color
        
        # Occasionally try to remove a color by remapping
        if rng.random() < 0.1:
            used_colors = sorted(set(current))
            if len(used_colors) > 3:
                # Try to merge two colors
                c1, c2 = rng.sample(used_colors, 2)
                # Check if we can merge c2 into c1
                test = list(current)
                for i in range(n):
                    if test[i] == c2:
                        test[i] = c1
                
                if _validate(test):
                    test_count = _count_colors(test)
                    if test_count < best_count:
                        best_count = test_count
                        best = test
                        current = test
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 128
    ap_constraints = _precompute_ap_constraints(n)
    
    # Try greedy with different seeds and max_colors
    best = None
    best_count = 128
    
    # First, try to find a valid coloring with few colors
    for max_colors in [3, 4, 5]:
        if time.time() > deadline - 15:
            break
        
        for seed in range(50):
            if time.time() > deadline - 15:
                break
            
            result = _greedy_with_bitmasks(n, seed, max_colors)
            if result is not None and _validate(result):
                count = _count_colors(result)
                if count < best_count:
                    best_count = count
                    best = result
                    if best_count == 3:
                        # Try to improve with local search
                        if time.time() < deadline - 10:
                            improved = _local_search(best, n, seed, iterations=5000, time_limit=5.0)
                            improved_count = _count_colors(improved)
                            if improved_count < best_count:
                                best_count = improved_count
                                best = improved
                        return best
    
    # If greedy didn't find good results, try more seeds
    if best is None or best_count > 5:
        for seed in range(200):
            if time.time() > deadline - 10:
                break
            
            result = _greedy_with_bitmasks(n, seed, 6)
            if result is not None and _validate(result):
                count = _count_colors(result)
                if count < best_count:
                    best_count = count
                    best = result
                    if best_count <= 4:
                        break
    
    # Final local search polish
    if best is not None and time.time() < deadline - 5:
        remaining = min(5.0, deadline - time.time() - 2)
        improved = _local_search(best, n, seed=42, iterations=20000, time_limit=remaining)
        improved_count = _count_colors(improved)
        if improved_count < best_count:
            best_count = improved_count
            best = improved
    
    if best is None:
        # Fallback: simple deterministic coloring
        best = [i % 4 for i in range(n)]
    
    return best
