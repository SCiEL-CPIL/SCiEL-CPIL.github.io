import time
import numpy as np
from itertools import product

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d), 1-indexed a,d
# Convert to 0-indexed internally.
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

def _progression_ok(colors, idx_set):
    # colors is a list of assigned colors or -1 for unassigned
    # idx_set: tuple of 4 indices
    present = set()
    for i in idx_set:
        c = colors[i]
        if c != -1:
            present.add(c)
    if len(present) >= 3:
        return True
    # If we already have 2 colors, adding more can't reduce to <3 unless... 
    # Actually if present has 2 colors, the remaining must include a 3rd color.
    # We can't confirm yet. Return "not yet violated".
    return True  # not a violation yet (could become 3 later)

def _progression_violated(colors, idx_set, k_colors):
    # Determine if this progression is already IMPOSSIBLE to satisfy
    present = set()
    unassigned = 0
    for i in idx_set:
        c = colors[i]
        if c == -1:
            unassigned += 1
        else:
            present.add(c)
    if len(present) >= 3:
        return False  # already satisfied
    # Need to end with >=3 distinct colors
    if len(present) + unassigned < 3:
        return True  # impossible
    return False

def _find_periodic_pattern(period, k_colors, deadline):
    """
    Exhaustively search for a periodic pattern of given period and k_colors
    that satisfies the 3-color progression constraint for all x <= 128.
    Returns a list of 128 colors if found, else None.
    """
    # A periodic pattern is determined by its first 'period' values.
    # For each assignment of colors[0:period], check if it works for all 128 positions.
    
    # Precompute: for a given periodic pattern, what is the color at position i?
    # color[i] = pattern[i % period]
    
    # We need to check all APs: for each AP (i1, i2, i3, i4), the colors must have >= 3 distinct values.
    # For a periodic pattern, color[j] = pattern[j % period]
    
    # Enumerate all possible patterns of length 'period' with k_colors
    # But we can prune: the pattern must use at least 3 colors (otherwise no AP can have 3 colors)
    
    # Optimization: only consider patterns that use exactly 3 colors (minimize colors used)
    # Generate all patterns of length 'period' using colors 0,1,2 that use all 3 colors
    
    # To reduce search space, we can fix pattern[0]=0, pattern[1]=1 (symmetry breaking)
    # and require pattern to use color 2 somewhere
    
    valid_patterns = []
    
    # Try all assignments of period positions with colors 0,1,2
    # Use symmetry breaking: pattern[0]=0
    # Then try all combinations for remaining period-1 positions
    
    # For small periods (2-8), this is feasible
    if period <= 8:
        # Enumerate all patterns with pattern[0]=0
        for rest in product(range(k_colors), repeat=period-1):
            pattern = [0] + list(rest)
            
            # Check if pattern uses at least 3 colors
            if len(set(pattern)) < 3:
                continue
            
            # Check if this periodic pattern satisfies all APs for positions 0..127
            valid = True
            for ap in _ALL_APS:
                colors_in_ap = {pattern[i % period] for i in ap}
                if len(colors_in_ap) < 3:
                    valid = False
                    break
            
            if valid:
                # Expand to 128 positions
                expanded = [pattern[i % period] for i in range(128)]
                valid_patterns.append(expanded)
                
                # Return first valid pattern found (to save time)
                return expanded
    
    return None

def construct():
    start = time.time()
    DEADLINE = 75.0  # safety margin before 90s

    n = 128
    ap_count = len(_ALL_APS)
    # Precompute for each position, which progressions it participates in (as a member)
    pos_in_aps = [[] for _ in range(n)]
    for ap in _ALL_APS:
        for i in ap:
            pos_in_aps[i].append(ap)

    best = None
    best_ncolors = n  # start with worst

    # First, try to find a periodic pattern with 3 colors
    # Try periods from small to large
    for period in [3, 4, 5, 6, 7, 8, 9, 10, 12]:
        if time.time() > DEADLINE:
            break
        
        # Try to find a valid periodic pattern with 3 colors
        pattern = _find_periodic_pattern(period, 3, DEADLINE)
        if pattern is not None:
            best = pattern
            best_ncolors = 3
            break
    
    # If no periodic pattern found, try backtracking search
    if best is None:
        # Try increasing number of colors
        for k_colors in range(3, 6):
            if time.time() > DEADLINE:
                break
            # Backtracking search with symmetry breaking and ordering
            colors = [-1] * n

            # Heuristic: order positions by most-constrained first
            # Compute degree (number of APs each position is in)
            degree = [len(pos_in_aps[i]) for i in range(n)]
            order = list(range(n))
            order.sort(key=lambda i: (-degree[i], i))

            # We'll assign in 'order' but track index
            pos_to_fill = order
            assigned_count = 0

            # Forward checking: for each position, allowed colors given current assignment
            def allowed_colors(pos):
                al = set(range(k_colors))
                # Check each AP containing pos: if the other 3 are assigned, ensure adding any color keeps possibility
                for ap in pos_in_aps[pos]:
                    others = [i for i in ap if i != pos]
                    oc = {colors[i] for i in others}
                    unassigned_others = sum(1 for i in others if colors[i] == -1)
                    # After we assign pos with color c, the set of colors in AP must be extendable to >=3
                    # Currently oc has some colors, and unassigned_others can add more.
                    # We need: (oc ∪ {c}) ∪ (future) to reach >=3.
                    # Worst case future adds at most unassigned_others new colors, but they could be same as existing.
                    # The best they can do is add new colors not in oc∪{c}, up to unassigned_others distinct new ones.
                    current = oc | {c}
                    max_possible = len(current) + unassigned_others
                    if max_possible < 3:
                        al.discard(c)
                return al

            solution_found = False
            node_count = 0

            def backtrack(idx_in_order):
                nonlocal solution_found, node_count
                if time.time() > DEADLINE:
                    return False
                if solution_found:
                    return True
                node_count += 1
                if node_count % 5000 == 0 and time.time() > DEADLINE:
                    return False

                if idx_in_order == n:
                    # All assigned; verify all APs
                    for ap in _ALL_APS:
                        if len({colors[i] for i in ap}) < 3:
                            return False
                    solution_found = True
                    return True

                pos = pos_to_fill[idx_in_order]
                for c in allowed_colors(pos):
                    colors[pos] = c
                    # Check no AP containing pos is now violated (impossible)
                    ok = True
                    for ap in pos_in_aps[pos]:
                        if _progression_violated(colors, ap, k_colors):
                            ok = False
                            break
                    if ok:
                        if backtrack(idx_in_order + 1):
                            return True
                    colors[pos] = -1
                return False

            if backtrack(0):
                # Extract used colors
                used = sorted(set(c for c in colors if c != -1))
                # Remap to consecutive labels 0..k-1 (or keep original; labels need not be consecutive)
                # Keep as is; count distinct
                best = colors[:]
                best_ncolors = len(used)
                break
            else:
                # Didn't find with k_colors within time; try more colors
                pass

    if best is not None:
        # Ensure all are valid non-negative and within 0..127
        return best
    else:
        # Fallback: use seed-like greedy but with more colors, or a simple pattern
        # Fallback: color by i mod 3 gives 3 colors but may not satisfy. 
        # Safer fallback: assign each position a unique color (128 colors) -> always satisfies.
        # But that's many colors. Better: try a quick greedy with many colors.
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3 + 1) if x-1-3*d >= 0)
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                            for d in range(1, (x-1)//2 + 1) if x-1-2*d >= 0)):
                    colors.append(color)
                    break
            else:
                colors.append(0)
        return colors

# Quick self-check during module load is not allowed (must be fast); skip.
