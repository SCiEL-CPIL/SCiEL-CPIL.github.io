"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _check_validity(colors, n=128):
    """Check if coloring satisfies: every nonconstant 4-term AP has >=3 distinct colors."""
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            c0 = colors[a]
            c1 = colors[a + d]
            c2 = colors[a + 2*d]
            c3 = colors[a + 3*d]
            if len({c0, c1, c2, c3}) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _precompute_constraints():
    """Precompute for each position x (0-indexed) the list of (i1, i2, i3) 
    such that x, i1, i2, i3 form a 4-term AP in positions (1-indexed: a, a+d, a+2d, a+3d).
    Here x is the last element (a+3d). 
    So a = (x+1) - 3d.
    i1 = a-1, i2 = a+d-1, i3 = a+2d-1 (0-indexed).
    """
    constraints = [[] for _ in range(128)]
    for x in range(128):
        pos = x + 1  # 1-indexed position
        # d >= 1, a = pos - 3d >= 1 => 3d <= pos - 1 => d <= (pos-1)//3
        for d in range(1, (pos - 1) // 3 + 1):
            a = pos - 3 * d
            if a < 1:
                break
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            constraints[x].append((i1, i2, i3))
    return constraints

def _greedy_construct(max_colors, constraints, seed=0):
    """Greedy construction with backtracking-lite (restarts) using max_colors palette."""
    if max_colors < 2:
        return None
    
    # We'll try multiple random restarts if the first greedy fails
    # But for speed, we first try a deterministic greedy with a specific heuristic.
    
    # Heuristic: For each position, try colors in order of (used? 0 : 1, color_id)
    # This minimizes new colors.
    
    def attempt(seed_val):
        rng = random.Random(seed_val)
        colors = [None] * 128
        used = set()
        
        for x in range(128):
            valid_colors = []
            for c in range(max_colors):
                ok = True
                for i1, i2, i3 in constraints[x]:
                    c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                    # Check if {c1, c2, c3, c} has size >= 3
                    # Fast check:
                    if c1 == c2 == c3:
                        # {c1, c} has size 1 or 2. Invalid.
                        ok = False
                        break
                    # Count distinct among c1, c2, c3
                    # If distinct count is 3, adding c makes it 3 or 4. Always valid.
                    # If distinct count is 2 (e.g., A, A, B), then c must be != A and != B?
                    # No. {A, A, B, c}. 
                    # If c=A: {A, B} size 2. Invalid.
                    # If c=B: {A, B} size 2. Invalid.
                    # If c=C (new): {A, B, C} size 3. Valid.
                    # If distinct count is 1 (A, A, A) handled above.
                    
                    # Let's just compute the set for simplicity, it's small.
                    if len({c1, c2, c3, c}) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Sort valid colors: prefer already used, then lower id
            valid_colors.sort(key=lambda c: (0 if c in used else 1, c))
            
            # Randomly pick from top 3 valid to allow some diversity in restarts?
            # For now, deterministic pick (smallest preferred)
            chosen = valid_colors[0]
            colors[x] = chosen
            used.add(chosen)
        
        return colors

    # Try deterministic first
    res = attempt(0)
    if res:
        return res
    
    # If deterministic fails, try a few random restarts with different tie-breaking
    # But since we sort, there's no tie-breaking randomness unless we shuffle valid_colors.
    # Let's modify attempt to allow randomness in selection.
    
    def attempt_random(seed_val):
        rng = random.Random(seed_val)
        colors = [None] * 128
        used = set()
        
        for x in range(128):
            valid_colors = []
            for c in range(max_colors):
                ok = True
                for i1, i2, i3 in constraints[x]:
                    c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                    if len({c1, c2, c3, c}) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Heuristic: 80% chance pick best (min new colors, min id), 20% random valid
            valid_colors.sort(key=lambda c: (0 if c in used else 1, c))
            if rng.random() < 0.8:
                chosen = valid_colors[0]
            else:
                # Pick from top 3 to avoid terrible choices
                top_k = valid_colors[:min(3, len(valid_colors))]
                chosen = rng.choice(top_k)
            
            colors[x] = chosen
            used.add(chosen)
        
        return colors

    for s in range(10):
        res = attempt_random(s)
        if res:
            return res
            
    return None

def _local_search(initial_colors, constraints, max_iterations=5000, deadline=None):
    """Local search to reduce number of colors.
    Strategy: Identify a color with highest frequency. Try to recolor those positions to other colors.
    If we can recolor ALL positions of color C to other colors such that validity holds, we remove C.
    """
    colors = initial_colors[:]
    n = 128
    
    def count_colors(c):
        return len(set(c))
    
    current_count = count_colors(colors)
    
    for _ in range(max_iterations):
        if deadline and time.time() > deadline:
            break
            
        # Find the color to eliminate
        freq = defaultdict(int)
        for c in colors:
            freq[c] += 1
        
        # Target the most frequent color
        if not freq:
            break
        target_color = max(freq, key=lambda k: freq[k])
        
        # List of positions with target_color
        positions_to_change = [i for i in range(n) if colors[i] == target_color]
        
        # Try to recolor each position in positions_to_change to a different valid color
        # We do this greedily: pick a position, find a valid alternative color (not target_color), assign it.
        # If we can recolor ALL positions of target_color, then we have reduced the color count.
        
        # To make it more robust, we can try to recolor them one by one.
        # But recoloring one might make another invalid? No, recoloring to a valid color maintains validity.
        # However, we must ensure that when we recolor pos p to c', c' is valid for p given CURRENT colors.
        # Since we are changing colors, the "current" state changes.
        
        # Simple approach: 
        # 1. Make a copy of colors.
        # 2. For each pos in positions_to_change:
        #    - Find a valid color c' != target_color for pos in the current copy.
        #    - If found, assign c' to pos in the copy.
        #    - If not found, fail this attempt.
        # 3. If all succeeded, update colors to the copy.
        
        new_colors = colors[:]
        success = True
        
        # Randomize order of positions to change to avoid bias?
        # Let's shuffle positions_to_change
        pos_list = positions_to_change[:]
        random.shuffle(pos_list)
        
        for p in pos_list:
            found = False
            for c_new in range(128):
                if c_new == target_color:
                    continue
                # Check if c_new is valid for p in new_colors
                ok = True
                for i1, i2, i3 in constraints[p]:
                    c1, c2, c3 = new_colors[i1], new_colors[i2], new_colors[i3]
                    if len({c1, c2, c3, c_new}) < 3:
                        ok = False
                        break
                if ok:
                    new_colors[p] = c_new
                    found = True
                    break
            if not found:
                success = False
                break
        
        if success:
            colors = new_colors
            current_count = count_colors(colors)
        else:
            # If we fail to eliminate the most frequent color, try the next most frequent?
            # Or just break to avoid getting stuck in a loop?
            # Let's try the next most frequent color.
            # Sort colors by frequency desc
            sorted_colors = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)
            eliminated = False
            for tc in sorted_colors:
                if tc == target_color:
                    continue
                pos_list2 = [i for i in range(n) if colors[i] == tc]
                new_colors2 = colors[:]
                success2 = True
                for p in pos_list2:
                    found2 = False
                    for c_new in range(128):
                        if c_new == tc:
                            continue
                        ok2 = True
                        for i1, i2, i3 in constraints[p]:
                            c1, c2, c3 = new_colors2[i1], new_colors2[i2], new_colors2[i3]
                            if len({c1, c2, c3, c_new}) < 3:
                                ok2 = False
                                break
                        if ok2:
                            new_colors2[p] = c_new
                            found2 = True
                            break
                    if not found2:
                        success2 = False
                        break
                if success2:
                    colors = new_colors2
                    current_count = count_colors(colors)
                    eliminated = True
                    break
            if not eliminated:
                # No color could be eliminated in this pass.
                # To avoid infinite loop, we can break or randomize.
                # For now, break.
                break
                
    return colors

def construct():
    """Return n=128 integer color labels in 0..127. Every nonconstant four-term progression
    of positions must use AT LEAST three distinct colors. Minimize number of labels used."""
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    constraints = _precompute_constraints()
    
    best_result = None
    best_count = 128
    
    # Try to construct with small number of colors
    for max_c in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline:
            break
            
        # Try greedy construction
        res = _greedy_construct(max_c, constraints)
        
        if res is not None:
            # Verify
            if _check_validity(res):
                cnt = _count_colors(res)
                if cnt < best_count:
                    best_count = cnt
                    best_result = res[:]
                    
                # Try to improve with local search
                if time.time() < deadline - 5.0:
                    improved = _local_search(res, constraints, max_iterations=2000, deadline=deadline - 2.0)
                    if improved is not None and _check_validity(improved):
                        imp_cnt = _count_colors(improved)
                        if imp_cnt < best_count:
                            best_count = imp_cnt
                            best_result = improved[:]
                            
                # If we found a solution with 2 colors, we can't do better
                if best_count == 2:
                    return best_result
    
    # If no small color solution found, use the original greedy as fallback
    if best_result is None:
        colors = []
        for x in range(1, 129):
            for color in range(128):
                # Check constraints for position x (1-indexed)
                # Constraints are precomputed for 0-indexed.
                # x is 1-indexed, so index is x-1.
                idx = x - 1
                ok = True
                for i1, i2, i3 in constraints[idx]:
                    c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                    if len({c1, c2, c3, color}) < 3:
                        ok = False
                        break
                if ok:
                    colors.append(color)
                    break
        if _check_validity(colors):
            best_result = colors
            best_count = _count_colors(colors)
    
    # Final safety check
    if best_result is None:
        # This should not happen, but return a dummy valid sequence if needed
        # A simple valid sequence: color[i] = i % 128? No, that uses 128 colors.
        # Just return the greedy result which is known to be valid.
        colors = []
        for x in range(1, 129):
            for color in range(128):
                idx = x - 1
                ok = True
                for i1, i2, i3 in constraints[idx]:
                    c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                    if len({c1, c2, c3, color}) < 3:
                        ok = False
                        break
                if ok:
                    colors.append(color)
                    break
        best_result = colors
    
    return best_result
# EVOLVE-BLOCK-END
