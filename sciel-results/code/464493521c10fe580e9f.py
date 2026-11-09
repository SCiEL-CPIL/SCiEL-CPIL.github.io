import time
import random
import sys

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    n = 128

    # Precompute constraints for each position x (0-indexed)
    # constraints[x] contains tuples (i1, i2, i3) such that x, i1, i2, i3 form a 4-term AP
    # The AP is a, a+d, a+2d, a+3d.
    # If x is the last term (a+3d), then a = x - 3d.
    # i1 = a, i2 = a+d, i3 = a+2d.
    # We must ensure a >= 0.
    # Also, x can be any term in the AP.
    # It is easier to define constraints for "assigning color to x".
    # When we assign color to x, we must check all APs that include x.
    # An AP is defined by (start, diff).
    # Let's precompute for each index x, the list of other 3 indices in each AP containing x.
    
    # For each AP (a, d), indices are a, a+d, a+2d, a+3d.
    # For each index i in this AP, the other 3 are the rest.
    # We store for each index i, a list of tuples (j, k, l) representing the other 3 indices.
    
    other_indices = [[] for _ in range(n)]
    
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            idxs = [a, a + d, a + 2 * d, a + 3 * d]
            # For each index in the AP, add the other 3
            for i in range(4):
                others = [idxs[j] for j in range(4) if j != i]
                other_indices[idxs[i]].append(tuple(others))

    def is_valid(x, c, colors):
        """
        Check if assigning color c to position x is valid given current colors.
        colors is a list where colors[i] is the color of i, or -1 if unassigned.
        """
        for o1, o2, o3 in other_indices[x]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            
            # If any of the others are unassigned, we can't check yet?
            # In a forward greedy, we assign 0, 1, ..., n-1.
            # So when assigning x, all indices < x are assigned.
            # However, an AP can have indices > x.
            # If we use forward greedy, we only know colors for < x.
            # We cannot check constraints involving indices > x until they are assigned.
            # This makes simple forward greedy invalid for global constraints unless we look ahead or use backtracking.
            
            # Alternative: Precompute constraints only for "past" indices?
            # No, the constraint is global.
            
            # Let's stick to the logic in Inspiration 3:
            # It assumed we are filling x, and checking constraints where x is the LAST element?
            # "AP: a, a+d, a+2d, a+3d = x. a = x - 3d."
            # This only checks APs where x is the largest index.
            # But a 4-AP can have x as the smallest, 2nd, or 3rd element.
            # If we only check when x is the largest, we miss constraints where x is smaller.
            # However, when the largest element is assigned, all 4 are known, so we can check then.
            # So, if we iterate x from 0 to 127, we can check validity of the APs where x is the MAX index.
            # This ensures that by the time we finish, all APs have been checked (when their max index was assigned).
            
            # So, we only need to check APs where x is the max index.
            # Let's re-precompute: max_constraints[x] = list of (i1, i2, i3) where x is the 4th element.
            
            pass # This logic needs to be fixed in the precomputation step below.

    # Re-precompute: constraints where x is the LAST element of the AP.
    # AP: a, a+d, a+2d, a+3d = x.
    # a = x - 3d.
    # i1 = a, i2 = a+d, i3 = a+2d.
    # We need a >= 0.
    
    last_constraints = [[] for _ in range(n)]
    for x in range(n):
        # x is the last element.
        # a = x - 3d.
        # d can be 1, 2, ... such that x - 3d >= 0.
        # 3d <= x => d <= x // 3.
        for d in range(1, x // 3 + 1):
            a = x - 3 * d
            i1 = a
            i2 = a + d
            i3 = a + 2 * d
            last_constraints[x].append((i1, i2, i3))

    def check_assignment(x, c, colors):
        """
        Check if assigning color c to position x satisfies all 4-AP constraints where x is the last element.
        colors[i] must be assigned for all i < x.
        """
        for i1, i2, i3 in last_constraints[x]:
            c1 = colors[i1]
            c2 = colors[i2]
            c3 = colors[i3]
            
            # We need len({c1, c2, c3, c}) >= 3
            
            # Fast check without sets:
            # Count distinct values among c1, c2, c3, c.
            # If c1 == c2 == c3:
            #   Set is {c1, c}. Size 1 if c==c1, else 2. Always < 3. Invalid.
            if c1 == c2 and c2 == c3:
                return False
            
            # If c1 == c2 and c3 != c1:
            #   Set is {c1, c3, c}.
            #   Size is 2 if c == c1 or c == c3.
            #   Size is 3 if c != c1 and c != c3.
            #   So invalid if c == c1 or c == c3.
            if c1 == c2:
                if c == c1 or c == c3:
                    return False
                continue
            
            # If c1 != c2 and c2 == c3:
            #   Set is {c1, c2, c}.
            #   Invalid if c == c1 or c == c2.
            if c2 == c3:
                if c == c1 or c == c2:
                    return False
                continue
            
            # If c1 == c3 and c2 != c1:
            #   Set is {c1, c2, c}.
            #   Invalid if c == c1 or c == c2.
            if c1 == c3:
                if c == c1 or c == c2:
                    return False
                continue
            
            # If c1, c2, c3 are all distinct:
            #   Set is {c1, c2, c3, c}.
            #   Size is at least 3. Always valid.
            continue
            
        return True

    def greedy_construct(k, seed_val):
        """
        Try to construct a valid coloring using at most k colors.
        Returns list of colors or None.
        """
        rng = random.Random(seed_val)
        colors = [-1] * n
        used_colors = set()
        
        # We can use backtracking, but for n=128, simple greedy with restarts might be enough.
        # Let's try a simple greedy first. If it fails, we restart.
        
        for x in range(n):
            # Determine valid colors for x
            valid = []
            for c in range(k):
                if check_assignment(x, c, colors):
                    valid.append(c)
            
            if not valid:
                return None
            
            # Heuristic: Prefer colors already used to minimize distinct count
            # But we are fixing k. We just need to find *a* valid coloring with <= k colors.
            # To minimize actual colors used later, we can bias towards used colors.
            
            # Shuffle valid colors to introduce randomness
            rng.shuffle(valid)
            
            # Prioritize used colors
            # Split valid into used and new
            used_valid = [c for c in valid if c in used_colors]
            new_valid = [c for c in valid if c not in used_colors]
            
            if used_valid:
                choice = used_valid[0] # Picked randomly from shuffled used_valid
            else:
                choice = new_valid[0]
                
            colors[x] = choice
            used_colors.add(choice)
            
        return colors

    # Search for minimum k
    best_result = None
    best_k = 128
    
    # Try k=3
    if time.time() < DEADLINE - 5:
        for seed in range(200):
            if time.time() > DEADLINE - 5: break
            res = greedy_construct(3, seed)
            if res is not None:
                # Verify
                ok = True
                for x in range(n):
                    for i1, i2, i3 in last_constraints[x]:
                        if len({res[i1], res[i2], res[i3], res[x]}) < 3:
                            ok = False
                            break
                    if not ok: break
                if ok:
                    return res # Score will be Seed/3
    
    # Try k=4
    if time.time() < DEADLINE - 5:
        for seed in range(200):
            if time.time() > DEADLINE - 5: break
            res = greedy_construct(4, seed)
            if res is not None:
                ok = True
                for x in range(n):
                    for i1, i2, i3 in last_constraints[x]:
                        if len({res[i1], res[i2], res[i3], res[x]}) < 3:
                            ok = False
                            break
                    if not ok: break
                if ok:
                    # Count distinct
                    distinct = len(set(res))
                    if distinct < best_k:
                        best_k = distinct
                        best_result = res
                    # If we found a 3-coloring inside k=4 search, return immediately?
                    # The greedy might use only 3 colors even if k=4 is allowed.
                    if distinct == 3:
                        return res

    # Try k=5
    if time.time() < DEADLINE - 5:
        for seed in range(100):
            if time.time() > DEADLINE - 5: break
            res = greedy_construct(5, seed)
            if res is not None:
                ok = True
                for x in range(n):
                    for i1, i2, i3 in last_constraints[x]:
                        if len({res[i1], res[i2], res[i3], res[x]}) < 3:
                            ok = False
                            break
                    if not ok: break
                if ok:
                    distinct = len(set(res))
                    if distinct < best_k:
                        best_k = distinct
                        best_result = res
                    if distinct == 3:
                        return res

    # Fallback: Original greedy (known valid, likely uses more colors)
    if best_result is None:
        colors = []
        for x in range(1, 129):
            found = False
            for color in range(128):
                valid = True
                # Check all APs where x (1-based) is the last element
                # a = x - 3d
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    if a < 1: continue
                    i1 = a - 1
                    i2 = a + d - 1
                    i3 = a + 2 * d - 1
                    if len({colors[i1], colors[i2], colors[i3], color}) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    found = True
                    break
            if not found:
                # Should not happen with 128 colors
                colors.append(0)
        return colors

    return best_result

