import time
import random
import numpy as np

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

def _bitmask_greedy(n=128, k_colors=3, seed=0, max_restarts=1000):
    """
    Uses bitmasks for O(1) conflict checks.
    For each position i, we precompute the mask of forbidden colors for each 4-AP ending at i.
    Actually, the constraint is: for any 4-AP (a, a+d, a+2d, a+3d), the set of colors must have size >= 3.
    This is equivalent to: it is NOT the case that all 4 positions have the same color, 
    AND it is NOT the case that exactly 2 distinct colors are used (i.e., the multiset of colors is {c1, c1, c2, c2} or {c1, c1, c1, c2}).
    Wait, "at least 3 distinct colors" means the number of unique colors in the set {c_a, c_b, c_c, c_e} is >= 3.
    This fails if the number of unique colors is 1 or 2.
    
    Let's analyze the failure conditions for a 4-AP with colors (c1, c2, c3, c4):
    Failure if |{c1, c2, c3, c4}| <= 2.
    This happens if:
    1. All 4 are the same: c1=c2=c3=c4.
    2. Three of one, one of another: e.g., c1=c2=c3 != c4.
    3. Two of one, two of another: e.g., c1=c2, c3=c4, c1!=c3.
    
    When assigning color c to position i (which is the last element a+3d in some APs, or part of other APs),
    we need to ensure that for every 4-AP containing i, the condition holds.
    
    In a greedy forward pass (assigning 0 to n-1), when we assign position i,
    the only 4-APs that become "complete" or "partially constrained" in a way that restricts i's color 
    based *only* on previously assigned colors are those where i is the last element (a+3d=i).
    Why? Because if i is the first element (a=i), the other elements are in the future (unassigned).
    If i is the second (a+d=i), the 3rd and 4th are in the future.
    If i is the third (a+2d=i), the 4th is in the future.
    
    So, during a simple left-to-right greedy assignment, the only constraints that MUST be satisfied 
    using only already-assigned colors are those where i is the LAST element of the AP.
    However, we must ALSO ensure that we don't paint ourselves into a corner for future constraints?
    No, standard greedy just checks immediate constraints. If we only check APs ending at i, 
    we might create a situation where a future position j cannot be colored.
    But the task asks for a valid coloring. A simple greedy that only checks APs ending at i 
    produces a valid coloring IF and ONLY IF no future constraint is violated.
    But wait, if we only check APs ending at i, we guarantee that every AP ending at i is valid.
    Since every AP has a unique last element, checking APs ending at i for all i ensures ALL APs are valid.
    
    So, the problem reduces to:
    For each i, given colors of 0..i-1, find a color c for i such that for all d where i-3d >= 0,
    the set {colors[i-3d], colors[i-2d], colors[i-d], c} has size >= 3.
    
    Let's use bitmasks to speed this up.
    For a fixed i and d, let c1=colors[i-3d], c2=colors[i-2d], c3=colors[i-d].
    We need |{c1, c2, c3, c}| >= 3.
    
    If we have K colors (0 to K-1), we can represent the set of colors used in {c1, c2, c3} as a bitmask.
    Let mask = (1<<c1) | (1<<c2) | (1<<c3).
    We want to choose c such that popcount(mask | (1<<c)) >= 3.
    
    This condition fails if popcount(mask | (1<<c)) <= 2.
    
    Case 1: mask has 3 bits set. Then popcount(mask | (1<<c)) is always >= 3. So any c is valid for this AP.
    Case 2: mask has 2 bits set, say bits x and y.
       If we choose c=x or c=y, the new mask is still {x,y}, popcount 2. Invalid.
       If we choose c=z (z != x, z != y), the new mask is {x,y,z}, popcount 3. Valid.
       So c cannot be x or y.
    Case 3: mask has 1 bit set, say bit x.
       If we choose c=x, new mask {x}, popcount 1. Invalid.
       If we choose c=y (y != x), new mask {x,y}, popcount 2. Invalid.
       Wait. If mask={x}, and we choose c=y, the set is {x,y}. Size 2. This is < 3. So Invalid.
       So if mask has 1 bit, ANY c results in popcount <= 2?
       Let's re-read: "Every nonconstant four-term progression ... must use AT LEAST three distinct colors."
       If c1=c2=c3=x, and we pick c=x, colors are {x,x,x,x}, distinct count 1. Fail.
       If c1=c2=c3=x, and we pick c=y, colors are {x,x,x,y}, distinct count 2. Fail.
       So if c1=c2=c3, it is IMPOSSIBLE to satisfy the condition with a 4th color?
       Yes. If the first three are the same, the 4th must be different to get 2 distinct, but we need 3 distinct.
       So, if c1=c2=c3, there is NO valid color for the 4th position.
       
    This implies that a valid 3-coloring requires that for every 3-term AP (i-3d, i-2d, i-d), 
    the three colors are NOT all the same. In fact, they must contain at least 2 distinct colors?
    No, if they contain 2 distinct colors, say x, x, y.
    Mask = {x, y}.
    If we pick c=z (new), set is {x,y,z} (size 3). Valid.
    If we pick c=x, set is {x,y} (size 2). Invalid.
    If we pick c=y, set is {x,y} (size 2). Invalid.
    So if c1,c2,c3 are {x,x,y}, c must be a NEW color (not x, not y).
    
    If c1,c2,c3 are {x,y,z} (all distinct), mask={x,y,z}.
    Any c works.
    
    So the constraint on c for a specific AP (c1,c2,c3) is:
    - If c1==c2==c3: No valid c. (Dead end)
    - If c1,c2,c3 have exactly 2 distinct values {u,v}: c must not be u and c must not be v.
    - If c1,c2,c3 have exactly 3 distinct values: c can be anything.
    
    This is very strict. It means we need a coloring where no 3-term AP has all same colors,
    and if a 3-term AP has 2 colors, the 4th MUST be a 3rd color.
    
    Let's implement the greedy with bitmasks.
    For each i, we compute the union of forbidden colors from all APs ending at i.
    
    Forbidden set F is empty initially.
    For each d such that i-3d >= 0:
        c1 = colors[i-3d], c2 = colors[i-2d], c3 = colors[i-d]
        if c1 == c2 == c3:
            return None # Dead end
        elif len({c1, c2, c3}) == 2:
            F.add(c1)
            F.add(c2)
            F.add(c3) # effectively F = {c1, c2}
        else: # len == 3
            pass # No restriction
    
    Valid colors = All colors in [0, K-1] minus F.
    If Valid is empty, return None.
    Pick a color from Valid.
    
    To minimize colors used, we should prefer colors already in use?
    Or just pick the smallest valid color?
    The task says "Minimize the number of labels actually used".
    So we should try to use few distinct colors.
    
    Strategy:
    Try K=3, K=4, etc.
    For a given K, run greedy multiple times with different random seeds (randomizing the choice among valid colors).
    Prefer colors that are already used to keep the count low.
    
    """
    rng = random.Random(seed)
    
    for restart in range(max_restarts):
        colors = [0] * n
        used_colors = set([0])
        colors[0] = 0
        
        success = True
        for i in range(1, n):
            forbidden = set()
            
            # Check all APs ending at i
            # i = a + 3d => a = i - 3d
            max_d = i // 3
            for d in range(1, max_d + 1):
                a = i - 3 * d
                b = i - 2 * d
                c = i - d
                
                c1 = colors[a]
                c2 = colors[b]
                c3 = colors[c]
                
                if c1 == c2 == c3:
                    success = False
                    break
                elif c1 == c2 or c1 == c3 or c2 == c3:
                    # Exactly 2 distinct colors
                    forbidden.add(c1)
                    forbidden.add(c2)
                    forbidden.add(c3)
                else:
                    # 3 distinct colors, no restriction on 4th
                    pass
            
            if not success:
                break
            
            # Find valid colors
            valid_colors = [c for c in range(k_colors) if c not in forbidden]
            
            if not valid_colors:
                success = False
                break
            
            # Choose a color
            # Heuristic: Prefer colors already used to minimize distinct count
            # But sometimes we might be forced to use a new color.
            # Let's try to pick a color from used_colors if possible, else smallest valid.
            # Randomize the choice to explore different paths.
            
            # Sort valid colors: prefer used, then smaller
            # Actually, just pick randomly from valid, but bias towards used?
            # Let's try: 50% chance pick from used (if any valid), else pick smallest valid.
            
            valid_used = [c for c in valid_colors if c in used_colors]
            if valid_used and (rng.random() < 0.7 or not valid_colors):
                c = rng.choice(valid_used)
            else:
                # Pick smallest valid to keep numbers low? Or random?
                # Picking smallest might lead to using 0,1,2 early.
                c = rng.choice(valid_colors)
            
            colors[i] = c
            used_colors.add(c)
        
        if success:
            # Verify validity (should be valid by construction, but just in case)
            if _validate(colors):
                return colors
    
    return None

def _local_search_reduce_colors(colors, n=128, max_iters=500, seed=0):
    """
    Try to reduce the number of colors used in a valid coloring.
    If we have a coloring with K colors, try to merge two colors or recolor positions 
    to eliminate one color.
    """
    rng = random.Random(seed)
    current = list(colors)
    current_colors = set(current)
    
    # We want to minimize len(current_colors).
    # Try to eliminate a color by recoloring all positions with that color to other colors.
    
    for _ in range(max_iters):
        if len(current_colors) <= 3:
            break
            
        # Pick a color to eliminate
        # Prefer colors with fewer positions? Or random?
        # Let's pick a random color that is not 0,1,2? 
        # Or just any color.
        target_color = rng.choice(list(current_colors))
        
        positions = [i for i in range(n) if current[i] == target_color]
        
        # Try to recolor these positions one by one to other colors
        # This is a sub-problem.
        # Let's just try a simple greedy recoloring of these positions.
        # For each pos in positions, try to change current[pos] to a color in current_colors - {target_color}
        # such that the new coloring is valid.
        
        new_current = list(current)
        success = True
        
        # We need to check validity for all APs involving the changed positions.
        # Since we are changing multiple positions, we should check validity at the end 
        # or use a more careful backtracking. 
        # Given the time limit, a simple "try and validate at end" might be too slow if it fails often.
        # But n=128 is small.
        
        for pos in positions:
            # Try other colors
            other_colors = [c for c in current_colors if c != target_color]
            rng.shuffle(other_colors)
            
            found = False
            for new_color in other_colors:
                new_current[pos] = new_color
                
                # Quick check: does this violate any AP containing pos?
                valid = True
                # Check APs where pos is a, b, c, or e
                # To be safe, just check all APs containing pos.
                # Precompute APs containing pos?
                
                # For now, just check all APs in the whole array? No, too slow.
                # Check APs where pos is one of the elements.
                
                # pos is a: a=pos, d=1..
                for d in range(1, (n - pos) // 3 + 1):
                    e = pos + 3 * d
                    if e >= n: break
                    cols = {new_current[pos], new_current[pos+d], new_current[pos+2*d], new_current[e]}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid: continue
                
                # pos is b: a=pos-d, b=pos
                for d in range(1, pos // 1 + 1):
                    a = pos - d
                    c = pos + d
                    e = pos + 2 * d
                    if e >= n: break
                    cols = {new_current[a], new_current[pos], new_current[c], new_current[e]}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid: continue
                
                # pos is c: a=pos-2d, b=pos-d, c=pos
                for d in range(1, pos // 2 + 1):
                    a = pos - 2 * d
                    b = pos - d
                    e = pos + d
                    if e >= n: break
                    cols = {new_current[a], new_current[b], new_current[pos], new_current[e]}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid: continue
                
                # pos is e: a=pos-3d, b=pos-2d, c=pos-d
                for d in range(1, pos // 3 + 1):
                    a = pos - 3 * d
                    b = pos - 2 * d
                    c = pos - d
                    cols = {new_current[a], new_current[b], new_current[c], new_current[pos]}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid: continue
                
                if valid:
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success:
            # Check if we actually reduced the color count
            new_colors = set(new_current)
            if len(new_colors) < len(current_colors):
                current = new_current
                current_colors = new_colors
            # If not reduced, maybe we should revert? 
            # The goal is to minimize colors. If we didn't reduce, we might have made it worse or same.
            # Since we are trying to eliminate target_color, if we succeeded in recoloring 
            # but target_color is still present (impossible, we recolored all positions of it),
            # then target_color is gone. So len(new_colors) < len(current_colors) should be true 
            # unless some other color was introduced? No, we only used existing colors.
            # So if success, we definitely removed target_color.
            # Wait, if we recolor pos to new_color, and new_color was already in current_colors,
            # then target_color is removed from the set of used colors.
            # So len(new_colors) = len(current_colors) - 1.
            
            # So if success, we have improved.
            # If not success, we revert (current remains unchanged).
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    
    # 1. Get seed fallback
    seed_colors = _seed_fallback()
    best = seed_colors
    best_count = len(set(seed_colors))
    
    # 2. Try bitmask greedy for K=3, 4, 5...
    # Prioritize K=3 since it's the theoretical minimum (can't be < 3).
    
    for K in range(3, 8):
        if time.time() > deadline - 10:
            break
            
        # Run multiple restarts
        num_restarts = 1000
        found = False
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            result = _bitmask_greedy(n, K, seed=r)
            if result is not None:
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    found = True
                    if best_count == 3:
                        return best
        
        # If we found a solution with K colors, we can try to improve it with local search
        # But if K=3, we are done.
        if found and K > 3:
            # Try local search to reduce colors further
            # But local search is slow. Only do it if we have time.
            if time.time() < deadline - 20:
                improved = _local_search_reduce_colors(best, n, max_iters=50, seed=42)
                imp_count = len(set(improved))
                if imp_count < best_count:
                    best = improved
                    best_count = imp_count
                    if best_count == 3:
                        return best
        
        # If we found a solution with K=3, return immediately.
        if best_count == 3:
            return best
            
    # 3. If we didn't find a better solution, return the best we have (seed fallback or K=4/5 solution)
    return best

if __name__ == "__main__":
    # Quick self-test
    c = construct()
    print("Colors used:", len(set(c)))
    print("Valid:", _validate(c))
