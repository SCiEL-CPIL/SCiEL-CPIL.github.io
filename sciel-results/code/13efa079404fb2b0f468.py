"""Colorings with three colors in every four-term progression."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 128
    
    # Precompute for each position i, the list of (p0, p1, p2) such that (p0, p1, p2, i) is a 4-AP
    # p0 = i - 3d, p1 = i - 2d, p2 = i - d
    # d >= 1, p0 >= 0
    preds = [[] for _ in range(n)]
    for i in range(n):
        max_d = i // 3
        for d in range(1, max_d + 1):
            p2 = i - d
            p1 = i - 2 * d
            p0 = i - 3 * d
            if p0 >= 0:
                preds[i].append((p0, p1, p2))
    
    def is_valid_assignment(colors, i, c):
        """Check if assigning color c to position i is valid given colors[0..i-1]."""
        for p0, p1, p2 in preds[i]:
            c0 = colors[p0]
            c1 = colors[p1]
            c2 = colors[p2]
            
            # Determine distinct colors in {c0, c1, c2}
            # We need the 4-tuple (c0, c1, c2, c) to have >= 3 distinct colors.
            
            # Case 1: c0, c1, c2 all same
            if c0 == c1 and c1 == c2:
                # Then (c0, c0, c0, c) has distinct colors {c0, c} if c!=c0, or {c0} if c==c0.
                # Max distinct is 2. Always invalid.
                return False
            
            # Case 2: c0, c1, c2 have 2 distinct colors
            # Let S be the set of colors in {c0, c1, c2}. |S| = 2.
            # If c is in S, then the 4-tuple uses colors from S. Distinct <= 2. Invalid.
            # If c not in S, then distinct = 3. Valid.
            
            # Case 3: c0, c1, c2 have 3 distinct colors
            # Then S has size 3.
            # If c in S, distinct = 3. Valid.
            # If c not in S, distinct = 4. Valid.
            # So always valid.
            
            # Implementation:
            # Count distinct in {c0, c1, c2}
            # If 1: return False
            # If 2: return c not in {c0, c1, c2}
            # If 3: return True
            
            # Optimization:
            if c0 == c1:
                if c1 == c2:
                    return False # 1 distinct
                else:
                    # c0==c1 != c2. S = {c0, c2}
                    if c == c0 or c == c2:
                        return False
                    else:
                        continue
            elif c0 == c2:
                # c0==c2 != c1. S = {c0, c1}
                if c == c0 or c == c1:
                    return False
                else:
                    continue
            elif c1 == c2:
                # c1==c2 != c0. S = {c0, c1}
                if c == c0 or c == c1:
                    return False
                else:
                    continue
            else:
                # c0, c1, c2 all distinct. S size 3. Always valid.
                continue
                
        return True

    def solve(k):
        """Try to find a valid coloring with k colors. Returns list of colors or None."""
        colors = [-1] * n
        rng = random.Random(42 + k)
        
        # Backtracking search
        # We use a recursive function, but Python recursion limit might be an issue if depth is 128?
        # 128 is fine.
        
        def backtrack(i):
            if time.time() > deadline:
                return False
            
            if i == n:
                return True
            
            # Try colors in random order to break symmetry
            color_order = list(range(k))
            rng.shuffle(color_order)
            
            for c in color_order:
                if is_valid_assignment(colors, i, c):
                    colors[i] = c
                    if backtrack(i + 1):
                        return True
                    colors[i] = -1
            return False
        
        return backtrack(0)

    # Try increasing number of colors
    for k in range(3, 12):
        if time.time() > deadline:
            break
        
        # Run multiple attempts with different random seeds/shuffles if needed?
        # The shuffle in backtrack handles this.
        # However, backtracking can get stuck in bad paths.
        # We can add a timeout per k or a node limit.
        
        # Let's run the solver.
        # To improve success rate, we might want to randomize the initial choices more aggressively
        # or use a heuristic.
        
        # For now, just run it.
        found = solve(k)
        if found:
            # Verify the solution
            valid = True
            for i in range(n):
                for p0, p1, p2 in preds[i]:
                    c0 = colors[p0]
                    c1 = colors[p1]
                    c2 = colors[p2]
                    c3 = colors[i]
                    distinct = len({c0, c1, c2, c3})
                    if distinct < 3:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return colors
    
    # Fallback: If we didn't find a solution with <= 11 colors, 
    # we should return a valid solution with more colors if possible, 
    # or the best we found.
    # But the problem requires a valid solution.
    # If we ran out of time, we might return an invalid one? 
    # "An invalid result... scores 0".
    # We must ensure we return a VALID result.
    
    # If we failed to find a solution with k <= 11 within time, 
    # we can try a larger k which is easier to solve (fewer constraints per color choice? No, more colors = easier to satisfy "not in S").
    # Actually, larger k makes it EASIER to find a valid color because the probability of c being in S (size 2) is lower.
    # So if k=3 fails, k=4 is more likely to succeed quickly.
    
    # Let's try a few more k values if we have time.
    for k in range(12, 20):
        if time.time() > deadline:
            break
        found = solve(k)
        if found:
            valid = True
            for i in range(n):
                for p0, p1, p2 in preds[i]:
                    c0 = colors[p0]
                    c1 = colors[p1]
                    c2 = colors[p2]
                    c3 = colors[i]
                    distinct = len({c0, c1, c2, c3})
                    if distinct < 3:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return colors

    # Emergency fallback: Construct a valid coloring using many colors.
    # If we use 128 distinct colors, every 4-AP has 4 distinct colors. This is always valid.
    # But we must return it before timeout.
    # This is a valid solution, though not minimal.
    return list(range(n))

