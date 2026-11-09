"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30 present,
    minimizing the largest element.
    
    Strategy: Backtracking search with pruning.
    - Maintain sorted list of elements
    - Track set of all differences used
    - Track which of 1..30 are still needed
    - When adding a new element, compute all new differences
    - Prune if any difference collides with existing or creates duplicate
    - Prioritize elements that cover multiple missing required differences
    """
    
    REQUIRED = set(range(1, 31))
    
    def solve():
        best = None
        best_max = float('inf')
        
        # We'll search for sets of various sizes, starting from smaller
        # A Sidon set with n elements has n*(n-1)/2 differences
        # We need at least 30 differences, so n >= 8 (8*7/2=28 < 30, so n>=9: 9*8/2=36)
        
        def backtrack(elements, used_diffs, needed, max_val):
            nonlocal best, best_max
            
            # If we've covered all required differences, record if better
            if not needed:
                if max_val < best_max:
                    best_max = max_val
                    best = list(elements)
                return True  # Signal to stop exploring this branch further
            
            # Pruning: if current max already >= best_max, no point
            if max_val >= best_max:
                return False
            
            # Try adding new elements
            # Candidate positions: we want to try small values first to minimize max
            # But also consider positions that help cover needed differences
            
            current_max = max(elements) if elements else 0
            
            # Generate candidate new elements
            # Key insight: a new element x creates differences |x - e| for each e in elements
            # We want these to include some from 'needed' and not collide with used_diffs
            
            # Try candidates in a smart order
            candidates = set()
            
            # For each needed difference d, and each existing element e:
            # x = e + d or x = e - d (if >= 0)
            for d in needed:
                for e in elements:
                    x1 = e + d
                    x2 = e - d
                    if x1 >= 0:
                        candidates.add(x1)
                    if x2 >= 0:
                        candidates.add(x2)
            
            # Also try some small values near existing elements
            for e in elements:
                for offset in range(1, 20):
                    x1 = e + offset
                    x2 = e - offset
                    if x1 >= 0:
                        candidates.add(x1)
                    if x2 >= 0:
                        candidates.add(x2)
            
            # Sort candidates: prefer those that cover more needed differences
            def candidate_score(x):
                if x in elements:
                    return (-1, x)  # Skip duplicates, but keep in list to filter
                new_diffs = set()
                for e in elements:
                    d = abs(x - e)
                    if d == 0:
                        continue
                    new_diffs.add(d)
                # Count how many needed differences this covers
                covers_needed = len(new_diffs & needed)
                # Penalize if it creates differences that are in used_diffs
                conflicts = len(new_diffs & used_diffs)
                return (-covers_needed, conflicts, x)
            
            # Filter out invalid candidates and sort
            valid_candidates = []
            for x in candidates:
                if x in elements:
                    continue
                # Check if adding x would create any duplicate difference
                new_diffs = set()
                valid = True
                for e in elements:
                    d = abs(x - e)
                    if d == 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                # Also check for duplicates among new_diffs themselves
                # (can't happen since each e gives a unique d for fixed x... wait, no)
                # Actually for fixed x and different e, |x-e| can repeat? No, 
                # if |x-e1| = |x-e2| then e1 = e2 or e1 + e2 = 2x.
                # So we need to check that.
                if valid:
                    # Check for internal duplicates in new_diffs
                    seen = set()
                    for e in elements:
                        d = abs(x - e)
                        if d == 0:
                            continue
                        if d in seen:
                            valid = False
                            break
                        seen.add(d)
                if valid:
                    valid_candidates.append(x)
            
            valid_candidates.sort(key=candidate_score)
            
            # Limit candidates to keep search tractable
            # Try top candidates first
            for x in valid_candidates[:50]:  # Limit to top 50
                new_elements = sorted(elements + [x])
                new_used = set(used_diffs)
                new_needed = set(needed)
                valid = True
                for e in elements:
                    d = abs(x - e)
                    if d == 0:
                        continue
                    if d in new_used:
                        valid = False
                        break
                    new_used.add(d)
                    if d in new_needed:
                        new_needed.discard(d)
                
                if not valid:
                    continue
                
                if backtrack(new_elements, new_used, new_needed, max(max_val, x)):
                    # Found a solution, can stop
                    return True
            
            return False
        
        # Start with [0]
        initial_elements = [0]
        initial_used = set()
        initial_needed = set(REQUIRED)
        
        # We need to add elements. The first element we add will be some positive number.
        # Let's try different starting points
        
        # Actually, let's think about this more carefully.
        # With just {0}, we have no differences. We need to add elements to create differences 1-30.
        
        # Let's try a more structured approach:
        # Start with {0}, then systematically build up
        
        # Try different initial second elements
        for second in range(1, 100):
            elements = [0, second]
            used = {second}
            needed = set(REQUIRED)
            needed.discard(second)
            
            if not needed and second < best_max:
                best_max = second
                best = list(elements)
            
            if needed:
                # Continue backtracking
                max_val = second
                if backtrack(elements, used, needed, max_val):
                    if best is not None:
                        break
        
        if best is not None:
            return best
        
        # Fallback: use a known construction
        # Singer's construction for prime power q gives a perfect difference set of size q+1 in Z_{q^2+q+1}
        # For q=5: size 6 in Z_31, differences cover 1..30
        # But we need integers, not modular. We can embed.
        # Actually for q=5, the set {0, 1, 4, 14, 16, 25} mod 31 has all differences 1..30
        # Let's verify and use it if it works as integers (no wrap-around issues)
        
        # The Singer set for q=5: {0, 1, 4, 14, 16, 25}
        # Differences: let's compute
        singer = [0, 1, 4, 14, 16, 25]
        diffs = set()
        for i in range(len(singer)):
            for j in range(i+1, len(singer)):
                diffs.add(singer[j] - singer[i])
        
        if REQUIRED.issubset(diffs) and len(diffs) == len(singer)*(len(singer)-1)//2:
            # Check all differences are distinct
            if len(diffs) == 15:  # 6*5/2 = 15
                # This is a valid Sidon set with all 1..30? No, only 15 differences
                # Wait, we need 30 differences but only have 15. So this doesn't work directly.
                pass
        
        # We need more elements. Let's try a larger Singer set or hybrid.
        # For now, return a reasonable set
        # Let's just return a known good set
        
        # Actually, let me just do a more thorough search with the backtracker
        # starting from scratch
        
        # Reset and do a cleaner search
        best = None
        best_max = float('inf')
        
        def backtrack2(elements, used_diffs, needed):
            nonlocal best, best_max
            
            if not needed:
                max_val = max(elements)
                if max_val < best_max:
                    best_max = max_val
                    best = list(elements)
                return True
            
            max_val = max(elements)
            if max_val >= best_max:
                return False
            
            # Generate candidates
            candidates = set()
            for d in needed:
                for e in elements:
                    candidates.add(e + d)
                    if e - d >= 0:
                        candidates.add(e - d)
            
            # Add some heuristic candidates
            for e in elements:
                for delta in [1, 2, 3, 5, 7, 11, 13]:
                    if e + delta >= 0:
                        candidates.add(e + delta)
                    if e - delta >= 0:
                        candidates.add(e - delta)
            
            def score(x):
                if x in elements:
                    return float('inf')
                new_diffs = []
                for e in elements:
                    d = abs(x - e)
                    new_diffs.append(d)
                # Check for duplicates in new_diffs
                if len(new_diffs) != len(set(new_diffs)):
                    return float('inf')
                # Check conflicts with used
                for d in new_diffs:
                    if d in used_diffs:
                        return float('inf')
                covers = sum(1 for d in new_diffs if d in needed)
                return (-covers, max(new_diffs) if new_diffs else 0, x)
            
            valid_cands = []
            for x in candidates:
                s = score(x)
                if s != float('inf'):
                    valid_cands.append((s, x))
            
            valid_cands.sort()
            
            for _, x in valid_cands[:30]:
                new_elements = sorted(elements + [x])
                new_used = set(used_diffs)
                new_needed = set(needed)
                ok = True
                for e in elements:
                    d = abs(x - e)
                    if d in new_used:
                        ok = False
                        break
                    new_used.add(d)
                    if d in new_needed:
                        new_needed.discard(d)
                if ok:
                    if backtrack2(new_elements, new_used, new_needed):
                        return True
            
            return False
        
        # Start with {0} and try various second elements
        for second in range(1, 50):
            if best is not None and second >= best_max:
                break
            elements = [0, second]
            used = {second}
            needed = set(REQUIRED)
            needed.discard(second)
            
            if not needed:
                if second < best_max:
                    best_max = second
                    best = list(elements)
                continue
            
            backtrack2(elements, used, needed)
            if best is not None:
                break
        
        if best is not None:
            return best
        
        # Ultimate fallback: return a valid (if not optimal) set
        # Build greedily but more carefully
        a = [0]
        used = set()
        needed = set(REQUIRED)
        
        for _ in range(31):  # Add up to 31 more elements
            if not needed:
                break
            best_x = None
            best_score = None
            for x in range(1, 200):
                if x in a:
                    continue
                new_diffs = set()
                ok = True
                for e in a:
                    d = abs(x - e)
                    if d == 0:
                        continue
                    if d in used or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                covers = len(new_diffs & needed)
                if covers == 0:
                    continue
                s = (-covers, x)
                if best_score is None or s < best_score:
                    best_score = s
                    best_x = x
            if best_x is None:
                # Try larger x
                for x in range(200, 500):
                    if x in a:
                        continue
                    new_diffs = set()
                    ok = True
                    for e in a:
                        d = abs(x - e)
                        if d == 0:
                            continue
                        if d in used or d in new_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue
                    covers = len(new_diffs & needed)
                    if covers == 0:
                        continue
                    s = (-covers, x)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_x = x
            if best_x is not None:
                a.append(best_x)
                a.sort()
                for e in a[:-1]:
                    d = abs(best_x - e)
                    used.add(d)
                    needed.discard(d)
        
        return a
    
    result = solve()
    return result

# EVOLVE-BLOCK-END
