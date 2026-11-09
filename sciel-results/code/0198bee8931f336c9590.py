"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions (i, i+d, i+2d, i+3d) within 0..n-1
    progressions = []
    for i in range(n):
        max_d = (n - 1 - i) // 3
        for d in range(1, max_d + 1):
            progressions.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # For each position, store the indices of progressions it is part of
    pos_progs = [[] for _ in range(n)]
    for pi, prog in enumerate(progressions):
        for pos in prog:
            pos_progs[pos].append(pi)

    def is_valid_4ap(c1, c2, c3, c4):
        """Check if a 4-AP uses at least 3 distinct colors."""
        if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
            return True # Partial assignment, assume valid for now (checked later or via propagation)
        return len({c1, c2, c3, c4}) >= 3

    def check_propagation(pos, color, colors):
        """
        Check if assigning 'color' to 'pos' creates an immediate contradiction
        in any progression containing 'pos'.
        Returns False if contradiction, True otherwise.
        """
        for pi in pos_progs[pos]:
            prog = progressions[pi]
            # Get the other 3 positions
            others = [p for p in prog if p != pos]
            
            # Get current colors of others
            oc1, oc2, oc3 = colors[others[0]], colors[others[1]], colors[others[2]]
            
            # If any other is unassigned, we can't do a full check yet, 
            # but we can check if the assigned ones already force a contradiction.
            # However, simple forward checking: if all 4 are assigned, check validity.
            if oc1 != -1 and oc2 != -1 and oc3 != -1:
                if not is_valid_4ap(color, oc1, oc2, oc3):
                    return False
            
            # Additional pruning: 
            # If 3 positions are assigned and identical, the 4th CANNOT make it valid.
            # e.g., 0, 0, 0, x -> {0, x} size <= 2. Always invalid.
            # So if any 3 are same, it's a dead end.
            # If 3 are assigned:
            #   If all 3 same: invalid.
            #   If 2 same, 1 diff: valid only if 4th is the 3rd color.
            #   If all 3 diff: valid for any 4th.
            
            # Let's implement this stronger check
            assigned_others = [oc1, oc2, oc3]
            if all(c != -1 for c in assigned_others):
                # Check if the set of 4 colors is valid
                if not is_valid_4ap(color, oc1, oc2, oc3):
                    return False
            
            # Check for "3 identical" in the 4 positions (including the one we're assigning)
            # This is covered by is_valid_4ap if all are assigned.
            # But what if only 2 are assigned?
            # We can't prune much unless we track domains. 
            # For simplicity and speed, we rely on the "all assigned" check 
            # and the fact that we assign in order. 
            # Actually, MRV heuristic will help.
            
        return True

    def try_solve(num_colors):
        """
        Backtracking solver with forward checking and MRV heuristic.
        """
        colors = [-1] * n
        # Domains for each position
        domains = [set(range(num_colors)) for _ in range(n)]
        
        # Symmetry breaking: fix first few positions
        # colors[0] = 0
        # colors[1] = 1 (if num_colors >= 2)
        # colors[2] = 2 (if num_colors >= 3)
        
        # Let's just fix color 0 for position 0 to break symmetry
        colors[0] = 0
        domains[0] = {0}
        
        # Precompute constraints for faster checking?
        # The pos_progs list is sufficient.
        
        def get_mrv_var():
            """Find unassigned variable with minimum remaining values."""
            min_size = num_colors + 1
            min_var = -1
            for i in range(n):
                if colors[i] == -1:
                    size = len(domains[i])
                    if size < min_size:
                        min_size = size
                        min_var = i
                        if size == 1: # Forced
                            return i
            return min_var

        def backtrack(depth):
            if time.time() > deadline - 1:
                return False
            
            if depth == n:
                return True
            
            # Select variable with MRV
            var = get_mrv_var()
            if var == -1:
                return True # All assigned
            
            # Try values in domain
            # Order values by least constraining? For now, just iterate.
            for color in domains[var]:
                # Check if assignment is valid
                if check_propagation(var, color, colors):
                    colors[var] = color
                    domains[var] = {color}
                    
                    # Propagate: Remove invalid values from neighbors' domains?
                    # This is full arc consistency (AC-3), which might be too slow.
                    # Let's stick to simple forward checking for now.
                    
                    if backtrack(depth + 1):
                        return True
                    
                    colors[var] = -1
                    domains[var] = set(range(num_colors))
            
            return False
        
        return backtrack(0)

    # Try to find a solution with 3 colors
    # Note: 2 colors is impossible as proved earlier.
    # So we start with 3.
    
    # To speed up, we can use a simple greedy initialization to guide the backtracker
    # or just let the backtracker run. With N=128, it might be slow if the constraint graph is dense.
    # Let's try a hybrid: Greedy assignment, then local search to fix conflicts, 
    # but restricted to 3 colors.
    
    def greedy_3_color():
        """
        Greedy assignment trying to use only 3 colors.
        """
        colors = [-1] * n
        for i in range(n):
            # Try colors 0, 1, 2
            found = False
            for c in range(3):
                # Check all progressions containing i
                valid = True
                for pi in pos_progs[i]:
                    prog = progressions[pi]
                    others = [p for p in prog if p != i]
                    # If any other is unassigned, we can't fully check, 
                    # but greedy usually assigns left-to-right so others might be unassigned.
                    # We only check if all others are assigned.
                    all_assigned = all(colors[o] != -1 for o in others)
                    if all_assigned:
                        oc1, oc2, oc3 = colors[others[0]], colors[others[1]], colors[others[2]]
                        if not is_valid_4ap(c, oc1, oc2, oc3):
                            valid = False
                            break
                if valid:
                    colors[i] = c
                    found = True
                    break
            if not found:
                return None
        return colors

    # First, try greedy 3-color
    greedy_sol = greedy_3_color()
    if greedy_sol:
        # Verify it
        valid = True
        for prog in progressions:
            if len({greedy_sol[prog[0]], greedy_sol[prog[1]], greedy_sol[prog[2]], greedy_sol[prog[3]]}) < 3:
                valid = False
                break
        if valid:
            return greedy_sol

    # If greedy failed, try backtracking with 3 colors
    # Add a time check
    if time.time() < deadline - 10:
        # We need a slightly smarter backtracker or just hope it's fast enough
        # For N=128, plain backtracking might be too slow without AC-3.
        # Let's try a limited backtracking.
        
        # Actually, let's try to construct a valid 3-coloring using a more structured approach.
        # The constraint "no 4-AP uses < 3 colors" is equivalent to "no 4-AP is monochromatic or uses only 2 colors".
        
        # Let's try a simple local search starting from random 3-coloring.
        def local_search_3_color():
            colors = [random.randint(0, 2) for _ in range(n)]
            for _ in range(10000):
                if time.time() > deadline - 5:
                    break
                # Find a violating progression
                violated = None
                for prog in progressions:
                    if len({colors[prog[0]], colors[prog[1]], colors[prog[2]], colors[prog[3]]}) < 3:
                        violated = prog
                        break
                
                if not violated:
                    return colors
                
                # Pick a random position in the violated progression and change its color
                # to one that might fix it.
                pos_idx = random.choice(violated)
                current_c = colors[pos_idx]
                # Try other colors
                for new_c in range(3):
                    if new_c == current_c:
                        continue
                    colors[pos_idx] = new_c
                    # Check if this specific progression is now fixed
                    if len({colors[prog[0]], colors[prog[1]], colors[prog[2]], colors[prog[3]]}) >= 3:
                        # Check if we broke anything else? 
                        # Simple hill climbing: accept if total violations don't increase?
                        # For simplicity, just accept and continue.
                        break
                # If no change made it better locally, revert? 
                # Or just keep it.
            return None

        # Run a few random restarts of local search
        random.seed(42)
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            sol = local_search_3_color()
            if sol:
                # Final verification
                valid = True
                for prog in progressions:
                    if len({sol[prog[0]], sol[prog[1]], sol[prog[2]], sol[prog[3]]}) < 3:
                        valid = False
                        break
                if valid:
                    return sol

    # Fallback: Use the original greedy approach which uses many colors but is guaranteed to work
    # and is fast.
    colors = []
    for x in range(1, 129):
        # x is 1-indexed position, so index is x-1
        # We need to find a color for index x-1
        # The seed code logic:
        # for color in range(128):
        #   if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
        #     and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
        #      colors.append(color);break
        
        # Let's reuse the seed's greedy logic which is robust
        found = False
        for color in range(128):
            valid = True
            # Check all 4-APs ending at or before x-1 that include x-1?
            # The seed checks progressions where x-1 is the LAST term?
            # No, the seed checks:
            # d in range(1, (x-1)//3 + 1)
            # j in (1,2,3)
            # indices: x-1-j*d.
            # If x-1 is the start (i), then i+d, i+2d, i+3d are future.
            # If x-1 is the end (i+3d), then i = x-1-3d.
            # The seed's condition `colors[x-1-j*d]` implies it's looking at previous positions.
            # This suggests it's checking 4-APs where the current position is the LAST one?
            # Or is it checking all?
            # `all(... for d in range(1, (x-1)//3+1))`
            # If x=4 (index 3), d=1. j=1,2,3.
            # indices: 3-1=2, 3-2=1, 3-3=0.
            # So it checks (0,1,2,3).
            # This only checks 4-APs where the current position is the LAST element.
            # What about 4-APs where the current position is the 1st, 2nd, or 3rd?
            # Those will be checked when the LAST element of that progression is assigned.
            # So this greedy approach is valid: it ensures that when a progression is "completed"
            # (last element assigned), it is valid.
            
            for d in range(1, (x - 1) // 3 + 1):
                # Check progression ending at x-1 with step d
                # Start: x-1-3d
                i = x - 1 - 3 * d
                if i < 0:
                    continue
                c1 = colors[i]
                c2 = colors[i + d]
                c3 = colors[i + 2 * d]
                c4 = color
                if len({c1, c2, c3, c4}) < 3:
                    valid = False
                    break
            
            # The seed also had a second condition:
            # `and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))`
            # This seems to check 3-term progressions?
            # If we have a 3-term AP (a, b, c) and we assign c, do we care?
            # The constraint is on 4-term APs.
            # However, if we have a 3-term AP (a, b, c) with colors A, A, A,
            # then any 4th term d would make it (A, A, A, D) which has size 2. Invalid.
            # So, we must avoid creating a monochromatic 3-term AP?
            # Not necessarily. If we have A, A, A, we can't extend it to a valid 4-AP.
            # But maybe that 3-AP is not part of any 4-AP?
            # If i, i+d, i+2d are assigned A, A, A.
            # If there exists a 4th term i+3d < 128, then it's invalid.
            # The greedy check above only checks when the 4th term is assigned.
            # If we assign the 3rd term, we don't know if a 4th term will exist.
            # If i+3d < 128, then a 4-AP exists.
            # So, if we assign the 3rd term of a potential 4-AP, and the first 2 are same,
            # and we assign the same color, we create a "dead end" for the 4th term.
            # The 4th term would have to be a new color, but if we are limited to 3 colors,
            # and the first 2 are 0,0, and 3rd is 0, the 4th MUST be 1 or 2.
            # But {0,0,0,1} has size 2. Invalid.
            # So, we CANNOT have three identical colors in a 4-AP prefix.
            # Therefore, when assigning position x-1, we must ensure we don't complete
            # a 3-term AP with all same colors IF a 4th term exists.
            
            # Let's check this condition:
            # For any d, if x-1 is the 3rd term (i+2d), i.e., i = x-1-2d.
            # Then i+3d = x-1+d. If x-1+d < 128, a 4-AP exists.
            # If colors[i] == colors[i+d] == color, then it's invalid.
            
            for d in range(1, (x - 1) // 2 + 1):
                i = x - 1 - 2 * d
                if i < 0:
                    continue
                if i + 3 * d < 128: # 4th term exists
                    if colors[i] == colors[i + d] and colors[i] == color:
                        valid = False
                        break
                        # Also if colors[i] == colors[i+d] != color?
                        # Then we have A, A, B. 4th term C.
                        # {A, A, B, C}. If C is A or B, size 2. If C is new, size 3.
                        # So we MUST assign a new color C.
                        # If we are limited to 3 colors, and A, B are used, C is the 3rd.
                        # So it's possible.
                        # But if A, B are 0, 1, and we assign 1 to 3rd term?
                        # 0, 0, 1. 4th term must be 2.
                        # If we assign 0 to 3rd term? 0, 0, 0. Dead end.
                        # So the check `colors[i] == colors[i+d] == color` is correct for dead ends.
            
            if valid:
                colors.append(color)
                found = True
                break
        
        if not found:
            # Should not happen with 128 colors
            colors.append(0)

    return colors

# EVOLVE-BLOCK-END
