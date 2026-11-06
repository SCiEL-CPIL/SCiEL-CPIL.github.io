import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute constraints
    # For each position i, store the 4-term APs that end at i.
    # A 4-term AP is (a, a+d, a+2d, a+3d). If it ends at i, then a+3d=i.
    # The constraint is that {color[a], color[a+d], color[a+2d], color[i]} must have size >= 3.
    
    # We will use a CSP approach.
    # Variables: 0..n-1
    # Domain: 0..k-1
    
    # Precompute for each i, the list of (a, d) such that a+3d=i
    aps_ending = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 3 + 1):
            a = i - 3 * d
            if a >= 0:
                aps_ending[i].append((a, d))
                
    # Also precompute 3-term APs ending at i for lookahead/forward checking efficiency?
    # Actually, standard forward checking checks if assigning value v to variable i
    # leaves any unassigned variable j with an empty domain.
    # But here, constraints are not binary. They involve 4 variables.
    # However, we can check: if we assign color[i] = v, does it make any future AP impossible?
    # An AP (a, a+d, a+2d, a+3d) is "active" if some of its terms are assigned.
    # If 3 terms are assigned and the 4th is unassigned, the 4th is constrained.
    # If 3 terms are assigned and they use only 2 colors, the 4th MUST use a 3rd color.
    # If 3 terms are assigned and they use only 1 color, the 4th MUST use a 2nd color.
    # If 3 terms are assigned and they use 3 colors, the 4th can be anything.
    
    # Let's implement a recursive backtracking with MRV (Minimum Remaining Values) heuristic.
    
    def solve(num_colors):
        colors = [-1] * n
        # domains[i] is a set of possible colors for i
        domains = [set(range(num_colors)) for _ in range(n)]
        
        # To speed up MRV, we can maintain counts of remaining values.
        # But sets are fine for n=128.
        
        # We need a way to quickly check if a variable's domain is empty.
        # And to find the variable with the smallest domain.
        
        # Let's use a simpler forward checking:
        # When we assign colors[i] = v:
        # 1. Remove v from domains of neighbors? No, constraints are 4-way.
        # 2. Check all APs involving i.
        #    For each AP (a, b, c, d) involving i:
        #    If 3 are assigned, check if the 4th has a valid color.
        #    If 2 are assigned, check if the remaining 2 have any valid combination?
        #    This is expensive.
        
        # Alternative: Since n=128 is small, we can just check validity of the partial assignment
        # and use MRV to pick the next variable.
        
        # Let's optimize the check.
        # Precompute for each i, the list of APs where i is the 4th term (already done: aps_ending[i]).
        # Also need APs where i is the 1st, 2nd, or 3rd term.
        # Let's precompute all APs and map them to their indices.
        
        all_aps = []
        for a in range(n):
            for d in range(1, (n - a - 1) // 3 + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if e < n:
                    all_aps.append((a, b, c, e))
                    
        # Map index -> list of AP indices
        idx_to_aps = [[] for _ in range(n)]
        for ap_idx, ap in enumerate(all_aps):
            for idx in ap:
                idx_to_aps[idx].append(ap_idx)
                
        # Backtracking function
        # state: colors array, domains array
        # We will use a global counter for time check.
        
        def is_consistent_partial():
            # Check all APs. If an AP has all 4 assigned, check constraint.
            # If an AP has 3 assigned, check if the 4th has a valid color in its domain.
            for ap in all_aps:
                a, b, c, e = ap
                assigned = []
                unassigned_idx = -1
                for idx in ap:
                    if colors[idx] != -1:
                        assigned.append(colors[idx])
                    else:
                        unassigned_idx = idx
                        
                if len(assigned) == 4:
                    if len(set(assigned)) < 3:
                        return False
                elif len(assigned) == 3:
                    # The unassigned index must have a color that makes the set size >= 3
                    s = set(assigned)
                    if len(s) == 1:
                        # Need a color different from the one assigned
                        # domains[unassigned_idx] must contain a color not in s
                        if not (domains[unassigned_idx] - s):
                            return False
                    elif len(s) == 2:
                        # Need a color different from the two assigned
                        # domains[unassigned_idx] must contain a color not in s
                        if not (domains[unassigned_idx] - s):
                            return False
                    else: # len(s) == 3
                        # Any color works
                        pass
                        
            return True
            
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find unassigned variable with smallest domain (MRV)
            min_size = num_colors + 1
            min_var = -1
            for i in range(n):
                if colors[i] == -1:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        min_var = i
                        if min_size == 0:
                            return False # Fail fast
                            
            if min_var == -1:
                return True # All assigned
            
            # Try each color in domains[min_var]
            # Heuristic: Try colors that are "popular" or "least constraining"?
            # For now, just iterate.
            for c in list(domains[min_var]):
                # Assign
                colors[min_var] = c
                old_domain = domains[min_var].copy()
                domains[min_var] = {c}
                
                # Forward checking
                if is_consistent_partial():
                    if backtrack():
                        return True
                
                # Undo
                colors[min_var] = -1
                domains[min_var] = old_domain
                
            return False
            
        return backtrack()
        
    # Try increasing number of colors
    for k in [3, 4, 5, 6]:
        if solve(k):
            return colors
    
    # Fallback: return all distinct colors
    return list(range(n))

# The above implementation might be too slow in pure Python due to the overhead of sets and list copies.
# Let's optimize or use a different approach if needed.
# Given the 90s limit, and n=128, a pure Python backtracker might struggle if the search space is large.
# However, the constraint is quite strong.
# Let's add a greedy pre-fill to reduce the search space.

# Revised Strategy:
# 1. Greedy fill with k colors.
# 2. If fails, use Backtracking with MRV on the remaining unassigned or all variables.

# Let's refine the backtracking to be faster.
# Instead of copying domains, we can use a stack or just recompute MRV.
# Recomputing MRV is O(n*k). n=128, k=6. 128*6 = 768 ops. Very fast.
# is_consistent_partial is O(NumAPs). NumAPs ~ 128*128/6 ~ 2000.
# 2000 ops per step.
# If depth is 128, and branching factor is small, it should be okay.
# But if branching factor is high, it explodes.

# Let's try to improve the heuristic:
# When assigning, choose the color that leaves the largest minimum domain for neighbors?
# That's expensive.
# Just MRV is usually good.

# Let's stick with the CSP approach but ensure it's optimized.
# I will implement a cleaner version.

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute APs
    all_aps = []
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                all_aps.append((a, b, c, e))
                
    # Map index -> list of AP indices
    idx_to_aps = [[] for _ in range(n)]
    for ap_idx, ap in enumerate(all_aps):
        for idx in ap:
            idx_to_aps[idx].append(ap_idx)
            
    def solve(k):
        colors = [-1] * n
        # domains[i] is a bitmask or list. List of available colors.
        # Using lists for easy iteration.
        domains = [list(range(k)) for _ in range(n)]
        
        # Count of unassigned
        unassigned_count = n
        
        def check_consistency():
            for ap in all_aps:
                a, b, c, e = ap
                assigned_vals = []
                unassigned_idx = -1
                for idx in ap:
                    if colors[idx] != -1:
                        assigned_vals.append(colors[idx])
                    else:
                        unassigned_idx = idx
                        
                if len(assigned_vals) == 4:
                    if len(set(assigned_vals)) < 3:
                        return False
                elif len(assigned_vals) == 3:
                    s = set(assigned_vals)
                    # The unassigned variable must have a color not in s if |s| < 3
                    if len(s) < 3:
                        # Check if domains[unassigned_idx] has any color not in s
                        has_valid = False
                        for col in domains[unassigned_idx]:
                            if col not in s:
                                has_valid = True
                                break
                        if not has_valid:
                            return False
            return True

        def backtrack():
            nonlocal unassigned_count
            if time.time() > deadline:
                return False
                
            if unassigned_count == 0:
                return True
            
            # MRV: Find unassigned variable with smallest domain
            min_size = k + 1
            min_var = -1
            for i in range(n):
                if colors[i] == -1:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        min_var = i
                        if min_size == 0:
                            return False
            
            if min_var == -1:
                return True # Should not happen if unassigned_count > 0
            
            # Try colors
            for c in domains[min_var]:
                # Assign
                colors[min_var] = c
                unassigned_count -= 1
                
                # Update domains of min_var to just {c} for consistency check
                old_domain = domains[min_var]
                domains[min_var] = [c]
                
                if check_consistency():
                    if backtrack():
                        return True
                
                # Undo
                colors[min_var] = -1
                unassigned_count += 1
                domains[min_var] = old_domain
                
            return False
            
        return backtrack()
        
    # Try k=3 first
    for k in [3, 4, 5, 6]:
        if solve(k):
            return colors
            
    return list(range(n))

# To be safe against the 90s limit, I'll add a timeout check inside the loop.
# The current implementation checks time at the start of backtrack.
# This should be sufficient.

# One potential issue: The `check_consistency` function iterates over ALL APs every time.
# This is O(M) where M is number of APs.
# M is roughly n^2/6 = 2730.
# 2730 checks per node.
# If the tree is deep, this might be slow in Python.
# Optimization: Only check APs affected by the last assignment.
# When we assign colors[min_var], only APs containing min_var are affected.
# Let's modify check_consistency to take a list of APs to check.

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute APs
    all_aps = []
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                all_aps.append((a, b, c, e))
                
    # Map index -> list of AP indices
    idx_to_aps = [[] for _ in range(n)]
    for ap_idx, ap in enumerate(all_aps):
        for idx in ap:
            idx_to_aps[idx].append(ap_idx)
            
    def solve(k):
        colors = [-1] * n
        domains = [list(range(k)) for _ in range(n)]
        unassigned_count = n
        
        def check_aps(aps_to_check):
            for ap_idx in aps_to_check:
                a, b, c, e = all_aps[ap_idx]
                assigned_vals = []
                unassigned_idx = -1
                for idx in (a, b, c, e):
                    if colors[idx] != -1:
                        assigned_vals.append(colors[idx])
                    else:
                        unassigned_idx = idx
                        
                if len(assigned_vals) == 4:
                    if len(set(assigned_vals)) < 3:
                        return False
                elif len(assigned_vals) == 3:
                    s = set(assigned_vals)
                    if len(s) < 3:
                        has_valid = False
                        for col in domains[unassigned_idx]:
                            if col not in s:
                                has_valid = True
                                break
                        if not has_valid:
                            return False
            return True

        def backtrack():
            nonlocal unassigned_count
            if time.time() > deadline:
                return False
                
            if unassigned_count == 0:
                return True
            
            # MRV
            min_size = k + 1
            min_var = -1
            for i in range(n):
                if colors[i] == -1:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        min_var = i
                        if min_size == 0:
                            return False
            
            if min_var == -1:
                return True
            
            # Get APs affected by min_var
            affected_aps = idx_to_aps[min_var]
            
            for c in domains[min_var]:
                colors[min_var] = c
                unassigned_count -= 1
                old_domain = domains[min_var]
                domains[min_var] = [c]
                
                if check_aps(affected_aps):
                    if backtrack():
                        return True
                
                colors[min_var] = -1
                unassigned_count += 1
                domains[min_var] = old_domain
                
            return False
            
        return backtrack()
        
    for k in [3, 4, 5, 6]:
        if solve(k):
            return colors
            
    return list(range(n))
