import time
import numpy as np
from itertools import product
from collections import defaultdict

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d), 1-indexed a,d
# Convert to 0-indexed internally.
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

# Build adjacency list for propagation
# For each index, store the list of APs it is part of, and for each AP, the other 3 indices
_pos_aps = defaultdict(list)
for ap in _ALL_APS:
    for i in ap:
        others = [j for j in ap if j != i]
        _pos_aps[i].append((ap, others))

def _is_satisfied_ap(ap, colors):
    # colors is a list, -1 for unassigned
    present = set()
    for i in ap:
        c = colors[i]
        if c != -1:
            present.add(c)
    # If 3 or more assigned and all same, violation? No, we need >=3 distinct.
    # If 3 assigned and they are {0, 0, 0}, the 4th MUST be non-0.
    # If 3 assigned and they are {0, 1, 1}, the 4th can be anything? 
    #   If 4th is 0, set is {0,1}. Size 2. Violation.
    #   If 4th is 1, set is {0,1}. Size 2. Violation.
    #   If 4th is 2, set is {0,1,2}. Size 3. OK.
    # So the constraint is not just "don't make it 2 colors if it's already 2".
    # It's "ensure final set size >= 3".
    
    if len(present) >= 3:
        return True # Already satisfied, no restriction on unassigned
    # If len(present) < 3, we need to ensure it's possible to reach 3.
    # This is handled by domain filtering.
    return True # "Not yet violated" in the sense of definite failure, but we need to check domains.

def _filter_domain(domain, ap, colors, pos):
    """
    Filter the domain of 'pos' given the current colors in 'ap' (excluding pos).
    Returns a set of allowed colors for pos.
    """
    # Get colors of the other 3 positions
    others_colors = []
    unassigned_count = 0
    for i in ap:
        if i == pos:
            continue
        c = colors[i]
        if c == -1:
            unassigned_count += 1
        else:
            others_colors.append(c)
            
    # We want to keep color c in domain if there EXISTS an assignment for the unassigned positions
    # such that the total distinct colors in the AP is >= 3.
    
    # Let S be the set of colors in others_colors.
    S = set(others_colors)
    
    allowed = set()
    for c in domain:
        # If we pick c, the current set of colors in the AP (including pos) is S ∪ {c}.
        current_set = S | {c}
        
        # We have 'unassigned_count' positions left.
        # Can we assign them to reach size >= 3?
        # If len(current_set) >= 3, yes, we can assign anything (even same colors) to the rest.
        if len(current_set) >= 3:
            allowed.add(c)
            continue
            
        # If len(current_set) < 3, we need the unassigned positions to introduce new colors.
        # Let missing = 3 - len(current_set).
        # We need at least 'missing' new distinct colors from the unassigned positions.
        # The unassigned positions can take any colors from the global palette (0..k-1).
        # So if unassigned_count >= missing, it is possible.
        # Wait, do the unassigned positions have to be distinct from each other? No, but they must provide the missing colors.
        # Since they are distinct positions, they CAN take distinct colors.
        # So the condition is simply: unassigned_count >= (3 - len(current_set))
        
        if unassigned_count >= (3 - len(current_set)):
            allowed.add(c)
            
    return allowed

def _find_periodic_pattern(period, k_colors, deadline):
    """
    Exhaustively search for a periodic pattern of given period and k_colors
    that satisfies the 3-color progression constraint for all x <= 128.
    Returns a list of 128 colors if found, else None.
    """
    if period > 12: # Too large for exhaustive search
        return None
        
    valid_patterns = []
    
    # Symmetry breaking: pattern[0]=0
    # If period >= 2, pattern[1]=1
    # If period >= 3, pattern[2]=2? No, that forces 3 colors.
    # We want to use AT MOST k_colors, but we want to MINIMIZE.
    # The periodic search tries specific k.
    
    # We iterate over k_colors starting from 3.
    # So in this function, we assume we are trying to use exactly k_colors?
    # No, we just need a valid pattern. The number of colors used will be determined by the pattern.
    # But we want to minimize colors. So we should try patterns with 3 colors first.
    # The caller loops k_colors = 3, 4...
    # So we should only generate patterns using colors 0..k_colors-1.
    
    # To ensure we don't miss patterns using fewer colors, we should allow patterns using subsets.
    # But if we are trying k=3, and find a pattern using only 2 colors, it's invalid anyway (needs 3 distinct in APs).
    # So any valid pattern MUST use at least 3 colors.
    # So we can just generate patterns using 0..k_colors-1.
    
    if period == 1:
        return None # Impossible
    
    # Generate patterns
    # To speed up, we can use backtracking on the pattern itself with the same constraint logic?
    # No, the pattern is short. Just iterate.
    
    # Optimization: Only check APs that fall within the first period?
    # No, the APs span the whole 128 range.
    # But color[i] = pattern[i % period].
    # So we just need to check all _ALL_APS.
    
    # Heuristic: Fix pattern[0]=0.
    # If k_colors >= 2, fix pattern[1]=1?
    # If k_colors >= 3, fix pattern[2]=2?
    # This breaks symmetry.
    
    fixed = {}
    if period > 0: fixed[0] = 0
    if period > 1 and k_colors > 1: fixed[1] = 1
    if period > 2 and k_colors > 2: fixed[2] = 2
    
    remaining = [i for i in range(period) if i not in fixed]
    
    def check_pattern(pattern):
        for ap in _ALL_APS:
            colors_in_ap = {pattern[i % period] for i in ap}
            if len(colors_in_ap) < 3:
                return False
        return True

    for rest in product(range(k_colors), repeat=len(remaining)):
        pattern = [0] * period
        for i, idx in enumerate(remaining):
            pattern[idx] = rest[i]
        for idx, val in fixed.items():
            pattern[idx] = val
            
        # Check if pattern uses <= k_colors (it does by construction)
        # Check validity
        if check_pattern(pattern):
            expanded = [pattern[i % period] for i in range(128)]
            return expanded
            
    return None

def _solve_backtracking(n, k_colors, deadline):
    """
    Backtracking search with MRV and Forward Checking.
    """
    colors = [-1] * n
    domains = [set(range(k_colors)) for _ in range(n)]
    
    # Symmetry breaking:
    # Fix color[0] = 0
    # Fix color[1] = 1 (if k>=2)
    # Fix color[2] = 2 (if k>=3)
    # This reduces the search space significantly.
    
    def assign(pos, c):
        colors[pos] = c
        domains[pos] = {c}
        
    def unassign(pos):
        colors[pos] = -1
        # Domain restoration is complex with forward checking. 
        # Better to use a stack of states or recompute domains.
        # Given n=128, recomputing domains for neighbors is fast.
        
    # We will use a recursive function that maintains the state.
    # To make it efficient, we'll copy the domains stack? No, too memory heavy.
    # We'll do "undo" by storing the old domain.
    
    # Actually, simpler:
    # The constraint is global.
    # Let's use a standard backtracking with a "trail" of changes.
    
    trail = []
    
    def propagate(pos, c):
        """
        Assign colors[pos]=c.
        Update domains of neighbors.
        Return False if any domain becomes empty.
        """
        # Record change
        trail.append((pos, colors[pos], domains[pos].copy()))
        assign(pos, c)
        
        for ap, others in _pos_aps[pos]:
            for other in others:
                if colors[other] == -1:
                    # Filter domain of 'other' based on 'pos' being assigned
                    # We need to filter based on the WHOLE AP context.
                    # So we filter domain[other] given the current colors of the other 3 in 'ap'.
                    new_domain = _filter_domain(domains[other], ap, colors, other)
                    if not new_domain:
                        return False
                    # If domain shrunk, record it?
                    # For simplicity, we just update. If we backtrack, we restore from trail.
                    # But 'other' might be in multiple APs. We must intersect constraints from ALL APs.
                    # _filter_domain only considers ONE ap.
                    # We need to intersect the filtered domain from all APs containing 'other'.
                    
        # The above loop is flawed. We need to recompute the full domain for 'other' considering ALL APs.
        # Let's do that.
        
        # Re-implementation of propagation:
        # For each neighbor, recompute allowed colors based on ALL APs it belongs to.
        
        for ap, others in _pos_aps[pos]:
            for other in others:
                if colors[other] == -1:
                    # Intersect constraints from all APs containing 'other'
                    final_domain = set(domains[other]) # Start with current domain
                    for ap2, others2 in _pos_aps[other]:
                        fd = _filter_domain(final_domain, ap2, colors, other)
                        final_domain = fd
                        if not final_domain:
                            return False
                    domains[other] = final_domain
                    # If domain changed, we might need to propagate further?
                    # This is AC-3. Let's keep it simple: just forward check (1-step).
                    # If we want AC-3, we'd need a queue.
                    
        return True

    # Wait, the `propagate` function above modifies `domains` but doesn't handle the "restore" logic correctly in the trail if we don't push every change.
    # Let's simplify: 
    # 1. Pick variable (MRV).
    # 2. Try each value.
    # 3. Assign.
    # 4. Check if any neighbor domain becomes empty.
    # 5. If OK, recurse.
    # 6. Unassign (restore domain from copy).
    
    # We need a way to pick the next variable.
    # MRV: unassigned var with smallest domain.
    
    def get_mrv_var():
        min_size = k_colors + 1
        var = -1
        for i in range(n):
            if colors[i] == -1:
                if len(domains[i]) < min_size:
                    min_size = len(domains[i])
                    var = i
                    if min_size == 1:
                        break
        return var

    def backtrack(depth):
        if time.time() > deadline:
            return False
        if depth == n:
            return True
            
        var = get_mrv_var()
        if var == -1:
            return True # All assigned
            
        # Try values in var's domain
        # Value ordering: try colors that appear most often in neighbors? Or just sorted.
        for c in sorted(domains[var]):
            # Save state
            old_colors_var = colors[var]
            old_domain_var = domains[var].copy()
            
            # Assign
            colors[var] = c
            domains[var] = {c}
            
            # Propagate: check neighbors
            valid = True
            # We need to save domains of neighbors that change to restore them.
            # Since we only do 1-step forward checking, we just check if any neighbor domain becomes empty.
            # We don't actually shrink the neighbor domains in this simple version?
            # No, we should shrink them to prune.
            
            # Let's do:
            # 1. For each neighbor, compute new domain.
            # 2. If empty, invalid.
            # 3. If shrunk, save old domain, update.
            
            neighbor_changes = []
            for ap, others in _pos_aps[var]:
                for other in others:
                    if colors[other] == -1:
                        # Compute intersection of constraints from all APs for 'other'
                        # This is expensive. 
                        # Optimization: Only recompute if 'other' is in an AP with 'var'.
                        # Actually, 'other' is in 'ap' which contains 'var'.
                        # So 'other's domain is constrained by 'ap' and potentially others.
                        
                        # Let's just check if the current domain of 'other' is still valid given 'var'=c.
                        # We can do this by checking if there EXISTS a value in domains[other] that is allowed.
                        # allowed = _filter_domain(domains[other], ap, colors, other)
                        # If allowed is empty, invalid.
                        
                        # But we also need to ensure consistency with OTHER ap's.
                        # So we must intersect.
                        
                        new_d = set(domains[other])
                        for ap2, others2 in _pos_aps[other]:
                            fd = _filter_domain(new_d, ap2, colors, other)
                            new_d = fd
                            if not new_d:
                                valid = False
                                break
                        if not valid:
                            break
                            
                        if new_d != domains[other]:
                            neighbor_changes.append((other, domains[other].copy()))
                            domains[other] = new_d
                            
                if not valid:
                    break
            
            if valid:
                if backtrack(depth + 1):
                    return True
            
            # Undo
            colors[var] = old_colors_var
            domains[var] = old_domain_var
            for other, old_d in neighbor_changes:
                domains[other] = old_d
                
        return False

    # Initialize with symmetry breaking
    # Fix 0, 1, 2
    if k_colors >= 3:
        if not propagate(0, 0): return False # Should not fail
        if not propagate(1, 1): return False
        if not propagate(2, 2): return False
        # Note: propagate above was flawed. Let's just manually set and check.
        
    # Let's restart with a cleaner implementation of the solver.
    return _solve_backtracking_clean(n, k_colors, deadline)

def _solve_backtracking_clean(n, k_colors, deadline):
    colors = [-1] * n
    domains = [set(range(k_colors)) for _ in range(n)]
    
    # Symmetry breaking
    # 0 -> 0
    # 1 -> 1
    # 2 -> 2
    # This assumes k>=3.
    
    def is_valid_assignment(pos, c):
        # Check if assigning c to pos violates any AP constraint immediately
        # i.e., if any AP containing pos becomes impossible to satisfy
        for ap, others in _pos_aps[pos]:
            # Check if this AP can still be satisfied
            # We need to check if there exists a completion.
            # Simple check:
            # If 3 are assigned and all same, 4th must be diff.
            # If 3 are assigned and 2 colors, 4th must be 3rd color.
            # This is complex. Use _filter_domain logic.
            
            # Actually, just check if c is in the allowed set for pos given the CURRENT state of others.
            # But 'others' might be unassigned.
            # _filter_domain handles unassigned.
            
            # We need to intersect constraints from all APs.
            allowed = set([c])
            for ap2, others2 in _pos_aps[pos]:
                fd = _filter_domain(allowed, ap2, colors, pos)
                allowed = fd
                if not allowed:
                    return False
            if not allowed:
                return False
        return True

    def backtrack(depth):
        if time.time() > deadline:
            return False
        if depth == n:
            return True
            
        # MRV
        var = -1
        min_size = k_colors + 1
        for i in range(n):
            if colors[i] == -1:
                if len(domains[i]) < min_size:
                    min_size = len(domains[i])
                    var = i
                    if min_size == 1:
                        break
        if var == -1:
            return True
            
        for c in sorted(domains[var]):
            if not is_valid_assignment(var, c):
                continue
                
            # Assign
            colors[var] = c
            domains[var] = {c}
            
            # Propagate to neighbors (Forward Checking)
            valid = True
            changes = []
            for ap, others in _pos_aps[var]:
                for other in others:
                    if colors[other] == -1:
                        # Recompute domain for 'other'
                        new_d = set(domains[other])
                        for ap2, others2 in _pos_aps[other]:
                            fd = _filter_domain(new_d, ap2, colors, other)
                            new_d = fd
                            if not new_d:
                                valid = False
                                break
                        if not valid:
                            break
                        if new_d != domains[other]:
                            changes.append((other, domains[other].copy()))
                            domains[other] = new_d
                if not valid:
                    break
            
            if valid:
                if backtrack(depth + 1):
                    return True
            
            # Undo
            colors[var] = -1
            domains[var] = set(range(k_colors)) # Restore full domain? No, we should restore the domain it had before.
            # We didn't save the old domain of 'var' specifically, but it was determined by previous assignments.
            # Since we are backtracking, the state of 'var' before this attempt was "unassigned".
            # Its domain was whatever it was.
            # We should save the old domain of 'var' before changing it.
            
            for other, old_d in changes:
                domains[other] = old_d
                
        return False

    # Init
    # Set 0,1,2
    # This is tricky with the clean version.
    # Let's just run backtrack(0) and let MRV pick.
    # Symmetry breaking is important.
    # We can force the first 3 assignments.
    
    # Force 0->0, 1->1, 2->2
    # Manually assign and propagate.
    
    # This is getting complicated to edit in place.
    # I will replace the whole `construct` and helper functions with a robust implementation.
    return None

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s

    n = 128
    ap_count = len(_ALL_APS)
    
    best = None
    best_ncolors = n

    # 1. Try periodic patterns
    for period in [3, 4, 5, 6, 7, 8, 9, 10, 12]:
        if time.time() > DEADLINE:
            break
        pattern = _find_periodic_pattern(period, 3, DEADLINE)
        if pattern is not None:
            best = pattern
            best_ncolors = 3
            break
            
    # 2. Try backtracking with increasing colors
    if best is None:
        for k_colors in range(3, 7):
            if time.time() > DEADLINE:
                break
            
            # Use the clean backtracker
            # Note: _solve_backtracking_clean is currently a stub returning None.
            # I need to implement it properly in the replacement.
            
            # For now, let's assume the replacement will contain a working solver.
            pass

    if best is not None:
        return best
    else:
        # Fallback
        colors = list(range(128))
        return colors
