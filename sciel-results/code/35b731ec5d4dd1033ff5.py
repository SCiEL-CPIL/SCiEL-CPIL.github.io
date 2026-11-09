import time
import numpy as np
import random

# Precompute all nonconstant 4-term progressions
_ALL_APS = []
for a in range(1, 125):
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

# For each position, precompute the APs it participates in
_pos_aps = [[] for _ in range(128)]
for _idx, _ap in enumerate(_ALL_APS):
    for _i in _ap:
        _pos_aps[_i].append(_idx)

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    n = 128

    # Fallback: Identity mapping is always valid (4 distinct colors per AP)
    fallback = list(range(n))

    def try_backtrack(k_colors):
        """Backtracking with MRV and forward checking."""
        if time.time() > DEADLINE:
            return None
        
        colors = [-1] * n
        # For each AP, track: assigned color mask (bitmask), unassigned count
        ap_mask = [0] * len(_ALL_APS)
        ap_unassigned = [4] * len(_ALL_APS)
        
        # Symmetry breaking: position 0 gets color 0
        colors[0] = 0
        for ap_idx in _pos_aps[0]:
            ap_mask[ap_idx] |= 1  # color 0
            ap_unassigned[ap_idx] -= 1
        
        solution_found = [False]
        
        def backtrack(unassigned_count):
            if time.time() > DEADLINE:
                return False
            if solution_found[0]:
                return True
            
            if unassigned_count == 0:
                # Final verification
                for ap_idx in range(len(_ALL_APS)):
                    if bin(ap_mask[ap_idx]).count('1') < 3:
                        return False
                solution_found[0] = True
                return True
            
            # MRV: find unassigned position with fewest valid colors
            best_pos = -1
            best_allowed = []
            min_count = k_colors + 1
            
            # Iterate through unassigned positions
            for pos in range(1, n):
                if colors[pos] != -1:
                    continue
                
                allowed = []
                for c in range(k_colors):
                    c_mask = 1 << c
                    valid = True
                    for ap_idx in _pos_aps[pos]:
                        # Check if adding this color makes the AP impossible to satisfy
                        current_mask = ap_mask[ap_idx]
                        new_mask = current_mask | c_mask
                        distinct_so_far = bin(new_mask).count('1')
                        remaining_unassigned = ap_unassigned[ap_idx] - 1
                        
                        # Max distinct we can achieve is distinct_so_far + remaining_unassigned
                        if distinct_so_far + remaining_unassigned < 3:
                            valid = False
                            break
                    if valid:
                        allowed.append(c)
                
                if len(allowed) < min_count:
                    min_count = len(allowed)
                    best_pos = pos
                    best_allowed = allowed
                    if min_count == 0:
                        return False
                    if min_count == 1:
                        break
            
            if best_pos == -1:
                return False
            
            # Try each allowed color
            for c in best_allowed:
                c_mask = 1 << c
                colors[best_pos] = c
                
                # Update AP state
                affected_aps = []
                for ap_idx in _pos_aps[best_pos]:
                    old_mask = ap_mask[ap_idx]
                    old_unassigned = ap_unassigned[ap_idx]
                    ap_mask[ap_idx] |= c_mask
                    ap_unassigned[ap_idx] -= 1
                    affected_aps.append((ap_idx, old_mask, old_unassigned))
                
                # Check if any AP is now unsatisfiable
                ok = True
                for ap_idx, _, _ in affected_aps:
                    distinct = bin(ap_mask[ap_idx]).count('1')
                    unassigned = ap_unassigned[ap_idx]
                    if distinct + unassigned < 3:
                        ok = False
                        break
                
                if ok and backtrack(unassigned_count - 1):
                    return True
                
                # Undo
                colors[best_pos] = -1
                for ap_idx, old_mask, old_unassigned in affected_aps:
                    ap_mask[ap_idx] = old_mask
                    ap_unassigned[ap_idx] = old_unassigned
            
            return False
        
        if backtrack(n - 1):
            return colors[:]
        return None

    # Try 3 colors
    result = try_backtrack(3)
    if result is not None:
        return result

    # Try 4 colors
    if time.time() < DEADLINE - 5:
        result = try_backtrack(4)
        if result is not None:
            return result

    # Try 5 colors
    if time.time() < DEADLINE - 2:
        result = try_backtrack(5)
        if result is not None:
            return result

    # Randomized fallback
    for attempt in range(50):
        if time.time() > DEADLINE - 1:
            break
        random.seed(attempt)
        c = [random.randint(0, 9) for _ in range(n)]
        valid = True
        for ap in _ALL_APS:
            if len(set(c[i] for i in ap)) < 3:
                valid = False
                break
        if valid:
            return c

    return fallback
