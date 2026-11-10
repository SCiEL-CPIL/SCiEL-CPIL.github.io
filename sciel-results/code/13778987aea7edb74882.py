import numpy as np
import time
import random
import sys

_p = 7
_kq = 3
_Q = 49
_SLOTS = 2401
_INF = 10  # vertical direction index

def _encode(xa, xb, ya, yb):
    return ((xa * 7 + xb) * 49) + (ya * 7 + yb)

def _xy_from_slot(s):
    yb = s % 7; s //= 7
    ya = s % 7; s //= 7
    xb = s % 7; s //= 7
    xa = s % 7
    return xa, xb, ya, yb

# Precompute all points
_PTS = np.zeros((_SLOTS, 4), dtype=np.uint8)
for s in range(_SLOTS):
    xa, xb, ya, yb = _xy_from_slot(s)
    _PTS[s] = [xa, xb, ya, yb]

# Precompute lines for each direction
# lines[d] is a list of 49 arrays, each of length 49 (the points on that line)
_lines = []
# line_of[d][s] = index of the line in _lines[d] that contains point s
_line_of = []

for d in range(50):
    if d == _INF:
        # v = (0,1): lines are vertical, grouped by x
        dl = []
        d_lof = np.zeros(_SLOTS, dtype=np.int16)
        for xa in range(7):
            for xb in range(7):
                line = []
                for ya in range(7):
                    for yb in range(7):
                        s = _encode(xa, xb, ya, yb)
                        line.append(s)
                        d_lof[s] = len(dl)
                dl.append(np.array(line, dtype=np.int32))
        _lines.append(dl)
        _line_of.append(d_lof)
    else:
        da, db = (d // 7, d % 7)
        # v = (1, da + db*alpha)
        dl = []
        d_lof = np.zeros(_SLOTS, dtype=np.int16)
        for ya0 in range(7):
            for yb0 in range(7):
                line = []
                for ta in range(7):
                    for tb in range(7):
                        # t = ta + tb*alpha
                        # x = 0 + t = t
                        # y = y0 + t*(da + db*alpha)
                        txa = (ta * da + tb * db * _kq) % _p
                        txb = (ta * db + tb * da) % _p
                        nxa = ta % _p
                        nxb = tb % _p
                        nya = (ya0 + txa) % _p
                        nyb = (yb0 + txb) % _p
                        s = _encode(nxa, nxb, nya, nyb)
                        line.append(s)
                        d_lof[s] = len(dl)
                dl.append(np.array(line, dtype=np.int32))
        _lines.append(dl)
        _line_of.append(d_lof)

# Precompute for each point x, the list of "requirements"
# Requirement (x, d) is satisfied if all points in Line_d(x) except x are in N.
# We need to satisfy at least one requirement for each x.

def _compute_validity(N_arr):
    """
    Returns an array 'satisfied' of shape (2401,) where satisfied[x] is True
    if there exists a direction d such that Line_d(x) \ {x} is subset of N.
    Also returns the count of unsatisfied points.
    """
    satisfied = np.zeros(_SLOTS, dtype=bool)
    # For each direction d, we can check which points x have their line (minus x) in N.
    # This is equivalent to: For each line L in direction d, if all points in L except x are in N,
    # then x is satisfied by direction d.
    # Actually, it's easier to iterate by line.
    # For a specific line L (set of 49 points) in direction d:
    # If N contains 48 or 49 points of L, then the missing points (if any) in L are satisfied by d?
    # No. If x is in L, and N contains L \ {x}, then x is satisfied by d.
    # So, for a line L in direction d:
    # Let k = number of points in L that are in N.
    # If k == 49: All 49 points in L are satisfied by d (since for any x in L, L\{x} is in N).
    # If k == 48: The 1 point NOT in N is satisfied by d. The points IN N are also satisfied by d?
    #   Let x be the point NOT in N. L\{x} is in N. So x is satisfied.
    #   Let y be a point IN N. L\{y} contains x (which is NOT in N). So L\{y} is NOT a subset of N.
    #   So y is NOT satisfied by d via this line.
    # If k <= 47: No point in L is satisfied by d.
    
    # So, for each line L in direction d:
    #   k = sum(N_arr[L])
    #   if k == 49: satisfied[L] = True (all points in L get satisfied by d)
    #   if k == 48: satisfied[L[N_arr[L]==0]] = True (the single point not in N gets satisfied)
    
    for d in range(50):
        for line in _lines[d]:
            k = np.sum(N_arr[line])
            if k == 49:
                satisfied[line] = True
            elif k == 48:
                # Find the point in line not in N
                # N_arr[line] is a bool array of length 49
                missing_idx = np.where(N_arr[line] == 0)[0]
                if len(missing_idx) == 1:
                    missing_point = line[missing_idx[0]]
                    satisfied[missing_point] = True
    return satisfied

def _to_array(N_arr):
    """Convert N_arr (bool array of size 2401) to the output array of N."""
    pts = []
    for s in range(_SLOTS):
        if N_arr[s]:
            xa, xb, ya, yb = _xy_from_slot(s)
            pts.append([xa, xb, ya, yb])
    return np.array(pts, dtype=np.int64).reshape(-1, 2, 2)

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 580
    
    # Initialize N with a random set of points
    # We want to minimize |N|, so start with a moderate size and try to shrink/fix
    # A valid N must be large enough to cover lines.
    # Let's start with a random set of size 1500 and use Simulated Annealing.
    
    rng = random.Random(42)
    
    # Initial N: Random subset
    # To ensure we have a chance of validity, maybe start with a structured set?
    # Or just random.
    N_arr = np.zeros(_SLOTS, dtype=bool)
    initial_size = 1500
    indices = np.arange(_SLOTS)
    rng.shuffle(indices)
    N_arr[indices[:initial_size]] = True
    
    # Calculate initial state
    satisfied = _compute_validity(N_arr)
    num_unsat = np.sum(~satisfied)
    current_size = np.sum(N_arr)
    
    # Objective: Minimize (alpha * current_size + beta * num_unsat)
    # We want num_unsat == 0.
    # Let's use a penalty approach.
    # Score = current_size + 1000 * num_unsat
    # We want to minimize this.
    
    best_N = N_arr.copy()
    best_score = current_size + 1000 * num_unsat
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.99
    
    # Precompute neighbors for faster updates?
    # Updating validity for a single point change is expensive if we recalculate everything.
    # But 2401 points * 50 dirs * 49 pts/line is too slow for full recalc per step.
    # We need incremental updates.
    
    # Alternative: Just do full recalculation if it's fast enough?
    # _compute_validity is O(50 * 49 * 49) ~ 120k ops.
    # 10000 steps * 120k = 1.2B ops. Too slow in Python.
    
    # We need a faster validity check or a different heuristic.
    
    # Heuristic: Greedy Removal
    # Start with a valid set (e.g., all points).
    # Try to remove points one by one. If removing a point breaks validity, keep it.
    # This minimizes N greedily.
    
    # 1. Start with N = All points (Valid, size 2401)
    N_arr = np.ones(_SLOTS, dtype=bool)
    satisfied = _compute_validity(N_arr)
    # satisfied should be all True
    
    # 2. Randomly shuffle points
    order = np.arange(_SLOTS)
    rng.shuffle(order)
    
    # 3. For each point in order, try to remove it
    # To check if removal is safe, we need to know if the point was "critical".
    # A point p is critical if removing it causes some x to lose its last valid direction.
    
    # Efficient Check for Removal:
    # When we remove p from N, which lines are affected?
    # All lines passing through p. There are 50 such lines (one per direction).
    # For each direction d, let L be the line through p in dir d.
    # Before removal: k = sum(N[L]).
    # After removal: k' = k - 1.
    # The status of points in L changes.
    # Specifically, if k was 49, now 48. The point p (now removed) might no longer be satisfied by d?
    # Wait, p is removed from N, so p is in S.
    # The condition for x is: exists d s.t. L_d(x)\{x} subset N.
    
    # Let's track for each point x, the set of directions that satisfy it.
    # sat_dirs[x] = bitmask of directions d such that L_d(x)\{x} subset N.
    # Initially, for N=All, sat_dirs[x] = all 50 directions.
    
    sat_dirs = [0xFFFFFFFFFF] * _SLOTS
    
    # Update logic when removing p:
    # For each direction d:
    #   L = line through p in dir d.
    #   Before: k = 49 (if p was in N and all others in N).
    #   Actually, we don't know k for all lines. We should maintain counts.
    
    # Let's maintain line_counts[d][line_idx] = number of points in N on that line.
    line_counts = np.zeros((50, 49), dtype=np.int16)
    for d in range(50):
        for line_idx, line in enumerate(_lines[d]):
            line_counts[d, line_idx] = np.sum(N_arr[line])
    
    # Initial sat_dirs:
    # For each x, d is valid if line_counts[d][line_of[d][x]] >= 49?
    # No. If x is in N, L\{x} has 48 points. If all 48 are in N, then count is 49 (including x) or 48 (if x not in N)?
    # Let count_d(x) = number of points in N on Line_d(x).
    # Condition for x to be satisfied by d:
    #   If x in N: We need L_d(x)\{x} subset N. This means count_d(x) - 1 >= 48 => count_d(x) >= 49.
    #   If x not in N: We need L_d(x)\{x} subset N. This means count_d(x) >= 48.
    # So:
    #   If N_arr[x]: valid if line_counts[d][line_of[d][x]] == 49
    #   If not N_arr[x]: valid if line_counts[d][line_of[d][x]] == 48
    
    # Let's recompute sat_dirs based on this.
    for x in range(_SLOTS):
        mask = 0
        for d in range(50):
            li = _line_of[d][x]
            cnt = line_counts[d, li]
            if N_arr[x]:
                if cnt == 49:
                    mask |= (1 << d)
            else:
                if cnt == 48:
                    mask |= (1 << d)
        sat_dirs[x] = mask
    
    # Now, greedy removal:
    # For p in order:
    #   If N_arr[p] is False, skip (already removed).
    #   Tentatively remove p: N_arr[p] = False.
    #   Update line_counts for lines through p.
    #   Check if any x becomes unsatisfied (sat_dirs[x] becomes 0).
    #   If yes, revert.
    #   If no, keep removal.
    
    # To check "any x becomes unsatisfied", we only need to check points whose sat_dirs might change.
    # Removing p affects lines through p.
    # Points affected are those on lines through p.
    # For each direction d, L = line through p.
    # All points x in L might have their validity for direction d changed.
    
    # Optimization:
    # Instead of checking all x in L, we can just recompute sat_dirs for x in L.
    # But we need to know if sat_dirs[x] becomes 0.
    
    # Let's implement the greedy removal with incremental updates.
    
    for p in order:
        if time.time() > deadline - 10:
            break
            
        if not N_arr[p]:
            continue
            
        # Check if removing p is safe
        # Store old state to revert
        old_N_p = True
        N_arr[p] = False
        
        changes = []
        safe = True
        
        for d in range(50):
            li = _line_of[d][p]
            line = _lines[d][li]
            old_count = line_counts[d, li]
            new_count = old_count - 1
            line_counts[d, li] = new_count
            
            # For each x in line, check if d is still a valid direction for x
            # And if removing d makes sat_dirs[x] == 0
            
            for x in line:
                # Determine if d was valid for x before
                # And if d is valid for x after
                
                # Before:
                if old_N_p: # p was in N
                    # x's count was old_count
                    if N_arr[x]: # x in N (note: x might be p, but N_arr[p] is now False in this loop context? No, we set N_arr[p]=False above)
                        # Careful: N_arr[x] for x=p is now False.
                        # For x != p, N_arr[x] is unchanged.
                        pass
                
                # Let's just re-evaluate validity of d for x based on new state
                # new_count is the count of N on this line.
                # x is in N? 
                x_in_N = N_arr[x]
                
                d_valid_before = False
                d_valid_after = False
                
                # We need to know if d was in sat_dirs[x] before.
                # We didn't store sat_dirs history.
                # But we can infer:
                # If sat_dirs[x] had bit d set, and we remove it, and no other bits are set, it's unsafe.
                
                # Let's just recompute the contribution of d to sat_dirs[x].
                # This is getting complicated.
                
                # Simpler approach:
                # Just recompute sat_dirs for all x in the affected lines.
                # If any becomes 0, revert.
                
                # But recomputing sat_dirs for x involves checking all 50 directions.
                # That's 50 * 49 * 49 ~ 120k ops per point p.
                # 2401 points * 120k = 288M ops. Might be too slow in Python.
                
                # Let's optimize:
                # We only care if sat_dirs[x] becomes 0.
                # sat_dirs[x] loses bit d only if d was the ONLY valid direction for x.
                # So, before removal, if sat_dirs[x] == (1 << d), then removing p (which might invalidate d for x) is dangerous.
                
                # So:
                # 1. Check if (sat_dirs[x] & ~(1 << d)) == 0. (d is the only valid dir)
                # 2. Check if d becomes invalid for x.
                
                if (sat_dirs[x] & ~(1 << d)) == 0:
                    # d is the only valid direction for x
                    # Check if d remains valid for x after removal
                    # d is valid for x if:
                    #   if x in N: new_count == 49
                    #   if x not in N: new_count == 48
                    
                    if x_in_N:
                        if new_count != 49:
                            safe = False
                            break
                    else:
                        if new_count != 48:
                            safe = False
                            break
            
            if not safe:
                break
        
        if safe:
            # Update sat_dirs for all x in affected lines
            for d in range(50):
                li = _line_of[d][p]
                line = _lines[d][li]
                new_count = line_counts[d, li]
                
                for x in line:
                    x_in_N = N_arr[x]
                    if x_in_N:
                        if new_count == 49:
                            sat_dirs[x] |= (1 << d)
                        else:
                            sat_dirs[x] &= ~(1 << d)
                    else:
                        if new_count == 48:
                            sat_dirs[x] |= (1 << d)
                        else:
                            sat_dirs[x] &= ~(1 << d)
        else:
            # Revert
            N_arr[p] = True
            for d in range(50):
                li = _line_of[d][p]
                line_counts[d, li] += 1
            # sat_dirs don't need reverting if we didn't apply changes?
            # Wait, in the "safe" check, we didn't modify sat_dirs.
            # We only modified line_counts.
            # So we just revert line_counts.
            # But we need to be sure sat_dirs is consistent.
            # Since we didn't update sat_dirs during the check, it is still consistent with the OLD state.
            # And we reverted N_arr and line_counts to OLD state.
            # So sat_dirs is correct.

    # Final verification
    satisfied = _compute_validity(N_arr)
    if not np.all(satisfied):
        # Fallback to all points
        N_arr = np.ones(_SLOTS, dtype=bool)
    
    return _to_array(N_arr)

