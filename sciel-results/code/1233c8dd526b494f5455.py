import time
import random
from itertools import combinations

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    Uses a bitmask-based CSP backtracking search with MRV heuristic and iterative deepening.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s margin before 150s kill

    TARGET = set(range(1, 31))
    # Bitmask for target differences 1..30
    # Bit i represents difference i+1
    TARGET_MASK = (1 << 30) - 1

    def is_sidon_fast(S):
        """Check if S is a Sidon set using a set of differences."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(S):
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return diffs

    def covers_all(S):
        diffs = get_diffs(S)
        return TARGET.issubset(diffs)

    def score_set(S):
        diffs = get_diffs(S)
        missing = len(TARGET - diffs)
        return (missing, max(S) if S else 0)

    def search(max_val, deadline):
        """
        Bitmask-based backtracking search for a Sidon set with max element <= max_val.
        """
        # Precompute candidate masks
        # For each x in 1..max_val, we need to know:
        # 1. Which differences it forms with 0? (just x)
        # 2. But we build incrementally.
        
        # State:
        # S: list of chosen elements (sorted)
        # used_diff_mask: bitmask of differences already present in S
        # used_val_mask: bitmask of values already in S (up to max_val)
        
        # Optimization: Precompute for each x, the differences it creates with any y < x
        # But y depends on S.
        
        # Let's use a recursive search.
        # To speed up, we will precompute a list of "valid next moves" dynamically?
        # No, dynamic is better for MRV.
        
        # MRV: Pick the uncovered target difference d with the fewest candidates x 
        # such that adding x covers d and doesn't create duplicate diffs.
        
        # Candidates for a difference d given current S:
        # We need x, y in S_new with |x-y|=d.
        # Since we add one element at a time, the new element x must pair with some y in S
        # to create d. So x = y + d or x = y - d.
        
        # We will maintain:
        # S: list
        # used_diff_mask: int
        # used_val_mask: int (bit i set if i in S)
        
        # For MRV, for each uncovered d in TARGET:
        #   candidates = []
        #   for y in S:
        #      x1 = y + d
        #      x2 = y - d
        #      for x in {x1, x2}:
        #         if x in used_val_mask: continue
        #         if x > max_val: continue
        #         # Check if adding x creates any duplicate difference
        #         # New diffs are |x - y'| for all y' in S
        #         # If any new diff is in used_diff_mask, invalid.
        #         # Also if any new diff is 0 (x in S, handled above)
        #         # And we must ensure the new diffs themselves are unique?
        #         # Yes, if |x-y1| == |x-y2| then y1==y2 or y1+y2==2x.
        #         # If y1 != y2, then x creates duplicate diff.
        #         # So we need to check if any pair (y1, y2) in S satisfies |x-y1| == |x-y2|.
        #         # This is equivalent to checking if x is the midpoint of any pair in S.
        #         # Or simply: compute set of new diffs, check intersection with used_diff_mask
        #         # and check if new diffs have duplicates among themselves.
        #         # Since S is Sidon, all diffs between pairs in S are unique.
        #         # The new diffs are |x-y| for y in S.
        #         # Duplicates among new diffs: |x-y1| == |x-y2| => y1==y2 or y1+y2=2x.
        #         # If y1!=y2, then 2x = y1+y2. So x must not be the midpoint of any two elements in S.
        
        #         # Check 1: Intersection with used_diff_mask
        #         # Check 2: Midpoint collision
         
        #         # If valid, add to candidates for d.
        
        #   If any d has 0 candidates, fail.
        #   Pick d with min candidates.
        #   Try each candidate x.
        
        # To make this fast, we need efficient checks.
        
        # Let's precompute midpoints? No, S changes.
        
        # We will implement a helper to check validity of x given S and used_diff_mask.
        
        def is_valid_add(x, S, used_diff_mask, used_val_mask):
            if x in used_val_mask: return False
            if x > max_val: return False
            if x == 0: return False # 0 is already in S
            
            # Check midpoint collision: does x create duplicate diffs with S?
            # |x-y1| == |x-y2| for distinct y1, y2 in S?
            # This happens if x = (y1+y2)/2.
            # So 2*x must not be sum of any two distinct elements in S.
            # We can check this by iterating pairs? O(N^2).
            # Or maintain a set of sums?
            # Let's just iterate pairs for now, N is small (<32).
            
            # Actually, simpler: 
            # new_diffs = []
            # for y in S:
            #   d = abs(x-y)
            #   if d in used_diff_mask: return False
            #   if d in new_diffs: return False # Duplicate among new diffs
            #   new_diffs.append(d)
            
            # This is O(N) per check.
            
            new_diffs = []
            for y in S:
                d = abs(x - y)
                if d == 0: return False # Should be caught by used_val_mask
                # Check against used
                if (1 << (d-1)) & used_diff_mask:
                    return False
                # Check against new
                # We can use a set or bitmask for new_diffs too
                if d in new_diffs:
                    return False
                new_diffs.append(d)
            return True

        # To speed up MRV, we need to count candidates for each d quickly.
        # We'll do a full scan of S for each d.
        
        def backtrack(S, used_diff_mask, used_val_mask, deadline):
            if time.time() > deadline:
                return None
            
            if used_diff_mask == TARGET_MASK:
                return S[:]
            
            if len(S) >= 32:
                return None
            
            # MRV
            uncovered = TARGET_MASK ^ used_diff_mask
            # Get list of uncovered differences
            uncovered_list = [i+1 for i in range(30) if (uncovered >> i) & 1]
            
            best_d = None
            best_cands = []
            min_cands = float('inf')
            
            for d in uncovered_list:
                cands = []
                for y in S:
                    # x = y + d
                    x1 = y + d
                    if x1 <= max_val and not (used_val_mask & (1 << x1)):
                        if is_valid_add(x1, S, used_diff_mask, used_val_mask):
                            cands.append(x1)
                    # x = y - d
                    x2 = y - d
                    if x2 > 0 and not (used_val_mask & (1 << x2)):
                        if is_valid_add(x2, S, used_diff_mask, used_val_mask):
                            cands.append(x2)
                
                # Remove duplicates in cands?
                # x1 and x2 could be same? No, d>0.
                # But different y's could yield same x.
                # e.g. y1+d = x and y2-d = x => y1-y2 = 2d.
                # So x might appear multiple times.
                # We should unique them.
                cands = list(set(cands))
                
                if len(cands) < min_cands:
                    min_cands = len(cands)
                    best_d = d
                    best_cands = cands
                    if min_cands == 0:
                        return None
                    if min_cands == 1:
                        break # Can't get better than 1
            
            if not best_cands:
                return None
                
            # Sort candidates? Smaller first?
            # To minimize max element, smaller x is better?
            # Not necessarily, but usually.
            best_cands.sort()
            
            for x in best_cands:
                new_S = S + [x]
                new_used_diff = used_diff_mask
                for y in S:
                    d = abs(x - y)
                    new_used_diff |= (1 << (d-1))
                new_used_val = used_val_mask | (1 << x)
                
                sol = backtrack(new_S, new_used_diff, new_used_val, deadline)
                if sol is not None:
                    return sol
            
            return None

        # Initial state
        S_init = [0]
        used_diff_init = 0
        used_val_init = 1 # Bit 0 set
        
        return backtrack(S_init, used_diff_init, used_val_init, deadline)

    best_S = None
    best_score = (30, 10**9)
    
    # Iterative deepening on max_val
    # Start from a reasonable bound. 
    # A perfect difference set of size k has diameter ~k^2.
    # For k=16 (to cover 1..30 with pairs?), no, we need 30 diffs.
    # Min size is ceil(sqrt(30)) ~ 6? No, Sidon set size k has k(k-1)/2 diffs.
    # We need at least 30 diffs. k(k-1)/2 >= 30 => k~8.
    # But we have constraints.
    # Let's start from 50 and go up? Or down from a known good?
    # The previous program found something around 100-200?
    # Let's try a range.
    
    # First, try to find ANY solution with a loose bound to have a fallback
    # But we want to minimize max.
    # Let's try bounds: 50, 60, 70, 80, 90, 100, 120, 150, 200, 300, 500, 1000
    
    bounds = [40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 250, 300, 400, 500, 1000]
    
    for bound in bounds:
        if time.time() > DEADLINE - 5:
            break
            
        # Try to find a solution with max_val = bound
        # We might need multiple restarts or just one deep search?
        # Backtracking with MRV is deterministic given the order.
        # Let's run one search.
        
        sol = search(bound, DEADLINE - 5)
        
        if sol is not None:
            # Verify
            if is_sidon_fast(sol) and covers_all(sol):
                s = score_set(sol)
                if s < best_score:
                    best_score = s
                    best_S = sol
                    # Found a solution for this bound.
                    # Since we are increasing bound, this is the minimal bound for this search path.
                    # But maybe a smaller bound was skipped? No, we iterate up.
                    # Wait, if we find a solution for bound=50, it's valid.
                    # Can we find a smaller one? No, we started at 40.
                    # So if 40 failed and 50 succeeded, 50 is the answer for this strategy.
                    # We can break if we are confident?
                    # MRV might miss solutions? No, it's complete.
                    # So if search(40) returned None, there is NO solution with max<=40.
                    # So 50 is the minimal possible max for this set of constraints?
                    # Yes, if the search is complete.
                    # So we can break.
                    break
        else:
            # No solution found for this bound (or timeout)
            # If timeout, we should stop?
            # If it didn't timeout, we continue to next bound.
            # How to distinguish?
            # search returns None on timeout or no solution.
            # We can check time.
            # If time.time() > DEADLINE - 5, we are out of time.
            pass

    # Fallback if no solution found (should not happen)
    if best_S is None:
        # Simple greedy
        best_S = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            for x in range(1, 1000000):
                if x in best_S:
                    continue
                trial = sorted(best_S + [x])
                if is_sidon_fast(trial):
                    trial_diffs = get_diffs(trial)
                    if d in trial_diffs:
                        best_S = trial
                        covered = trial_diffs & TARGET
                        break
            else:
                y = 2 * max(best_S) + d + 1
                best_S = sorted(best_S + [y, y + d])
                covered = get_diffs(best_S) & TARGET

    return best_S
