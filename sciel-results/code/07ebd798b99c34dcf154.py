"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import itertools
import sys

_DEADLINE = None
_BEST = None

def _init():
    global _DEADLINE, _BEST
    _DEADLINE = time.time() + 85.0  # 5s safety margin before 90s kill
    _BEST = None

def _check_safe(a, others):
    """Return True if no nonempty subset of `others` sums to Fraction(1, a)."""
    target = Fraction(1, a)
    m = len(others)
    if m == 0:
        return True
    
    # Use meet-in-the-middle for larger sets, brute force with pruning for smaller
    if m <= 18:
        # DFS with pruning: sort others by reciprocal descending (i.e., element ascending)
        o = sorted(others)  # smaller elements have larger reciprocals
        n = len(o)
        # Precompute suffix sums for pruning: if sum of remaining < target - acc, skip
        # Actually, since all positive, if acc > target we can prune (but acc only increases)
        # Better: if acc == target, we found a violation. If acc > target, prune.
        
        found_violation = [False]
        
        def dfs(i, acc):
            if found_violation[0]:
                return
            if acc == target:
                found_violation[0] = True
                return
            if i >= n:
                return
            if acc > target:
                return
            
            # Option 1: skip element i
            dfs(i + 1, acc)
            if found_violation[0]:
                return
            
            # Option 2: take element i
            new_acc = acc + Fraction(1, o[i])
            if new_acc <= target:
                dfs(i + 1, new_acc)
        
        dfs(0, Fraction(0))
        return not found_violation[0]
    else:
        # Meet-in-the-middle
        mid = m // 2
        left = others[:mid]
        right = others[mid:]
        
        # Generate all subset sums for left
        left_sums = set()
        for r in range(mid + 1):
            for combo in itertools.combinations(left, r):
                s = sum(Fraction(1, x) for x in combo)
                left_sums.add(s)
        
        # Generate all subset sums for right, check if target - s is in left_sums
        # But we must ensure the subset is nonempty overall. Since target > 0, 
        # if target is in left_sums or right_sums, that's a nonempty subset.
        # Actually, we need: exists nonempty S ⊆ others such that sum_{x in S} 1/x = 1/a
        # Split S into S_L ⊆ left, S_R ⊆ right. sum(S_L) + sum(S_R) = target.
        # S is nonempty iff S_L nonempty or S_R nonempty.
        
        # Check if target is in left_sums (S_R empty) or in right sums (S_L empty)
        # But we need to be careful: the empty subset has sum 0.
        # If target is in left_sums, it could be from empty subset only if target=0, which it isn't.
        # So any occurrence of target in left_sums or right_sums implies nonempty.
        
        # More carefully: for each sum_r in right_sums, check if (target - sum_r) is in left_sums.
        # If sum_r == 0 (empty right) and target in left_sums -> valid (nonempty left).
        # If sum_r > 0 and target - sum_r == 0 -> 0 in left_sums (empty left) -> valid (nonempty right).
        # If both nonempty -> valid.
        
        # So we just need to check: exists sum_l in left_sums, sum_r in right_sums such that
        # sum_l + sum_r == target AND (sum_l != 0 OR sum_r != 0).
        # Since target > 0, sum_l + sum_r == target implies not both are 0.
        
        right_sums = set()
        for r in range(len(right) + 1):
            for combo in itertools.combinations(right, r):
                s = sum(Fraction(1, x) for x in combo)
                right_sums.add(s)
        
        for sl in left_sums:
            sr_needed = target - sl
            if sr_needed in right_sums:
                # Check nonempty: since target > 0, if sl==0 and sr_needed==0 then target==0, contradiction.
                # So at least one is nonempty.
                return False
        return True

def _is_reciprocal_subsum_free(S):
    """Check if set S is reciprocal-subsum-free."""
    S_list = sorted(S)
    for a in S_list:
        others = [x for x in S_list if x != a]
        if not _check_safe(a, others):
            return False
    return True

def _greedy_construct():
    """Try to build a large reciprocal-subsum-free set using greedy approach."""
    # Start with larger numbers (smaller reciprocals) as they're easier to fit
    candidates = list(range(24, 0, -1))  # 24 down to 1
    selected = []
    
    for x in candidates:
        test_set = selected + [x]
        if _is_reciprocal_subsum_free(test_set):
            selected.append(x)
    
    return selected

def _random_search(start_time, time_limit):
    """Random local search to find larger sets."""
    global _BEST
    
    # Start with a good initial solution
    best = _greedy_construct()
    if _BEST is None or len(best) > len(_BEST):
        _BEST = list(best)
    
    # Try to improve by random additions/removals
    import random
    random.seed(42)
    
    current = list(best)
    
    while time.time() < time_limit - 5.0:
        # Try random move: add a random element, or swap, or remove
        move_type = random.choice(['add', 'swap', 'remove'])
        
        if move_type == 'add':
            candidates = [x for x in range(1, 25) if x not in current]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_set = current + [x]
            if _is_reciprocal_subsum_free(new_set):
                current = new_set
                if _BEST is None or len(current) > len(_BEST):
                    _BEST = list(current)
        elif move_type == 'swap':
            if len(current) == 0:
                continue
            out_elem = random.choice(current)
            in_candidates = [x for x in range(1, 25) if x not in current]
            if not in_candidates:
                continue
            in_elem = random.choice(in_candidates)
            new_set = [x for x in current if x != out_elem] + [in_elem]
            if _is_reciprocal_subsum_free(new_set):
                current = new_set
                if _BEST is None or len(current) > len(_BEST):
                    _BEST = list(current)
        elif move_type == 'remove':
            if len(current) <= 1:
                continue
            x = random.choice(current)
            new_set = [y for y in current if y != x]
            if _is_reciprocal_subsum_free(new_set):
                # Only accept if it leads to potential improvement (skip for now)
                pass
    
    return _BEST

def construct():
    _init()
    
    # First, try a systematic construction
    best = _greedy_construct()
    
    # Try to improve with random search
    time_limit = _DEADLINE
    best = _random_search(time.time(), time_limit)
    
    # If still time, try more systematic approaches
    if time.time() < time_limit - 10.0 and best is not None:
        # Try starting from different initial sets
        for start_range in [range(20, 25), range(15, 25), range(10, 25)]:
            if time.time() > time_limit - 10.0:
                break
            initial = list(start_range)
            # Check if initial is valid
            if _is_reciprocal_subsum_free(initial):
                # Try to extend
                current = list(initial)
                for x in range(1, 25):
                    if time.time() > time_limit - 10.0:
                        break
                    if x not in current:
                        test_set = current + [x]
                        if _is_reciprocal_subsum_free(test_set):
                            current = test_set
                if _BEST is None or len(current) > len(_BEST):
                    _BEST = list(current)
    
    # Ensure we return a nonempty valid set
    if _BEST is None or not _is_reciprocal_subsum_free(_BEST):
        # Fallback: single element is always valid
        _BEST = [24]
    
    return list(_BEST)
# EVOLVE-BLOCK-END
