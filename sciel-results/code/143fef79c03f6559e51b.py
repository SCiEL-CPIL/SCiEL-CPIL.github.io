"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce
from fractions import Fraction

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_all(nums):
    if not nums:
        return 1
    return reduce(_lcm, nums)

def _is_valid_fast(S, deadline=None):
    """
    Check if S is reciprocal-subsum-free using LCM-based integer arithmetic.
    Returns True if valid, False if not, None if timeout.
    """
    if not S:
        return True
    
    L = _lcm_all(S)
    # Convert reciprocals to integers: 1/x -> L/x
    # For each x in S, check if L/x can be expressed as sum of L/y for y in S, y != x
    
    for x in S:
        if deadline is not None and time.time() > deadline:
            return None
        target = L // x
        others = [y for y in S if y != x]
        if not others:
            continue
        vals = [L // y for y in others]
        m = len(vals)
        
        # Sum of all vals
        total_sum = sum(vals)
        if total_sum < target:
            continue  # Can't reach target
        
        # Use a set of achievable sums, pruning sums > target
        achievable = {0}
        for v in vals:
            if v > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                return False
    return True

def _is_valid_exact(S, deadline=None):
    """
    Exact check using Fraction for correctness verification.
    """
    if not S:
        return True
    
    S_list = list(S)
    for x in S_list:
        if deadline is not None and time.time() > deadline:
            return None
        target = Fraction(1, x)
        others = [y for y in S_list if y != x]
        if not others:
            continue
        m = len(others)
        others_sorted = sorted(others)
        
        # Suffix sum for pruning
        suffix_sum = [Fraction(0, 1)] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, others_sorted[i])
        
        if suffix_sum[0] < target:
            continue
        
        found = [False]
        
        def dfs(idx, curr):
            if found[0]:
                return
            if curr == target:
                if idx > 0:
                    found[0] = True
                return
            if idx == m:
                return
            if curr > target:
                return
            if curr + suffix_sum[idx] < target:
                return
            
            y = others_sorted[idx]
            new_curr = curr + Fraction(1, y)
            if new_curr <= target:
                dfs(idx + 1, new_curr)
                if found[0]:
                    return
            dfs(idx + 1, curr)
        
        dfs(0, Fraction(0, 1))
        if found[0]:
            return False
    return True

def _backtracking_search(deadline, target_size, start_set=None):
    """
    Backtracking search with forward checking to find a valid set of target_size.
    Returns a valid set of size target_size if found, else None.
    """
    if start_set is None:
        start_set = set()
    
    current = set(start_set)
    all_elements = list(range(1, 25))
    
    # Order elements by some heuristic - try larger elements first (they have smaller reciprocals)
    available = [e for e in all_elements if e not in current]
    available.sort(reverse=True)
    
    best_found = set(current)
    
    def backtrack(idx, current_set):
        nonlocal best_found
        if time.time() > deadline:
            return None
        if len(current_set) >= target_size:
            if _is_valid_fast(current_set, deadline) is True:
                best_found = set(current_set)
                return current_set
            return None
        
        if idx >= len(available):
            return None
        
        # Try including available[idx]
        elem = available[idx]
        candidate_with = current_set | {elem}
        
        # Quick check: is candidate_with valid?
        if _is_valid_fast(candidate_with, deadline) is True:
            result = backtrack(idx + 1, candidate_with)
            if result is not None:
                return result
        
        # Try excluding available[idx]
        result = backtrack(idx + 1, current_set)
        if result is not None:
            return result
        
        return None
    
    result = backtrack(0, current)
    if result is not None:
        return result
    return None

def _improve_with_backtracking(current_set, deadline):
    """
    Try to improve current_set by adding elements using backtracking.
    Returns the best set found.
    """
    best = set(current_set)
    best_size = len(best)
    
    # Try to add one more element
    for target_size in range(best_size + 1, min(best_size + 4, 25)):
        if time.time() > deadline:
            break
        
        result = _backtracking_search(deadline, target_size, best)
        if result is not None and len(result) > best_size:
            best = set(result)
            best_size = len(best)
    
    return best

def _local_search(S, deadline):
    """
    Local search: try to add elements to S or swap elements to improve.
    S is a set of integers. Returns the best set found.
    """
    current = set(S)
    best = set(current)
    best_size = len(best)
    
    # Try to add elements from [1,24] not in current
    available = [i for i in range(1, 25) if i not in current]
    random.shuffle(available)
    
    # First pass: try adding elements
    for elem in available:
        if time.time() > deadline:
            break
        candidate = current | {elem}
        if len(candidate) > best_size:
            if _is_valid_fast(candidate, deadline) is True:
                current = candidate
                best = set(current)
                best_size = len(best)
                # Refresh available
                available = [i for i in range(1, 25) if i not in current]
                random.shuffle(available)
    
    # Use backtracking to try to add more elements
    if time.time() < deadline:
        improved = _improve_with_backtracking(current, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Second pass: try swapping - remove one element, add another
    for _ in range(30):  # Limited iterations
        if time.time() > deadline:
            break
        if not current:
            break
        # Try removing a random element and adding a different one
        to_remove = random.choice(list(current))
        candidates_to_add = [i for i in range(1, 25) if i not in current]
        if not candidates_to_add:
            break
        to_add = random.choice(candidates_to_add)
        
        candidate = (current - {to_remove}) | {to_add}
        if len(candidate) >= best_size:
            if _is_valid_fast(candidate, deadline) is True:
                current = candidate
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(best)
    
    return best

def _random_initial(size, deadline):
    """Generate a random initial set of given size."""
    all_elems = list(range(1, 25))
    random.shuffle(all_elems)
    # Try to find a valid subset of the desired size from the shuffled list
    # Start with a small set and grow
    current = set()
    for elem in all_elems:
        if time.time() > deadline:
            break
        candidate = current | {elem}
        if _is_valid_fast(candidate, deadline) is True:
            current = candidate
            if len(current) >= size:
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit (using 55s to be safe)
    
    random.seed(42)  # Deterministic for reproducibility
    
    best_set = set()
    best_size = 0
    
    # Try backtracking search first to find a large valid set
    if time.time() < deadline:
        # Try to find a set of size 15
        result = _backtracking_search(deadline, 15)
        if result is not None:
            best_set = set(result)
            best_size = len(best_set)
    
    # Multiple random restarts with local search
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate a random initial set
        initial_size = random.randint(8, 14)
        initial = _random_initial(initial_size, deadline)
        
        if len(initial) > best_size:
            best_set = set(initial)
            best_size = len(initial)
        
        # Local search from this initial
        if time.time() > deadline:
            break
        improved = _local_search(initial, deadline)
        
        if len(improved) > best_size:
            best_set = set(improved)
            best_size = len(improved)
    
    # Also try greedy from high to low as a baseline
    if time.time() < deadline:
        greedy_set = set()
        for i in range(24, 0, -1):
            if time.time() > deadline:
                break
            candidate = greedy_set | {i}
            if _is_valid_fast(candidate, deadline) is True:
                greedy_set = candidate
        if len(greedy_set) > best_size:
            best_set = set(greedy_set)
            best_size = len(greedy_set)
    
    # Also try greedy from low to high
    if time.time() < deadline:
        greedy_set2 = set()
        for i in range(1, 25):
            if time.time() > deadline:
                break
            candidate = greedy_set2 | {i}
            if _is_valid_fast(candidate, deadline) is True:
                greedy_set2 = candidate
        if len(greedy_set2) > best_size:
            best_set = set(greedy_set2)
            best_size = len(greedy_set2)
    
    # Final verification with exact method
    if best_set:
        if time.time() < deadline:
            verify = _is_valid_exact(best_set, deadline)
            if verify is True:
                return sorted(best_set)
            elif verify is None:
                # Timeout on verification, return as-is (likely valid)
                return sorted(best_set)
            else:
                # Invalid, need to fix
                # Remove elements until valid
                current = set(best_set)
                while current:
                    if time.time() > deadline:
                        break
                    if _is_valid_fast(current, deadline) is True:
                        break
                    # Remove a random element
                    to_remove = random.choice(list(current))
                    current.discard(to_remove)
                if current:
                    return sorted(current)
    
    # Fallback
    return list(range(13, 25))

# EVOLVE-BLOCK-END
