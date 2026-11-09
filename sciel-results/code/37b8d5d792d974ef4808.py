"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 256}
_DEADLINE = None

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    d = 5
    while d * d <= x:
        if x % d == 0 or x % (d + 2) == 0:
            return False
        d += 6
    return True

def _valid_set(S):
    """Check if set S is valid: for all distinct a,b,c in S, a ∤ b*c."""
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            for k in range(j + 1, m):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _build_conflict_structure(nums):
    """
    Precompute conflict relationships for fast lookups.
    For each number x, precompute:
    - conflicts_with_pairs[x]: set of (a,b) pairs where x conflicts with {a,b}
    - This means: x ∤ a*b OR a ∤ x*b OR b ∤ x*a (any violation)
    """
    n = len(nums)
    # Map value to index
    val_to_idx = {nums[i]: i for i in range(n)}
    
    # For each number, track which other numbers it conflicts with
    # conflict[i] = set of j such that {nums[i], nums[j]} cannot both be in a valid set
    # with some third element... actually we need triple conflicts
    
    # Better approach: for each number x, precompute which pairs (a,b) conflict with x
    # A triple {a,b,c} is invalid if:
    #   a | b*c OR b | a*c OR c | a*b
    # So for a fixed x, x conflicts with pair (a,b) if:
    #   x | a*b OR a | x*b OR b | x*a
    
    # conflict_pairs[x_idx] = set of (a_idx, b_idx) that conflict with x
    conflict_pairs = [set() for _ in range(n)]
    
    for i in range(n):
        x = nums[i]
        for j in range(i + 1, n):
            a = nums[j]
            for k in range(j + 1, n):
                b = nums[k]
                # Check if {x, a, b} is invalid
                if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                    # x conflicts with pair (j, k)
                    conflict_pairs[i].add((j, k))
                    # j conflicts with pair (i, k)
                    conflict_pairs[j].add((i, k))
                    # k conflicts with pair (i, j)
                    conflict_pairs[k].add((i, j))
    
    return conflict_pairs, val_to_idx

def _can_add_fast(current_set_indices, x_idx, conflict_pairs, nums):
    """
    Check if x can be added to current_set without violating constraints.
    current_set_indices: set of indices in nums
    x_idx: index of x in nums
    """
    x = nums[x_idx]
    cs = list(current_set_indices)
    m = len(cs)
    
    # Check: for all pairs (a,b) in current_set, the triple {x,a,b} must be valid
    for i in range(m):
        a_idx = cs[i]
        a = nums[a_idx]
        for j in range(i + 1, m):
            b_idx = cs[j]
            b = nums[b_idx]
            if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                return False
    return True

def _conflict_density(x_idx, current_set_indices, conflict_pairs):
    """
    Count how many pairs in current_set conflict with x_idx.
    Higher density means including x forces more exclusions.
    """
    count = 0
    cs = list(current_set_indices)
    m = len(cs)
    for i in range(m):
        a_idx = cs[i]
        for j in range(i + 1, m):
            b_idx = cs[j]
            # Check if x conflicts with pair (a, b)
            if (a_idx, b_idx) in conflict_pairs[x_idx] or \
               (b_idx, a_idx) in conflict_pairs[x_idx]:
                count += 1
    return count

def _greedy_with_density(nums, conflict_pairs, val_to_idx, deadline):
    """
    Greedy algorithm that selects vertices in order of lowest conflict density.
    """
    selected = set()
    remaining = set(range(len(nums)))
    
    while remaining and time.time() < deadline:
        # For each candidate, compute conflict density with current selection
        best_candidate = None
        best_density = float('inf')
        
        for idx in remaining:
            density = _conflict_density(idx, selected, conflict_pairs)
            if density < best_density:
                best_density = density
                best_candidate = idx
        
        if best_candidate is None:
            break
        
        if _can_add_fast(selected, best_candidate, conflict_pairs, nums):
            selected.add(best_candidate)
            remaining.discard(best_candidate)
        else:
            remaining.discard(best_candidate)
    
    return selected

def _branch_and_bound_density(remaining, current_set, best, deadline, conflict_pairs, nums):
    """
    Specialized branch-and-bound for 3-uniform hypergraph independent set.
    Selects pivot with highest conflict density to maximize pruning.
    """
    if time.time() > deadline:
        return
    
    if not remaining:
        if len(current_set) > len(best[0]):
            best[0] = set(current_set)
        return
    
    # Upper bound
    if len(current_set) + len(remaining) <= len(best[0]):
        return
    
    # Select pivot: element with highest conflict density in remaining
    # (including it forces most exclusions)
    pivot = None
    pivot_density = -1
    
    for x_idx in remaining:
        density = _conflict_density(x_idx, current_set, conflict_pairs)
        # Also count conflicts with other remaining elements
        rem_conflicts = 0
        for y_idx in remaining:
            if y_idx == x_idx:
                continue
            # Check if x conflicts with y (in terms of being able to coexist)
            # For simplicity, just use density with current_set
            pass
        if density > pivot_density:
            pivot_density = density
            pivot = x_idx
    
    if pivot is None:
        return
    
    rest = remaining - {pivot}
    
    # Option 1: Include pivot (if possible)
    if _can_add_fast(current_set, pivot, conflict_pairs, nums):
        new_set = current_set | {pivot}
        # Filter remaining: keep only elements compatible with new_set
        filtered = set()
        for x_idx in rest:
            if _can_add_fast(new_set, x_idx, conflict_pairs, nums):
                filtered.add(x_idx)
        _branch_and_bound_density(filtered, new_set, best, deadline, conflict_pairs, nums)
    
    # Option 2: Exclude pivot
    _branch_and_bound_density(rest, current_set, best, deadline, conflict_pairs, nums)

def _local_search_fast(S_indices, all_indices, conflict_pairs, nums, deadline):
    """
    Fast local search using bitmask-style conflict checks.
    Tries 1-for-1 and 1-for-2 swaps.
    """
    best = set(S_indices)
    best_size = len(best)
    candidates = all_indices - best
    
    # 1-for-1 swaps
    for rem in list(best):
        if time.time() > deadline:
            break
        S2 = best - {rem}
        for add in candidates:
            if add in S2:
                continue
            if _can_add_fast(S2, add, conflict_pairs, nums):
                S2.add(add)
                if len(S2) > best_size:
                    best = S2
                    best_size = len(best)
                    candidates = all_indices - best
    
    # 1-for-2 swaps (remove one, try to add two)
    for rem in list(best):
        if time.time() > deadline:
            break
        S2 = best - {rem}
        added = []
        for add in candidates:
            if add in S2 or len(added) >= 2:
                continue
            if _can_add_fast(S2, add, conflict_pairs, nums):
                S2.add(add)
                added.append(add)
        if len(S2) > best_size:
            best = S2
            best_size = len(best)
            candidates = all_indices - best
    
    return best

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 85  # 85 second budget (leave 5s margin for 90s limit)
    
    n = P['n']
    nums = list(range(2, n + 1))
    all_indices = set(range(len(nums)))
    
    # Precompute conflict structure
    conflict_pairs, val_to_idx = _build_conflict_structure(nums)
    
    # Get primes as initial solution
    primes = [x for x in nums if _is_prime(x)]
    prime_indices = {val_to_idx[p] for p in primes}
    best_set = prime_indices
    best_size = len(prime_indices)
    
    # Try greedy with density ordering
    if time.time() < _DEADLINE:
        greedy_result = _greedy_with_density(nums, conflict_pairs, val_to_idx, _DEADLINE)
        if len(greedy_result) > best_size:
            best_set = greedy_result
            best_size = len(best_set)
    
    # Branch and bound from best solution
    if time.time() < _DEADLINE:
        remaining = all_indices - best_set
        best_wrapper = [best_set]
        
        # Sort remaining by conflict density (lowest first for inclusion)
        # but B&B will select highest density pivot internally
        _branch_and_bound_density(remaining, best_set, best_wrapper, _DEADLINE, conflict_pairs, nums)
        
        if len(best_wrapper[0]) > best_size:
            best_set = best_wrapper[0]
            best_size = len(best_set)
    
    # Local search polish
    if time.time() < _DEADLINE:
        best_set = _local_search_fast(best_set, all_indices, conflict_pairs, nums, _DEADLINE)
        best_size = len(best_set)
    
    # Convert back to values
    S_list = sorted(nums[i] for i in best_set)
    
    # Final verification
    if not _valid_set(S_list):
        # Fallback to primes
        S_list = sorted(primes)
    
    return list(S_list)
# EVOLVE-BLOCK-END
