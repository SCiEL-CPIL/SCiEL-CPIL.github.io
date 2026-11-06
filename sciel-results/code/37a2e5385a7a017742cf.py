"""Combined program: modular sieve pre-filter + fast bitset DP validation."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_set = []
    
    # Strategy 1: Greedy Descending with modular sieve
    if time.time() < deadline:
        s = _greedy_construct_with_sieve(range(512, 0, -1), deadline)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                return best_set
    
    # Strategy 2: Greedy Ascending with modular sieve
    if time.time() < deadline:
        s = _greedy_construct_with_sieve(range(1, 513), deadline)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                return best_set
    
    # Strategy 3: Random shuffles with modular sieve
    if len(best_set) < 48 and time.time() < deadline:
        for _ in range(30):
            if time.time() > deadline:
                break
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            s = _greedy_construct_with_sieve(candidates, deadline)
            if len(s) > len(best_set):
                best_set = s
                if len(best_set) == 48:
                    break
    
    # Strategy 4: If still not 48, try targeted local improvement
    if len(best_set) < 48 and time.time() < deadline:
        s = _local_improvement(best_set, deadline)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                return best_set
    
    return best_set

def _modular_sieve_precheck(selected, x):
    """
    Fast modular pre-check: for small primes p, check if x could be the mean
    of any subset of selected. If 2x mod p cannot equal sum of k elements mod p
    for any k >= 2, then x is safe from being a mean modulo p.
    This is a necessary condition, not sufficient, but filters out many candidates.
    """
    n = len(selected)
    if n < 2:
        return True
    
    primes = [3, 5, 7, 11, 13]
    
    for p in primes:
        # Compute achievable sums modulo p for subsets of size k >= 2
        # dp[k] = set of achievable sums mod p with k elements
        dp = [set() for _ in range(n + 1)]
        dp[0] = {0}
        
        for elem in selected:
            e = elem % p
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    for s in dp[k]:
                        dp[k + 1].add((s + e) % p)
        
        # Check if x mod p is a mean: sum(subset) == k * x for some k >= 2
        x_mod = x % p
        found = False
        for k in range(2, n + 1):
            target = (k * x_mod) % p
            if target in dp[k]:
                found = True
                break
        
        if not found:
            # x cannot be the mean modulo p, so it's safe from this perspective
            continue
        else:
            # x could be the mean modulo p, need further check
            pass
    
    return True

def _greedy_construct_with_sieve(candidates, deadline):
    """
    Greedily add elements from candidates if valid, using modular sieve pre-filter.
    """
    selected = []
    selected_set = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if x in selected_set:
            continue
        
        # Fast modular pre-check
        if not _modular_sieve_precheck(selected, x):
            continue
        
        # Full validation with bitset DP
        if _is_valid_addition(selected, x):
            selected.append(x)
            selected_set.add(x)
            
    return selected

def _is_valid_addition(selected, x):
    """
    Check if adding x to selected maintains the no-mean property.
    Uses bitset DP for exact verification.
    """
    n = len(selected)
    
    # Check 1: Is x the mean of a subset of selected?
    # Sum of k elements = k * x, k >= 2.
    if n >= 2:
        if _subset_mean_exists(selected, x):
            return False
    
    # Check 2: For each a in selected, is a the mean of a subset including x?
    # Subset T from selected \ {a}, size k >= 1.
    # Sum(T) + x = (k+1) * a
    # Sum(T) = (k+1) * a - x
    # We need to check if Sum(T) is achievable by k elements from selected \ {a}.
    
    for a in selected:
        others = [y for y in selected if y != a]
        if not others:
            continue
        
        if _subset_sum_with_target_exists(others, a, x):
            return False
    
    return True

def _subset_mean_exists(selected, x):
    """
    Check if there is a subset of 'selected' of size k >= 2
    such that sum(subset) == k * x.
    Uses bitset DP.
    """
    n = len(selected)
    if n < 2:
        return False
    
    max_sum = sum(selected)
    
    # dp[k] is a bitset of achievable sums with k elements
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements
    
    for elem in selected:
        for k in range(n - 1, -1, -1):
            if dp[k]:
                dp[k + 1] |= (dp[k] << elem)
    
    # Check if for any k >= 2, bit (k*x) is set in dp[k]
    for k in range(2, n + 1):
        target = k * x
        if target > max_sum:
            break
        if (dp[k] >> target) & 1:
            return True
    
    return False

def _subset_sum_with_target_exists(others, a, x):
    """
    Check if there exists k >= 1 such that sum of k elements from 'others'
    equals (k+1)*a - x.
    Uses bitset DP.
    """
    n = len(others)
    if n == 0:
        return False
    
    max_sum = sum(others)
    
    # dp[k] is a bitset of achievable sums with k elements
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for elem in others:
        for k in range(n - 1, -1, -1):
            if dp[k]:
                dp[k + 1] |= (dp[k] << elem)
    
    for k in range(1, n + 1):
        target = (k + 1) * a - x
        if target < 0:
            continue
        if target > max_sum:
            continue
        if (dp[k] >> target) & 1:
            return True
    
    return False

def _local_improvement(current_set, deadline):
    """
    Try to improve the current set by swapping or adding elements.
    """
    if not current_set:
        return current_set
    
    current_list = list(current_set)
    current_set_copy = set(current_list)
    
    # Try adding elements not in the set
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if len(current_list) >= 48:
            break
        if x in current_set_copy:
            continue
        
        if _is_valid_addition(current_list, x):
            current_list.append(x)
            current_set_copy.add(x)
    
    # Try swapping: remove one element, try to add two
    for i in range(len(current_list)):
        if time.time() > deadline:
            break
        if len(current_list) >= 48:
            break
        
        removed = current_list[i]
        temp_list = [x for j, x in enumerate(current_list) if j != i]
        temp_set = set(temp_list)
        
        # Try to add two new elements
        for x1 in range(1, 513):
            if time.time() > deadline:
                break
            if x1 in temp_set:
                continue
            if not _is_valid_addition(temp_list, x1):
                continue
            
            new_list1 = temp_list + [x1]
            new_set1 = temp_set | {x1}
            
            for x2 in range(1, 513):
                if time.time() > deadline:
                    break
                if x2 in new_set1:
                    continue
                if x2 == removed:
                    continue
                if not _is_valid_addition(new_list1, x2):
                    continue
                
                new_list2 = new_list1 + [x2]
                if len(new_list2) > len(current_list):
                    current_list = new_list2
                    current_set_copy = new_set1 | {x2}
                    break
            else:
                continue
            break
    
    return current_list
