"""B_3[1] set construction: maximize size of subset of [1,4096] with unique triple sums."""
import time
import random
import sys
import numpy as np


def _all_triple_sums(lst):
    """Compute all unordered triple sums for a sorted list."""
    n = len(lst)
    sums = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.append(lst[i] + lst[j] + lst[k])
    return sums


def _is_valid_triples(lst):
    """Check if all unordered triple sums are distinct."""
    if not lst:
        return True
    sums = _all_triple_sums(lst)
    return len(sums) == len(set(sums))


def _new_triple_sums(existing, x):
    """Compute triple sums involving x with existing sorted elements."""
    n = len(existing)
    sums = [3 * x]
    for i in range(n):
        sums.append(2 * x + existing[i])
        sums.append(x + 2 * existing[i])
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + existing[i] + existing[j])
    return sums


def _check_and_add(existing, x, existing_sums):
    """
    Check if x can be added to existing without triple sum collision.
    Returns True if added, False otherwise.
    """
    n = len(existing)
    new_sums = [3 * x]
    for i in range(n):
        new_sums.append(2 * x + existing[i])
        new_sums.append(x + 2 * existing[i])
    for i in range(n):
        for j in range(i + 1, n):
            new_sums.append(x + existing[i] + existing[j])
    
    for s in new_sums:
        if s in existing_sums:
            return False
    
    existing_sums.update(new_sums)
    existing.append(x)
    existing.sort()
    return True


def _exponential_base(b, max_val):
    """Build a set from powers of base b."""
    result = []
    val = 1
    while val <= max_val:
        result.append(val)
        val *= b
    return result


def _construct_b3_set():
    """Direct construction using powers of 4 as seed, then greedy extension."""
    base_set = _exponential_base(4, 4096)
    
    existing = list(base_set)
    existing_sums = set(_all_triple_sums(existing))
    existing_set = set(existing)
    
    for x in range(1, 4097):
        if x in existing_set:
            continue
        if _check_and_add(existing, x, existing_sums):
            existing_set.add(x)
    
    return existing


def _greedy_from_1():
    """Pure greedy from 1."""
    existing = []
    existing_sums = set()
    for x in range(1, 4097):
        _check_and_add(existing, x, existing_sums)
    return existing


def _greedy_with_step(rng, deadline, min_step=1, max_step=20):
    """Greedy with random step sizes."""
    selected = []
    existing_sums = set()
    
    x = rng.randint(1, min(50, 4096))
    while x <= 4096 and time.time() < deadline:
        _check_and_add(selected, x, existing_sums)
        x += rng.randint(min_step, max_step)
    
    return selected


def _greedy_with_adaptive_step(rng, deadline):
    """Greedy with adaptive step sizes based on density."""
    selected = []
    existing_sums = set()
    
    x = rng.randint(1, 100)
    step = rng.randint(3, 10)
    
    while x <= 4096 and time.time() < deadline:
        _check_and_add(selected, x, existing_sums)
        # Adapt step based on how many were added recently
        if selected and x > 0:
            density = len(selected) / max(1, x)
            if density > 0.01:
                step = max(2, int(step * 0.9))
            else:
                step = min(50, int(step * 1.1))
        x += step + rng.randint(0, 5)
    
    return selected


def _multi_seed_greedy(rng, deadline, num_seeds=20):
    """Try multiple random seeds for greedy construction."""
    best = []
    best_size = 0
    
    for seed in range(num_seeds):
        if time.time() > deadline:
            break
        
        local_rng = random.Random(seed * 1000 + 42)
        cand = _greedy_with_adaptive_step(local_rng, deadline)
        
        if len(cand) > best_size:
            best = list(cand)
            best_size = len(cand)
    
    return best


def _swap_improve(current, deadline, max_swaps=100):
    """
    Try to improve the set by removing one element and adding a better one.
    """
    current = list(current)
    current_set = set(current)
    best = list(current)
    best_size = len(current)
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        n = len(current)
        if n == 0:
            break
        
        existing_sums = set(_all_triple_sums(current))
        
        # Try removing each element and see if we can add a better one
        for idx in range(n):
            if time.time() > deadline:
                break
            
            removed = current[idx]
            temp = list(current)
            temp.pop(idx)
            temp.sort()
            temp_set = set(temp)
            temp_sums = set(_all_triple_sums(temp))
            
            # Try to find a value that can be added
            found = False
            for x in range(1, 4097):
                if x in temp_set:
                    continue
                if _check_and_add(temp, x, temp_sums):
                    if len(temp) > best_size:
                        current = list(temp)
                        current_set = set(current)
                        best = list(current)
                        best_size = len(current)
                    found = True
                    break
            
            if found:
                break
    
    return best


def _local_search(rng, initial_set, deadline, max_iterations=50):
    """
    Local search: alternate between greedy extension and swap improvement.
    """
    current = list(initial_set)
    best = list(current)
    best_size = len(current)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Phase 1: Greedy extension
        current_set = set(current)
        current_sums = set(_all_triple_sums(current))
        
        for x in range(1, 4097):
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            if _check_and_add(current, x, current_sums):
                current_set.add(x)
        
        if len(current) > best_size:
            best = list(current)
            best_size = len(current)
        
        # Phase 2: Swap improvement
        current = _swap_improve(current, deadline, max_swaps=20)
        
        if len(current) > best_size:
            best = list(current)
            best_size = len(current)
    
    return best


def _construct_with_polynomial_base():
    """
    Construct using a polynomial base approach.
    Use values that grow roughly as k^2 or k^3 to ensure spacing.
    """
    existing = []
    existing_sums = set()
    
    # Start with a geometric-like sequence
    for k in range(1, 100):
        x = k * k * 2  # quadratic growth
        if x > 4096:
            break
        _check_and_add(existing, x, existing_sums)
    
    # Then greedily fill in
    existing_set = set(existing)
    for x in range(1, 4097):
        if x in existing_set:
            continue
        if _check_and_add(existing, x, existing_sums):
            existing_set.add(x)
    
    return existing


def construct():
    """
    Find a large B_3[1] set in [1, 4096] where all unordered triple sums
    (allowing repetitions) are distinct.
    """
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    best = []
    best_size = 0
    
    # Strategy 1: Exponential base construction + greedy extension
    cand = _construct_b3_set()
    if _is_valid_triples(cand):
        best = list(cand)
        best_size = len(best)
    
    # Strategy 2: Pure greedy from 1
    if time.time() < deadline:
        cand = _greedy_from_1()
        if len(cand) > best_size and _is_valid_triples(cand):
            best = list(cand)
            best_size = len(cand)
    
    # Strategy 3: Multi-seed greedy with adaptive steps
    if time.time() < deadline:
        rng = random.Random(42)
        remaining = deadline - time.time()
        if remaining > 5:
            cand = _multi_seed_greedy(rng, start_time + remaining * 0.4, num_seeds=30)
            if len(cand) > best_size and _is_valid_triples(cand):
                best = list(cand)
                best_size = len(cand)
    
    # Strategy 4: Polynomial base construction
    if time.time() < deadline:
        cand = _construct_with_polynomial_base()
        if len(cand) > best_size and _is_valid_triples(cand):
            best = list(cand)
            best_size = len(cand)
    
    # Strategy 5: Local search improvement on best
    if len(best) > 0 and time.time() < deadline:
        rng = random.Random(777)
        remaining = deadline - time.time()
        if remaining > 10:
            cand = _local_search(rng, best, start_time + remaining * 0.7, max_iterations=30)
            if len(cand) > best_size and _is_valid_triples(cand):
                best = list(cand)
                best_size = len(cand)
    
    # Final validation
    if not _is_valid_triples(best):
        best = _construct_b3_set()
    
    return best


if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}")
    print(f"Set: {result[:20]}...")
    print(f"Valid: {_is_valid_triples(result)}")
