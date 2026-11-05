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


def _randomized_greedy(rng, deadline, max_attempts=2000):
    """Randomized greedy with varying step patterns."""
    best = []
    best_size = 0
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        selected = []
        existing_sums = set()
        step_base = rng.randint(1, 12)
        
        x = rng.randint(1, min(30, 4096))
        while x <= 4096:
            _check_and_add(selected, x, existing_sums)
            x += step_base + rng.randint(0, 8)
        
        if len(selected) > best_size:
            best = list(selected)
            best_size = len(selected)
    
    return best


def _greedy_with_perturbation(rng, deadline, max_attempts=1500):
    """Greedy with random perturbations."""
    best = []
    best_size = 0
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        selected = []
        existing_sums = set()
        start = rng.randint(1, 150)
        min_step = rng.randint(2, 15)
        
        x = start
        while x <= 4096:
            _check_and_add(selected, x, existing_sums)
            x += min_step + rng.randint(0, min_step + 3)
        
        if len(selected) > best_size:
            best = list(selected)
            best_size = len(selected)
    
    return best


def _backtracking_search(rng, deadline, max_depth=50, max_branches=5):
    """
    Backtracking search with MRV-like heuristic.
    At each level, try the most constrained candidates first.
    """
    best = []
    best_size = 0
    
    def backtrack(current, current_sums, depth, remaining_time):
        nonlocal best, best_size
        
        if time.time() > remaining_time:
            return
        
        if len(current) > best_size:
            best = list(current)
            best_size = len(current)
        
        if depth >= max_depth or len(current) >= 80:
            return
        
        # Find candidates: try values that have fewest conflicts
        # Use a heuristic: prefer values that are spaced out
        candidates = []
        n = len(current)
        
        # Generate candidate positions based on current set
        if n == 0:
            candidates = list(range(1, 50))
        else:
            # Try gaps in the current set
            max_val = current[-1] if current else 0
            min_val = current[0] if current else 1
            
            # Try values in gaps
            for gap_start in range(1, 4097, max(1, (4096 - 1) // 20)):
                for offset in range(0, 20):
                    x = gap_start + offset
                    if 1 <= x <= 4096 and x not in set(current):
                        candidates.append(x)
            
            # Also try extending beyond max
            for x in range(max_val + 1, min(max_val + 100, 4097)):
                candidates.append(x)
        
        # Shuffle candidates for variety
        rng.shuffle(candidates)
        
        # Try top candidates
        tried = 0
        for x in candidates:
            if tried >= max_branches:
                break
            if time.time() > remaining_time:
                break
            if x in set(current):
                continue
            
            # Check if x can be added
            new_sums = _new_triple_sums(current, x)
            collision = False
            for s in new_sums:
                if s in current_sums:
                    collision = True
                    break
            
            if not collision:
                tried += 1
                current.append(x)
                current.sort()
                current_sums.update(new_sums)
                
                backtrack(current, current_sums, depth + 1, remaining_time)
                
                # Undo
                current.pop()
                current.sort()
                current_sums.difference_update(new_sums)
        
        # Also try simple greedy extension from current state
        if time.time() < remaining_time:
            temp = list(current)
            temp_sums = set(current_sums)
            for x in range(1, 4097):
                if x in set(temp):
                    continue
                if _check_and_add(temp, x, temp_sums):
                    pass
            if len(temp) > best_size:
                best = list(temp)
                best_size = len(temp)
    
    backtrack([], set(), 0, deadline)
    return best


def _local_search_improve(rng, initial_set, deadline, max_iterations=500):
    """
    Local search: try to improve the set by swapping elements.
    """
    current = list(initial_set)
    current_set = set(current)
    best = list(current)
    best_size = len(current)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        existing_sums = set(_all_triple_sums(current))
        n = len(current)
        
        # Try adding any element
        added = False
        for x in range(1, 4097):
            if x in current_set:
                continue
            if _check_and_add(current, x, existing_sums):
                current_set.add(x)
                added = True
                break
        
        if added:
            if len(current) > best_size:
                best = list(current)
                best_size = len(current)
            continue
        
        # Try swapping: remove one, add another
        if n > 0:
            improved = False
            for idx in range(n):
                if time.time() > deadline:
                    break
                
                removed = current[idx]
                temp = list(current)
                temp.pop(idx)
                temp.sort()
                temp_sums = set(_all_triple_sums(temp))
                
                for x in range(1, 4097):
                    if x in set(temp):
                        continue
                    if _check_and_add(temp, x, temp_sums):
                        if len(temp) > best_size:
                            current = list(temp)
                            current_set = set(current)
                            best = list(current)
                            best_size = len(current)
                        improved = True
                        break
                
                if improved:
                    break
    
    return best


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
            best_size = len(best)
    
    # Strategy 3: Randomized greedy
    if time.time() < deadline:
        rng = random.Random(42)
        cand = _randomized_greedy(rng, deadline - (time.time() - start_time))
        if len(cand) > best_size and _is_valid_triples(cand):
            best = list(cand)
            best_size = len(best)
    
    # Strategy 4: Greedy with perturbation
    if time.time() < deadline:
        rng = random.Random(12345)
        cand = _greedy_with_perturbation(rng, deadline - (time.time() - start_time))
        if len(cand) > best_size and _is_valid_triples(cand):
            best = list(cand)
            best_size = len(best)
    
    # Strategy 5: Backtracking search
    if time.time() < deadline:
        rng = random.Random(999)
        remaining = deadline - time.time()
        if remaining > 5:
            cand = _backtracking_search(rng, start_time + remaining * 0.7, max_depth=40, max_branches=3)
            if len(cand) > best_size and _is_valid_triples(cand):
                best = list(cand)
                best_size = len(best)
    
    # Strategy 6: Local search improvement on best
    if len(best) > 0 and time.time() < deadline:
        rng = random.Random(777)
        remaining = deadline - time.time()
        if remaining > 3:
            cand = _local_search_improve(rng, best, start_time + remaining * 0.8, max_iterations=300)
            if len(cand) > best_size and _is_valid_triples(cand):
                best = list(cand)
                best_size = len(best)
    
    # Final validation
    if not _is_valid_triples(best):
        best = _construct_b3_set()
    
    return best


if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}")
    print(f"Set: {result[:20]}...")
    print(f"Valid: {_is_valid_triples(result)}")
