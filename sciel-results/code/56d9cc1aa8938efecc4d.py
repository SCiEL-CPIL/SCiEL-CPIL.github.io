"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

P = {'n': 256, 'maximum_size': 30}

def _subset_sums_valid(sums_set, existing_sums):
    """Check if adding sums_set to existing_sums maintains divisibility-incomparability."""
    # For each new sum, check against all existing sums
    for s in sums_set:
        if s in existing_sums:
            return False
        # Check divisibility with existing
        for e in existing_sums:
            if s > e:
                if s % e == 0:
                    return False
            elif e > s:
                if e % s == 0:
                    return False
    # Check divisibility within sums_set
    ss = list(sums_set)
    for i in range(len(ss)):
        for j in range(i+1, len(ss)):
            a, b = ss[i], ss[j]
            if a == b:
                continue
            if a > b:
                if a % b == 0:
                    return False
            else:
                if b % a == 0:
                    return False
    return True

def _compute_subset_sums(arr):
    """Compute all non-empty subset sums of arr as a set."""
    sums = set()
    for x in arr:
        new_sums = set()
        for s in sums:
            ns = s + x
            new_sums.add(ns)
            if ns in sums:
                # duplicate sum found - invalid
                return None
        if new_sums & sums:
            return None
        sums |= new_sums
    return sums

def _is_valid(arr):
    """Check if arr satisfies all constraints."""
    if len(arr) != len(set(arr)):
        return False
    if any(x < 1 or x > 256 for x in arr):
        return False
    sums = _compute_subset_sums(arr)
    if sums is None:
        return False
    # Check divisibility incomparability
    ss = sorted(sums)
    for i in range(len(ss)):
        for j in range(i+1, len(ss)):
            if ss[j] % ss[i] == 0:
                return False
    return True

def _greedy_build():
    """Build a valid set using greedy approach."""
    arr = []
    sums = set()
    for candidate in range(1, 257):
        if candidate in arr:
            continue
        new_sums = set()
        valid = True
        for s in sums:
            ns = s + candidate
            if ns in sums or ns in new_sums:
                valid = False
                break
            new_sums.add(ns)
        if not valid:
            continue
        # Check divisibility with all existing sums and new sums
        all_sums = sums | new_sums
        ss = sorted(all_sums)
        for i in range(len(ss)):
            for j in range(i+1, len(ss)):
                if ss[j] % ss[i] == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            arr.append(candidate)
            sums = all_sums
    return arr

def _local_search(arr, max_iters=50000, time_limit=85.0):
    """Local search to improve set size."""
    start_time = time.time()
    best = list(arr)
    best_len = len(best)
    
    # Precompute valid candidates
    all_candidates = list(range(1, 257))
    
    current = list(arr)
    current_sums = _compute_subset_sums(current)
    if current_sums is None:
        return best
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try to add a random element
        if len(current) < 30:
            candidates = [c for c in all_candidates if c not in current]
            random.shuffle(candidates)
            added = False
            for cand in candidates[:50]:  # limit tries per iteration
                new_arr = current + [cand]
                new_sums = _compute_subset_sums(new_arr)
                if new_sums is None:
                    continue
                # Check divisibility
                valid = True
                ss = sorted(new_sums)
                for i in range(len(ss)):
                    for j in range(i+1, len(ss)):
                        if ss[j] % ss[i] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    current = new_arr
                    current_sums = new_sums
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(best)
                    added = True
                    break
            if added:
                continue
        
        # Try swap: remove one, add another
        if len(current) >= 2:
            idx_to_remove = random.randint(0, len(current)-1)
            candidates = [c for c in all_candidates if c not in current]
            random.shuffle(candidates)
            for cand in candidates[:30]:
                new_arr = [x for i, x in enumerate(current) if i != idx_to_remove] + [cand]
                new_arr.sort()
                new_sums = _compute_subset_sums(new_arr)
                if new_sums is None:
                    continue
                valid = True
                ss = sorted(new_sums)
                for i in range(len(ss)):
                    for j in range(i+1, len(ss)):
                        if ss[j] % ss[i] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    current = new_arr
                    current_sums = new_sums
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(best)
                    break
    
    return best

def _simulated_annealing(arr, time_limit=85.0):
    """Simulated annealing to improve set size."""
    start_time = time.time()
    all_candidates = list(range(1, 257))
    
    def get_sums(arr):
        return _compute_subset_sums(arr)
    
    def is_valid_arr(arr):
        sums = get_sums(arr)
        if sums is None:
            return False
        ss = sorted(sums)
        for i in range(len(ss)):
            for j in range(i+1, len(ss)):
                if ss[j] % ss[i] == 0:
                    return False
        return True
    
    current = list(arr)
    current_sums = get_sums(current)
    if current_sums is None:
        return arr
    
    best = list(current)
    best_len = len(best)
    temperature = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        move_type = random.random()
        new_arr = list(current)
        
        if move_type < 0.4 and len(new_arr) < 30:
            # Add element
            candidates = [c for c in all_candidates if c not in new_arr]
            if candidates:
                new_arr.append(random.choice(candidates))
        elif move_type < 0.8 and len(new_arr) > 1:
            # Remove element
            idx = random.randint(0, len(new_arr)-1)
            new_arr.pop(idx)
        else:
            # Swap
            if len(new_arr) > 0:
                idx = random.randint(0, len(new_arr)-1)
                candidates = [c for c in all_candidates if c not in new_arr]
                if candidates:
                    new_arr[idx] = random.choice(candidates)
        
        new_sums = get_sums(new_arr)
        if new_sums is None:
            continue
        
        # Check divisibility
        valid = True
        ss = sorted(new_sums)
        for i in range(len(ss)):
            for j in range(i+1, len(ss)):
                if ss[j] % ss[i] == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Valid move - accept
            current = new_arr
            current_sums = new_sums
            if len(current) > best_len:
                best = list(current)
                best_len = len(best)
        else:
            # Invalid - maybe accept based on temperature
            # Penalty: count of violations
            penalty = 0
            for i in range(len(ss)):
                for j in range(i+1, len(ss)):
                    if ss[j] % ss[i] == 0:
                        penalty += 1
            # Also penalty for duplicate sums (already handled by None)
            if penalty > 0:
                # Reject invalid moves in this simplified version
                # In full SA we'd accept with probability exp(-penalty/temperature)
                pass
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    return best

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Phase 1: Greedy build
    arr = _greedy_build()
    
    # Phase 2: Local search
    if time.time() - start_time < deadline - 5:
        arr = _local_search(arr, max_iters=30000, time_limit=deadline - 5 - (time.time() - start_time))
    
    # Phase 3: Simulated annealing from best
    if time.time() - start_time < deadline - 2:
        sa_arr = _simulated_annealing(arr, time_limit=deadline - 2 - (time.time() - start_time))
        if len(sa_arr) > len(arr):
            arr = sa_arr
    
    # Final validation
    if not _is_valid(arr):
        # Fallback to a known valid small set
        arr = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in arr]

# EVOLVE-BLOCK-END
