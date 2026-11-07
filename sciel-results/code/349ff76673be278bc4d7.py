import time
import random
import math

def _compute_sums(nums):
    """Compute all non-empty subset sums."""
    n = len(nums)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s in sums:
            return None  # duplicate
        sums.add(s)
    return sums

def _check_divisibility(sums):
    """Check if any two distinct sums divide each other. Returns True if valid (no division)."""
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        a = sorted_sums[i]
        for j in range(i + 1, len(sorted_sums)):
            b = sorted_sums[j]
            if b % a == 0:
                return False
    return True

def _is_valid(nums):
    """Full validity check."""
    sums = _compute_sums(nums)
    if sums is None:
        return False
    return _check_divisibility(sums)

def _try_add(nums, sums, cand):
    """Try adding cand to nums with current sums. Returns (new_sums, valid) or (None, False)."""
    if cand in nums:
        return None, False
    new_sums = set(sums)
    new_sums.add(cand)
    for s in sums:
        ns = s + cand
        if ns in new_sums:
            return None, False
        new_sums.add(ns)
    if not _check_divisibility(new_sums):
        return None, False
    return new_sums, True

def _try_remove(nums, sums, idx):
    """Remove element at idx from nums with current sums. Returns (new_sums, valid) or (None, False)."""
    cand = nums[idx]
    new_nums = nums[:idx] + nums[idx+1:]
    # Recompute sums from scratch (simpler and still fast for small sets)
    new_sums = _compute_sums(new_nums)
    if new_sums is None:
        return None, False
    if not _check_divisibility(new_sums):
        return None, False
    return new_sums, True

def _greedy_grow(start_nums, deadline):
    """Greedy: keep adding elements that maintain validity."""
    nums = list(start_nums)
    sums = _compute_sums(nums)
    if sums is None or not _check_divisibility(sums):
        return nums
    
    candidates = [x for x in range(1, 257) if x not in nums]
    random.shuffle(candidates)
    
    while time.time() < deadline:
        added = False
        random.shuffle(candidates)
        for cand in candidates:
            if cand in nums:
                continue
            new_sums, valid = _try_add(nums, sums, cand)
            if valid:
                nums.append(cand)
                sums = new_sums
                candidates = [x for x in range(1, 257) if x not in nums]
                added = True
                break
        if not added:
            break
    return nums

def _anneal(start_nums, deadline, initial_temp=10.0, cooling=0.95):
    """Simulated annealing: accept worse moves to escape local optima."""
    nums = list(start_nums)
    sums = _compute_sums(nums)
    if sums is None:
        return nums
    
    # Score: size of set (higher is better), with penalty for invalidity
    def score(state):
        n = len(state)
        s = _compute_sums(state)
        if s is None:
            return -100
        if not _check_divisibility(s):
            return -50
        return n
    
    best_nums = list(nums)
    best_sums = set(sums)
    best_score = score(nums)
    temp = initial_temp
    
    while time.time() < deadline and temp > 0.1:
        # Generate a move: add, remove, or swap
        move_type = random.random()
        new_nums = list(nums)
        
        if move_type < 0.4 and len(nums) < 30:
            # Add a random element
            candidates = [x for x in range(1, 257) if x not in nums]
            if candidates:
                cand = random.choice(candidates)
                new_nums.append(cand)
        elif move_type < 0.7 and len(nums) > 1:
            # Remove a random element
            idx = random.randint(0, len(nums) - 1)
            new_nums.pop(idx)
        else:
            # Swap: remove one, add another
            if len(nums) > 1:
                idx = random.randint(0, len(nums) - 1)
                removed = new_nums.pop(idx)
                candidates = [x for x in range(1, 257) if x not in new_nums and x != removed]
                if candidates:
                    cand = random.choice(candidates)
                    new_nums.append(cand)
        
        new_score = score(new_nums)
        delta = new_score - score(nums)
        
        if delta >= 0 or random.random() < math.exp(delta / temp):
            nums = new_nums
            if new_score > best_score:
                best_score = new_score
                best_nums = list(nums)
                best_sums = _compute_sums(nums)
        
        temp *= cooling
    
    return best_nums

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s
    
    random.seed(42)
    
    best_set = [31, 32, 34]
    best_size = 3
    
    # Phase 1: Structured greedy from multiple seeds
    # Use numbers in a narrow high range to reduce divisibility conflicts
    seed_ranges = [
        (128, 256), (100, 200), (80, 160), (64, 128), (50, 100)
    ]
    
    for lo, hi in seed_ranges:
        if time.time() > deadline - 10:
            break
        # Create initial seed: pick 2-3 numbers from range
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            seed_size = random.randint(2, 3)
            seed = random.sample(range(lo, hi + 1), seed_size)
            result = _greedy_grow(seed, deadline - 5)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
                print(f"Greedy found size {best_size}: {sorted(best_set)}")
    
    # Phase 2: Simulated annealing from best found and random starts
    if time.time() < deadline - 15:
        # Anneal from current best
        result = _anneal(best_set, deadline - 10, initial_temp=5.0, cooling=0.97)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            print(f"Anneal from best found size {best_size}: {sorted(best_set)}")
    
    # Phase 3: More annealing from random starts
    remaining = deadline - time.time()
    if remaining > 5:
        num_starts = min(20, int(remaining / 3))
        for i in range(num_starts):
            if time.time() > deadline - 3:
                break
            # Random seed in high range
            seed_size = random.randint(2, 4)
            seed = random.sample(range(60, 257), seed_size)
            # Quick greedy first
            greedy_result = _greedy_grow(seed, deadline - 3)
            # Then anneal
            result = _anneal(greedy_result, deadline - 2, initial_temp=3.0, cooling=0.96)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
                print(f"Anneal {i} found size {best_size}: {sorted(best_set)}")
    
    # Final greedy polish on best
    if time.time() < deadline - 1:
        polished = _greedy_grow(best_set, deadline - 0.5)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = polished
    
    return [int(x) for x in best_set]
