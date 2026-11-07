import time
import random
from itertools import combinations

def _subset_sums_valid(nums):
    """Check if all non-empty subset sums are distinct and no two divide each other."""
    n = len(nums)
    # Use a set to track all subset sums
    sums = set()
    # Generate all non-empty subset sums
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.add(s)
    # Check: no two distinct sums divide each other
    sorted_sums = sorted(sums)
    for i, a in enumerate(sorted_sums):
        for j in range(i + 1, len(sorted_sums)):
            b = sorted_sums[j]
            if b % a == 0:
                return False
    return True

def _subset_sums_valid_fast(nums):
    """Faster check using incremental subset sum generation."""
    n = len(nums)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s in sums:
            return False  # duplicate sum means invalid
        sums.add(s)
    # Check divisibility
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        a = sorted_sums[i]
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    best_set = [31, 32, 34]  # fallback
    best_size = 3
    
    random.seed(42)
    
    # Strategy: try to build larger sets using numbers in a "safe" range
    # Key insight: if all subset sums fall in (M/2, M] for some M, no division possible
    # Better: use numbers such that all subset sums are pairwise non-dividing
    
    # Approach 1: Greedy with validation, trying to add elements
    # Approach 2: Random search with local improvement
    
    # Let's try a structured approach: pick numbers from a range where
    # subset sums are "spread out" enough
    
    # Try using numbers in [128, 256] - then single element sums are large
    # But subset of 2 elements can be up to 512, and 128 divides 256, 128 divides 384, etc.
    # So we need to be more careful.
    
    # Better: use a set where all elements are > sum of all smaller elements
    # This ensures all subset sums are distinct (like superincreasing)
    # But we also need no divisibility.
    
    # Let's do randomized search with hill climbing
    
    def try_build_from_seed(seed_nums, max_iter=5000):
        """Try to extend a seed set by adding random candidates."""
        current = list(seed_nums)
        current_sums = set()
        # Compute current subset sums
        n = len(current)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += current[i]
            current_sums.add(s)
        
        # Check current validity
        sorted_sums = sorted(current_sums)
        for i in range(len(sorted_sums)):
            for j in range(i+1, len(sorted_sums)):
                if sorted_sums[j] % sorted_sums[i] == 0:
                    break
            else:
                continue
            break
        else:
            pass  # valid
        # Actually let me just use the full check
        
        candidates = [x for x in range(1, 257) if x not in current]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return current
            random.shuffle(candidates)
            added = False
            for cand in candidates:
                new_set = current + [cand]
                if _subset_sums_valid_fast(new_set):
                    current = new_set
                    candidates = [x for x in range(1, 257) if x not in current]
                    added = True
                    break
            if not added:
                break
        return current
    
    # Try multiple random seeds
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random seed of 2-3 elements
        seed_size = random.randint(2, 3)
        seed = random.sample(range(1, 257), seed_size)
        
        result = try_build_from_seed(seed, max_iter=2000)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            print(f"Found size {best_size}: {sorted(best_set)}")
    
    # Also try structured approaches
    # Use numbers that are all odd and in a high range
    # If all numbers are odd, subset sums have various parities
    # Not obviously helpful
    
    # Try: all numbers in [100, 256], and check
    if time.time() < deadline - 5:
        # Greedy: start empty, add best candidate
        current = []
        for _ in range(30):
            best_cand = None
            for cand in range(256, 0, -1):
                if cand in current:
                    continue
                new_set = current + [cand]
                if len(new_set) > 12:  # too slow to check
                    break
                if _subset_sums_valid_fast(new_set):
                    best_cand = cand
                    break
            if best_cand is None:
                break
            current.append(best_cand)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
            print(f"Greedy found size {best_size}: {sorted(best_set)}")
    
    return [int(x) for x in best_set]
