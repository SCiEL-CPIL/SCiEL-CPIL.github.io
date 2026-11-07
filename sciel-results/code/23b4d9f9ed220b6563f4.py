"""Rewritten program for divisibility-incomparable subset sums task."""
import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    best_set = [31, 32, 34]
    best_size = 3
    rng = random.Random(42)
    
    def get_all_subset_sums(nums):
        """Get all nonempty subset sums as a set."""
        sums = set()
        n = len(nums)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += nums[i]
            sums.add(s)
        return sums
    
    def check_divisibility(sums_set):
        """Check if no two distinct sums divide one another."""
        sums_list = sorted(sums_set)
        for i in range(len(sums_list)):
            for j in range(i + 1, len(sums_list)):
                a, b = sums_list[i], sums_list[j]
                if b % a == 0:
                    return False
        return True
    
    def is_valid(nums):
        """Check if the set satisfies both constraints."""
        sums = get_all_subset_sums(nums)
        # Check distinctness (automatically satisfied by set, but check count)
        # Actually, if two subsets have same sum, the set will have fewer elements than 2^n - 1
        n = len(nums)
        if len(sums) != (1 << n) - 1:
            return False
        # Check divisibility
        return check_divisibility(sums)
    
    def try_greedy(start_set, rng_instance):
        """Try to grow the set greedily."""
        current = list(start_set)
        # Get all current subset sums
        current_sums = get_all_subset_sums(current)
        
        # Precompute: for each candidate, check if it can be added
        # We need to try all possible next elements
        candidates = [x for x in range(1, 257) if x not in current]
        rng_instance.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            # New sums would be: old_sums, old_sums + cand, and cand
            # Actually: new_sums = old_sums ∪ {s + cand for s in old_sums} ∪ {cand}
            # But we also need to check that cand itself doesn't conflict
            
            new_sums = set(current_sums)
            new_sums.add(cand)
            for s in current_sums:
                new_sums.add(s + cand)
            
            # Check distinctness: new_sums should have size = len(current_sums) + len(current_sums) = 2*len(current_sums)
            # Wait, that's not right. Let me think again.
            # Old sums: S (size = 2^n - 1)
            # New sums: S ∪ (S + cand) ∪ {cand}
            # Size should be 2^(n+1) - 1 = 2 * (2^n - 1) + 1 = 2*|S| + 1
            expected_size = 2 * len(current_sums) + 1
            if len(new_sums) != expected_size:
                continue
            
            # Check divisibility on new_sums
            if not check_divisibility(new_sums):
                continue
            
            # Valid! Add it
            current.append(cand)
            current_sums = new_sums
        
        return current
    
    # Try multiple random restarts with different strategies
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Strategy 1: Start with a random small set and grow
        # Use superincreasing-like sequences to help with distinctness
        start_size = rng.randint(1, 4)
        start_set = rng.sample(range(1, 257), start_size)
        
        result = try_greedy(start_set, rng)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            # Try to improve further with more greedy passes
            for _ in range(5):
                if time.time() > deadline:
                    break
                improved = try_greedy(best_set, rng)
                if len(improved) > len(best_set):
                    best_set = improved
                    best_size = len(best_set)
    
    # Also try some structured approaches
    # Superincreasing sequences
    for base in range(2, 8):
        if time.time() > deadline:
            break
        seq = []
        total = 0
        for i in range(1, 20):
            val = total + base ** i
            if val > 256:
                break
            seq.append(val)
            total += val
        if len(seq) > best_size and is_valid(seq):
            best_set = seq
            best_size = len(seq)
    
    # Try specific known good patterns
    # Powers of 2 shifted
    for shift in range(1, 10):
        if time.time() > deadline:
            break
        seq = [shift * (2 ** i) for i in range(1, 10) if shift * (2 ** i) <= 256]
        if len(seq) > best_size and is_valid(seq):
            best_set = seq
            best_size = len(seq)
    
    # Ensure we return a valid result
    if not is_valid(best_set):
        # Fallback to a known valid small set
        best_set = [31, 32, 34]
    
    # Final validation and type checking
    return [int(x) for x in best_set]

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
    print(f"Valid: {is_valid(result)}")
