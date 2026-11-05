import time
import random
import sys

def _is_valid_fast(arr):
    """
    Fast validity check for a list of distinct integers.
    Returns True if no element is the mean of any subset of >=2 other elements.
    Uses bitset DP for subset sums.
    """
    n = len(arr)
    if n < 3:
        return True
    
    # Sort for potential early termination or just consistency
    arr = sorted(arr)
    
    # Precompute total sum
    total_sum = sum(arr)
    
    # For each element a[i], check if it is the mean of any subset of the others.
    # Let S be the set of others. We need to check if there exists a subset T of S
    # with |T| >= 2 such that sum(T) = |T| * a[i].
    
    # Optimization: If a[i] is the smallest or largest, it might be easier to bound.
    # But we must check all.
    
    for i in range(n):
        a = arr[i]
        
        # The subset must consist of elements from arr excluding arr[i].
        # Let others = arr[:i] + arr[i+1:]
        
        # We need to check if sum(T) = k * a for any k >= 2.
        # Max possible sum of any subset of others is sum(others).
        # Min possible sum of a subset of size k is sum of k smallest in others.
        
        # DP approach:
        # dp[k] is a bitset where bit j is 1 if sum j is achievable by a subset of size k.
        # We only care about sums that are multiples of 'a'.
        
        others = arr[:i] + arr[i+1:]
        m = len(others)
        
        if m < 2:
            continue
            
        max_sum = sum(others)
        
        # We only need to check k such that k*a <= max_sum.
        # Also k <= m.
        max_k = min(m, max_sum // a)
        
        if max_k < 2:
            continue
            
        # Initialize DP
        # dp[0] = 1 (sum 0 with size 0)
        # dp[k] for k > 0 initially 0
        dp = [0] * (max_k + 1)
        dp[0] = 1
        
        for x in others:
            # Update dp in reverse order of k to avoid using x multiple times
            # We only need to update up to max_k
            for k in range(max_k, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
                    
            # Early check? Not really, we need the full DP to know if a specific sum is reachable.
            # But we can prune k if k*a > current max possible sum? No, max_sum is fixed.
            
        # Check if any dp[k] has a bit set at position k*a for k >= 2
        for k in range(2, max_k + 1):
            target = k * a
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Initial state: A known valid set or a random one.
    # Let's start with a greedy construction to get a decent baseline.
    
    def greedy_init():
        current = []
        current_set = set()
        # Try adding numbers in a specific order
        # Using powers of 2 and 3 is a good heuristic for Sidon-like properties
        candidates = []
        for i in range(10):
            for j in range(10):
                v = (2**i) * (3**j)
                if v <= 512:
                    candidates.append(v)
        candidates = sorted(set(candidates))
        
        # Add random numbers to fill gaps
        for x in range(1, 513):
            if x not in set(candidates):
                candidates.append(x)
        random.shuffle(candidates[20:]) # Keep structured ones at front
        
        for x in candidates:
            if len(current) >= 48:
                break
            if x in current_set:
                continue
            # Fast check: try to add
            test_set = current + [x]
            # We don't want to run full _is_valid_fast on every step if it's slow.
            # But for init, let's just be greedy and check at the end?
            # No, we need a valid start.
            # Let's use a simpler check for init: just check pairwise means?
            # No, the problem forbids subset means.
            # Greedy with full check is slow but n is small (<=48).
            # _is_valid_fast is O(n^2 * 2^n)? No, DP is O(n * max_sum) per element.
            # max_sum ~ 20000. n=48. 48 * 20000 * 48 ~ 46M ops per check.
            # In Python, this is too slow for 48 steps * 512 candidates.
            
            # Alternative: Just pick a known valid set?
            # Or use a faster incremental check?
            # Let's just start with a random set of size 20 and expand?
            
            # Actually, let's just use a random valid set from a previous run if we had one.
            # Since we don't, let's construct one carefully.
            
            # Heuristic: Numbers with distinct prime factorizations are less likely to form arithmetic progressions.
            # Let's just try to add x if _is_valid_fast(current + [x]) is True.
            # To speed this up, we can precompute some things or just accept it might take a few seconds.
            
            if _is_valid_fast(test_set):
                current.append(x)
                current_set.add(x)
                
        return current

    best_set = greedy_init()
    best_len = len(best_set)
    
    # Simulated Annealing
    # State: a list of integers.
    # We will try to swap elements or replace elements.
    # Objective: Maximize len(set) subject to validity.
    # Penalty for invalidity? Or just reject invalid moves?
    # If we reject invalid moves, we are stuck in a local optimum of valid sets.
    # To escape, we might need to allow temporary invalidity?
    # But the verifier only accepts valid sets.
    # So we must maintain validity.
    # This is a constrained optimization.
    # Move: Remove one element, add another.
    # If the new set is valid and same size, accept.
    # If the new set is valid and larger size (impossible with 1-for-1 swap), accept.
    # To increase size, we need to find an element that can be added without removing one?
    # That's the "expand" move.
    
    # Let's alternate between:
    # 1. Expansion: Try to add a new element to the set (size increases by 1).
    # 2. Swap: Replace an element with another to "free up" space or improve structure for future expansion.
    
    current_set = best_set[:]
    current_len = best_len
    temperature = 100.0
    cooling_rate = 0.99
    
    all_nums = set(range(1, 513))
    
    while time.time() < deadline:
        if temperature < 1.0:
            temperature = 1.0
            
        # Choose move type
        # If we can expand, try to expand.
        # Else, try to swap to find a better configuration that allows expansion.
        
        # 1. Try to expand
        if current_len < 48:
            # Pick a random candidate not in set
            candidates = list(all_nums - set(current_set))
            if candidates:
                # Prioritize candidates that are "far" from existing elements?
                # Or just random.
                x = random.choice(candidates)
                test_set = current_set + [x]
                if _is_valid_fast(test_set):
                    current_set = test_set
                    current_len += 1
                    if current_len > best_len:
                        best_set = current_set[:]
                        best_len = current_len
                        # Reset temperature?
                        temperature = 100.0
                    continue
        
        # 2. Swap
        # Remove a random element, add a random different element
        if current_len > 0:
            idx_remove = random.randint(0, current_len - 1)
            x_remove = current_set[idx_remove]
            
            candidates = list(all_nums - set(current_set))
            if candidates:
                x_add = random.choice(candidates)
                
                # Construct new set
                test_set = current_set[:idx_remove] + current_set[idx_remove+1:] + [x_add]
                random.shuffle(test_set) # Keep order random to avoid bias? Not strictly necessary for set logic
                
                if _is_valid_fast(test_set):
                    # Accept swap if it improves "potential" or just to explore
                    # Since size is same, we accept to escape local minima.
                    # We can use a simple acceptance criterion: always accept if valid.
                    current_set = test_set
                    # current_len stays same
                    
                    # Occasionally, try to expand after a swap
                    if random.random() < 0.1 and current_len < 48:
                        candidates_exp = list(all_nums - set(current_set))
                        if candidates_exp:
                            x_exp = random.choice(candidates_exp)
                            test_exp = current_set + [x_exp]
                            if _is_valid_fast(test_exp):
                                current_set = test_exp
                                current_len += 1
                                if current_len > best_len:
                                    best_set = current_set[:]
                                    best_len = current_len
                                temperature = 100.0
                else:
                    # Reject invalid swap
                    pass
        
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1

    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _is_valid_fast(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
