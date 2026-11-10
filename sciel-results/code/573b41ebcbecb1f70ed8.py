import time
import random

def construct():
    deadline = time.time() + 88

    def can_add_fast(element, others):
        """
        Check if element is the mean of any subset of 'others' (size >= 2).
        Uses bitset DP.
        """
        n = len(others)
        if n < 2:
            return True  # Can always add if not enough others to form mean
        
        max_sum = sum(others)
        if max_sum < 2 * element:
            return True
            
        # dp[j] is a bitset where bit s is set if sum s is achievable with j elements
        # We only care up to sum = n * element (actually max_sum is tighter usually)
        # But bitsets grow. Let's just do it.
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for val in others:
            for j in range(n, 0, -1):
                if dp[j - 1]:
                    dp[j] |= (dp[j - 1] << val)
        
        # Check if any k >= 2 has target sum k*element
        # Max k is n.
        # We can break early if k*element > max_sum
        for k in range(2, n + 1):
            target_sum = k * element
            if target_sum > max_sum:
                break
            if (dp[k] >> target_sum) & 1:
                return False
        return True

    def is_valid_set(s):
        """
        Verify the entire set satisfies the constraint:
        No element is the mean of any subset of at least two OTHER elements.
        """
        n = len(s)
        if n < 3:
            return True
        for i in range(n):
            element = s[i]
            others = s[:i] + s[i+1:]
            if not can_add_fast(element, others):
                return False
        return True

    def can_add_with_context(element, current_set):
        """
        Check if 'element' can be added to 'current_set'.
        1. 'element' is not mean of any subset of current_set.
        2. No existing element 'a' in current_set becomes mean of a subset involving 'element'.
        """
        if not current_set:
            return True
        
        # Check 1: Is 'element' the mean of any subset of 'current_set'?
        if not can_add_fast(element, current_set):
            return False
        
        # Check 2: Does adding 'element' make any existing element 'a' a mean?
        # If 'a' becomes a mean, the subset MUST include 'element'.
        # Let T be a subset of current_set \ {a}.
        # a = mean(T U {element}) => sum(T) + element = (|T| + 1) * a
        # sum(T) = (|T| + 1) * a - element
        # Let k = |T| + 1. Then sum(T) = k * a - element.
        # T has size k - 1.
        # Since the original subset for 'a' must have size >= 2, and it includes 'element',
        # the total size is k. So k >= 2 => |T| >= 1.
        
        n = len(current_set)
        # Pre-compute sums? No, we need specific subset sums.
        # We iterate through each 'a' in current_set.
        for i, a in enumerate(current_set):
            others_without_a = current_set[:i] + current_set[i+1:]
            if len(others_without_a) < 1:
                continue
            
            # We need to check if there exists a subset T of others_without_a
            # of size j (where j = k-1 >= 1) such that sum(T) = (j+1)*a - element
            
            m = len(others_without_a)
            # Bitset DP for others_without_a
            # We only need up to size m.
            # Target sums are (j+1)*a - element.
            # Max target sum is (m+1)*a - element.
            # Min target sum is 2*a - element.
            
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for val in others_without_a:
                for j in range(m, 0, -1):
                    if dp[j - 1]:
                        dp[j] |= (dp[j - 1] << val)
            
            # Check for any valid j
            for j in range(1, m + 1):
                target_sum = (j + 1) * a - element
                # target_sum must be achievable by j elements from others_without_a
                # Also target_sum must be >= 0 and <= sum(others_without_a)
                # And obviously, the min sum of j elements is sum of smallest j, max is sum of largest j.
                # The bitset check covers this.
                if target_sum < 0:
                    continue
                # Optimization: if target_sum > sum(others_without_a), break? 
                # sum(others_without_a) is fixed. target_sum increases with j.
                # So we can break if target_sum > max_possible_sum_for_j?
                # Actually, if target_sum > sum(others_without_a), it's impossible.
                # sum(others_without_a) is the total sum.
                if target_sum > sum(others_without_a):
                    break
                if (dp[j] >> target_sum) & 1:
                    return False
        
        return True

    # --- Search Strategies ---

    # Strategy 1: Greedy with diverse orderings
    all_nums = list(range(1, 513))
    
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0: return False
        return True

    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    
    p2 = [2**i for i in range(1, 10) if 2**i <= 512]
    non_p2 = [x for x in range(1, 513) if x not in set(p2)]
    
    alt = []
    for i in range(256):
        alt.append(i + 1)
        alt.append(512 - i)
        
    rng = random.Random(42)
    rnd = all_nums[:]
    rng.shuffle(rnd)
    
    mod5 = []
    for r in range(5):
        mod5.extend([x for x in range(1, 513) if x % 5 == r])
        
    orderings = [
        p2 + non_p2,
        primes + non_primes,
        alt,
        list(range(1, 513)),
        list(range(512, 0, -1)),
        mod5,
        rnd
    ]

    best_set = []
    
    for ordering in orderings:
        if time.time() > deadline - 5:
            break
            
        selected = []
        for x in ordering:
            if time.time() > deadline - 5:
                break
            if len(selected) >= 48:
                break
            if can_add_with_context(x, selected):
                selected.append(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) == 48:
                return best_set

    # Strategy 2: Local Search / Hill Climbing
    # Try to improve the size of the set by swapping or adding.
    # Since we want to maximize size, we primarily want to ADD elements.
    # If we are stuck at size K < 48, we try to find an element x not in the set
    # such that adding it doesn't violate constraints.
    # If we can't add any single element, we might try removing one element and adding two?
    # Or removing one and adding one (if the new one allows more additions later)?
    # Let's stick to: 
    # 1. Try to add any remaining element.
    # 2. If that fails, try random restarts with more randomness.

    if len(best_set) < 48:
        current = best_set[:]
        current_set = set(current)
        remaining = [x for x in range(1, 513) if x not in current_set]
        
        # Try to add more if possible (should have been caught by greedy, but maybe order matters)
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            for x in remaining:
                if time.time() > deadline - 10:
                    break
                if len(current) >= 48:
                    break
                if can_add_with_context(x, current):
                    current.append(x)
                    current_set.add(x)
                    remaining.remove(x)
                    improved = True
                    break # Restart loop with new state
            if len(current) > len(best_set):
                best_set = current[:]
                if len(best_set) == 48:
                    return best_set

    # If still < 48, try random restarts with different seeds and more aggressive mixing
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed * 1000 + 1)
        order = list(range(1, 513))
        rng.shuffle(order)
        
        # Interleave with some structured orders?
        # Let's just use random order for now, but maybe bias towards numbers that are "hard" to be means?
        # e.g. large numbers? Small numbers?
        # Actually, random is fine.
        
        selected = []
        for x in order:
            if time.time() > deadline - 5:
                break
            if len(selected) >= 48:
                break
            if can_add_with_context(x, selected):
                selected.append(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) == 48:
                return best_set

    # Final check
    if best_set and not is_valid_set(best_set):
        # Fallback to a known valid small set if verification fails (should not happen)
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set
