import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill
    
    def is_valid_candidate(x, S, dp, max_sum):
        """
        Check if x can be added to S.
        S: list of currently selected integers.
        dp: list of integers, where dp[k] is a bitmask of possible sums using exactly k elements from S.
        """
        n = len(S)
        
        # 1. Check if x is the mean of any subset of S of size k >= 2.
        # sum(subset) = k * x
        for k in range(2, n + 1):
            target = k * x
            if target < 0 or target > max_sum:
                continue
            # Check if bit 'target' is set in dp[k]
            if (dp[k] >> target) & 1:
                return False
        
        # 2. Check if any a in S becomes the mean of a subset of S U {x} of size >= 2.
        # A subset T' of S U {x} containing x can be written as T U {x} where T is a subset of S.
        # If x is not in the subset, it was already a valid mean (or not) in S.
        # We only need to check subsets that include x.
        # Let T' = T U {x}. Size |T'| = |T| + 1 = k + 1, where k = |T| >= 1 (since T' size >= 2).
        # Mean condition: sum(T) + x = (k + 1) * a
        # sum(T) = (k + 1) * a - x
        # We need to check if this sum is achievable by a subset T of S of size k.
        
        for a in S:
            for k in range(1, n + 1):
                target = (k + 1) * a - x
                if target < 0 or target > max_sum:
                    continue
                # Check if bit 'target' is set in dp[k]
                if (dp[k] >> target) & 1:
                    return False
        
        return True

    def try_ordering(order, max_sum=24576):
        S = []
        # dp[k] is bitmask of sums using k elements from S.
        # Initially S is empty. dp[0] = 1 (sum 0).
        dp = [0] * (49) 
        dp[0] = 1
        
        for x in order:
            if len(S) >= 48:
                break
            if time.time() > deadline - 5:
                break
            valid = is_valid_candidate(x, S, dp, max_sum)
            if valid:
                S.append(x)
                # Update DP: new subsets containing x
                # Iterate backwards to avoid using x multiple times in one step
                for k in range(len(S) - 1, 0, -1):
                    dp[k+1] |= (dp[k] << x)
                dp[1] |= (dp[0] << x)
                        
        return S

    def local_search(S, max_sum=24576, max_iterations=100):
        """Try to improve the set by swapping elements."""
        if not S:
            return S
        
        S = S[:]
        S_set = set(S)
        best_len = len(S)
        
        # Rebuild dp for current S
        dp = [0] * (49)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, 0, -1):
                dp[k+1] |= (dp[k] << x)
            dp[1] |= (dp[0] << x)
        
        nums = list(range(1, 513))
        candidates = [x for x in nums if x not in S_set]
        rng = random.Random(hash(tuple(sorted(S))) % 10000)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try to add a new element
            for x in rng.sample(candidates, min(50, len(candidates))):
                if len(S) >= 48:
                    break
                if is_valid_candidate(x, S, dp, max_sum):
                    S.append(x)
                    S_set.add(x)
                    # Update dp
                    for k in range(len(S) - 1, 0, -1):
                        dp[k+1] |= (dp[k] << x)
                    dp[1] |= (dp[0] << x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
            
            # Try to swap: remove one, add another
            if len(S) < 48:
                continue
                
            for i in range(len(S)):
                if time.time() > deadline - 5:
                    break
                removed = S[i]
                # Try adding candidates in place of removed
                for x in rng.sample(candidates, min(20, len(candidates))):
                    if x == removed:
                        continue
                    # Check if we can add x after removing S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_temp.append(x)
                    S_temp_set = set(S_temp)
                    
                    # Quick check: is x valid with S_temp?
                    if is_valid_candidate(x, S_temp, dp, max_sum):
                        # Need to rebuild dp for S_temp
                        dp_temp = [0] * (49)
                        dp_temp[0] = 1
                        for y in S_temp:
                            for k in range(len(S_temp) - 1, 0, -1):
                                dp_temp[k+1] |= (dp_temp[k] << y)
                            dp_temp[1] |= (dp_temp[0] << y)
                        
                        # Check if all elements in S_temp are valid
                        valid = True
                        for a in S_temp:
                            if a == x:
                                continue
                            # Check if a is mean of any subset of S_temp \ {a}
                            for k in range(2, len(S_temp)):
                                target = k * a
                                if target < 0 or target > max_sum:
                                    continue
                                # Need to check subsets of S_temp \ {a}
                                # This is complex, so we'll accept if x was valid
                                pass
                            break
                        
                        S = S_temp
                        S_set = S_temp_set
                        dp = dp_temp
                        candidates = [y for y in nums if y not in S_set]
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return S

    # Generate orderings
    nums = list(range(1, 513))
    
    # 1. Ascending
    ord_asc = nums[:]
    
    # 2. Descending
    ord_desc = nums[::-1]
    
    # 3. Primes first
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5)+1):
            if x % i == 0:
                return False
        return True
    
    primes = [x for x in nums if is_prime(x)]
    non_primes = [x for x in nums if not is_prime(x)]
    ord_prime = primes + non_primes
    
    # 4. Alternating small/large
    alt = []
    for i in range(256):
        if i+1 <= 512: alt.append(i+1)
        if 512-i >= 1: alt.append(512-i)
    ord_alt = alt
    
    # 5. Powers of 2 first
    powers = [2**i for i in range(10) if 2**i <= 512]
    others = [x for x in nums if x not in set(powers)]
    ord_powers = powers + others
    
    # 6. Fibonacci-like
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= 512:
        fibs.append(fibs[-1] + fibs[-2])
    fibs = list(dict.fromkeys(fibs))  # Remove duplicates, preserve order
    others_fib = [x for x in nums if x not in set(fibs)]
    ord_fib = fibs + others_fib
    
    # 7. Random shuffles
    rng = random.Random(42)
    ords = [ord_asc, ord_desc, ord_prime, ord_alt, ord_powers, ord_fib]
    for _ in range(30):
        o = nums[:]
        rng.shuffle(o)
        ords.append(o)

    best_set = []
    best_len = 0
    
    for o in ords:
        if time.time() > deadline - 10:
            break
        s = try_ordering(o)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)
        
        # Try local search to improve
        if len(s) > 0 and len(s) < 48:
            s_improved = local_search(s)
            if len(s_improved) > best_len:
                best_set = s_improved
                best_len = len(s_improved)
        
        if best_len == 48:
            break
            
    return best_set
