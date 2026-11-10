import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def is_valid_candidate(x, S, dp):
        """
        Check if x can be added to S.
        S: list of currently selected integers.
        dp: list of integers, where dp[k] is a bitmask of possible sums using exactly k elements from S.
        dp[0] = 1 (sum 0 with 0 elements)
        """
        n = len(S)
        
        # 1. Check if x is the mean of any subset of S of size k >= 2.
        # sum(subset) = k * x
        for k in range(2, n + 1):
            target = k * x
            if target <= 0:
                continue
            if (dp[k] >> target) & 1:
                return False
        
        # 2. Check if any a in S becomes the mean of a subset of S U {x} of size >= 2
        # that includes x.
        # Let the subset be T U {x}, where T subset of S.
        # Sum(T) + x = (|T| + 1) * a
        # Sum(T) = (|T| + 1) * a - x
        # Let k = |T| + 1. Then k >= 2.
        # Sum(T) = k * a - x.
        # T has size k - 1.
        for a in S:
            for k in range(2, n + 2):
                target = k * a - x
                if target <= 0:
                    continue
                # Check if there is a subset of S of size k-1 with sum 'target'
                if (dp[k - 1] >> target) & 1:
                    return False
        
        return True

    def try_ordering(order):
        S = []
        dp = [0] * 50  # Max size 48, so 50 is safe
        dp[0] = 1
        
        for x in order:
            if len(S) >= 48:
                break
            if is_valid_candidate(x, S, dp):
                S.append(x)
                # Update DP: iterate backwards to avoid using x multiple times
                # dp[k] represents sums using k elements
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
        
        return S

    def calculate_conflict_scores(S):
        """
        For each x in S, calculate how many subsets of S\{x} have mean x.
        This helps identify 'bad' elements.
        """
        n = len(S)
        if n == 0:
            return {}
        
        # Build DP for S
        dp = [0] * 50
        dp[0] = 1
        for x in S:
            for k in range(n - 1, -1, -1):
                dp[k + 1] |= (dp[k] << x)
        
        scores = {}
        for x in S:
            count = 0
            # Check if x is mean of subset of S\{x}
            # We need DP of S\{x}.
            # Rebuilding DP for each x is O(n^2). Total O(n^3).
            # n=48, 48^3 = 110k. Very fast.
            dp_rem = [0] * 50
            dp_rem[0] = 1
            for y in S:
                if y == x:
                    continue
                for k in range(len(S) - 1, -1, -1):
                    # Careful: dp_rem size depends on number of elements processed
                    # We need to track current size
                    pass
            
            # Simpler: Just count by iterating k and checking bits in a DP built without x.
            # Let's just rebuild DP without x.
            dp_no_x = [0] * 50
            dp_no_x[0] = 1
            count_elements = 0
            for y in S:
                if y == x:
                    continue
                for k in range(count_elements, -1, -1):
                    dp_no_x[k + 1] |= (dp_no_x[k] << y)
                count_elements += 1
            
            for k in range(2, count_elements + 1):
                target = k * x
                if (dp_no_x[k] >> target) & 1:
                    count += 1
            scores[x] = count
        return scores

    # Generate diverse orderings
    nums = list(range(1, 513))
    
    # Basic orderings
    ord_asc = nums[:]
    ord_desc = nums[::-1]
    
    # Primes first
    def is_prime(x):
        if x < 2: return False
        if x < 4: return True
        if x % 2 == 0 or x % 3 == 0: return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0: return False
            i += 6
        return True
        
    primes = [x for x in nums if is_prime(x)]
    non_primes = [x for x in nums if not is_prime(x)]
    ord_prime = primes + non_primes
    
    # Random orderings
    rng = random.Random(42)
    
    best_set = []
    
    # Phase 1: Greedy with multiple orderings
    # We will run several random shuffles and structured orderings
    
    # Structured
    candidates = [ord_asc, ord_desc, ord_prime]
    
    # Add some random ones
    for i in range(20):
        o = nums[:]
        rng.shuffle(o)
        candidates.append(o)
        
    for o in candidates:
        if time.time() > deadline - 20: # Reserve time for local search
            break
        s = try_ordering(o)
        if len(s) > len(best_set):
            best_set = s
        if len(best_set) == 48:
            break

    # Phase 2: Local Search (Swaps)
    if time.time() < deadline - 5:
        # Try to improve best_set
        S = best_set[:]
        S_set = set(S)
        
        # Pre-calculate conflicts to prioritize removals
        # But dynamic conflict is better.
        
        # Simple random swap search
        attempts = 0
        max_attempts = 5000
        no_improve_count = 0
        threshold = 100
        
        while attempts < max_attempts and time.time() < deadline - 2:
            if no_improve_count > threshold:
                break
            
            attempts += 1
            
            # Pick random x from S
            x = rng.choice(S)
            # Pick random y from complement
            # To avoid O(N) scan for complement, we can just pick random y from 1..512 and check if in S
            # Or maintain a list of available numbers.
            # Since S is small, checking `if y not in S_set` is fast.
            
            y = rng.randint(1, 512)
            if y in S_set:
                continue
            
            # Try swap x -> y
            # New set: S' = S \ {x} U {y}
            
            # Check validity of S'
            # We can do this by checking if y is valid in S\{x}
            # And if any element in S\{x} is valid in S'
            
            # Step 1: Build DP for S_rem = S \ {x}
            S_rem = [z for z in S if z != x]
            n_rem = len(S_rem)
            
            dp_rem = [0] * 50
            dp_rem[0] = 1
            for z in S_rem:
                for k in range(n_rem - 1, -1, -1):
                    dp_rem[k + 1] |= (dp_rem[k] << z)
            
            # Check 1: Is y mean of any subset of S_rem?
            valid = True
            for k in range(2, n_rem + 1):
                target = k * y
                if (dp_rem[k] >> target) & 1:
                    valid = False
                    break
            
            if valid:
                # Check 2: Is any z in S_rem mean of any subset of S_rem U {y}?
                # Subset includes y.
                for z in S_rem:
                    for k in range(2, n_rem + 2):
                        target = k * z - y
                        if target <= 0:
                            continue
                        if (dp_rem[k - 1] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Perform swap
                S.remove(x)
                S.append(y)
                S_set.remove(x)
                S_set.add(y)
                best_set = S[:]
                no_improve_count = 0
            else:
                no_improve_count += 1

    return best_set
