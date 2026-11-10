def construct():
    import random
    from math import gcd
    from itertools import combinations

    n = 128
    
    # Precompute divisors for each number
    divisors = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if j % i == 0:
                divisors[j].append(i)
                
    # Precompute gcd table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            gcd_table[i][j] = gcd(i, j)

    def has_three_pairwise_coprime(nums):
        """
        Returns True if the list of numbers contains 3 distinct elements
        that are pairwise coprime.
        """
        if len(nums) < 3:
            return False
        
        # Optimization: if 1 is in the list, we just need two coprime numbers from the rest
        if 1 in nums:
            rest = [x for x in nums if x != 1]
            # Check if there are any two coprime numbers in rest
            for i in range(len(rest)):
                for j in range(i + 1, len(rest)):
                    if gcd(rest[i], rest[j]) == 1:
                        return True
            return False
            
        # General case: check all triples
        # This is O(K^3) but K is usually small
        for a, b, c in combinations(nums, 3):
            if gcd(a, b) == 1 and gcd(a, c) == 1 and gcd(b, c) == 1:
                return True
        return False

    def is_valid_addition(S, x):
        """
        Checks if adding x to S creates a bad triple.
        A bad triple {a,b,c} exists iff there is a d such that a/d, b/d, c/d
        are pairwise coprime.
        """
        # For each divisor d of x, check the reduced set
        for d in divisors[x]:
            # Build the reduced set for this d
            # R_d = { y/d for y in S if d|y } U { x/d }
            reduced_set = []
            for y in S:
                if y % d == 0:
                    reduced_set.append(y // d)
            reduced_set.append(x // d)
            
            if has_three_pairwise_coprime(reduced_set):
                return False
        return True

    # Greedy construction with randomization
    best_set = set()
    best_size = 0
    
    # Run multiple random greedy trials
    for trial in range(50):
        random.seed(42 + trial)
        current_set = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    # Local search to improve
    # Try to swap elements or add/remove
    # Given the time limit, a simple hill climbing might suffice
    # We will try to add elements that were rejected, by removing some existing ones?
    # Or just try random swaps.
    
    # A more effective local search:
    # Try to add each excluded element. If it fails, try to remove one conflicting element.
    
    final_set = best_set.copy()
    improved = True
    while improved:
        improved = False
        # Get elements not in final_set
        missing = [x for x in range(1, n + 1) if x not in final_set]
        random.shuffle(missing)
        
        for x in missing:
            if is_valid_addition(final_set, x):
                final_set.add(x)
                improved = True
                # Restart loop as set changed
                break
            
            # If not valid, try to find a swap
            # Find an element y in final_set such that removing y and adding x is valid
            # This is expensive. Let's just try a few random swaps.
            if len(final_set) > 0:
                y = random.choice(list(final_set))
                new_set = final_set - {y}
                if is_valid_addition(new_set, x):
                    final_set = new_set
                    final_set.add(x)
                    improved = True
                    break

    return list(final_set)
