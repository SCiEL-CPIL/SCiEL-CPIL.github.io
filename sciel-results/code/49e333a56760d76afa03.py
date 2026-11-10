# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 80  # Leave 10s margin for safety
    
    # Helper to compute dp bitsets for a list of numbers
    def compute_dp(nums):
        n = len(nums)
        if n == 0:
            return [1] # dp[0] = 1 (bit 0)
        # dp[k] is an integer where bit s is 1 if there's a subset of size k summing to s
        dp = [0] * (n + 1)
        dp[0] = 1  # bit 0 is 1
        for x in nums:
            # Update in reverse order of k to avoid using x multiple times
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        return dp

    def is_valid_addition(S, x):
        """
        Check if x can be added to S without violating constraints.
        S is a list of ints. x is not in S.
        Returns True if valid.
        """
        if x in S:
            return False
        
        n = len(S)
        if n == 0:
            return True
        
        # 1. Check if x is the mean of any subset of S of size k >= 2
        # sum(T) = k * x
        dp_S = compute_dp(S)
        sum_S = sum(S)
        for k in range(2, n + 1):
            target_sum = k * x
            if target_sum > sum_S:
                break
            if (dp_S[k] >> target_sum) & 1:
                return False
                
        # 2. Check if any a in S is the mean of a subset of S U {x} containing x
        # Subset T U {x}, T subset of S \ {a}
        # sum(T) + x = (|T| + 1) * a
        # sum(T) = (|T| + 1) * a - x
        for a in S:
            S_excl_a = [y for y in S if y != a]
            if not S_excl_a:
                continue
            dp_excl = compute_dp(S_excl_a)
            m = len(S_excl_a)
            sum_excl = sum(S_excl_a)
            for j in range(1, m + 1):
                target_sum = (j + 1) * a - x
                if target_sum < 0:
                    continue
                if target_sum > sum_excl:
                    break 
                if (dp_excl[j] >> target_sum) & 1:
                    return False
                    
        return True

    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True

    def greedy_construct(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= 48:
                    break
        return selected

    # Strategy: 
    # 1. Generate candidate orders.
    # 2. The "Prime First" heuristic is strong. Primes are less likely to be averages.
    # 3. We will try a few variations:
    #    a. Primes first, then composites in descending order.
    #    b. Primes first, then composites in ascending order.
    #    c. Randomized with prime bias.
    
    candidates = list(range(1, 513))
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    orders = []
    
    # Order 1: Primes descending, then composites descending
    orders.append(sorted(primes, reverse=True) + sorted(composites, reverse=True))
    
    # Order 2: Primes ascending, then composites ascending
    orders.append(sorted(primes) + sorted(composites))
    
    # Order 3: Primes descending, then composites ascending
    orders.append(sorted(primes, reverse=True) + sorted(composites))
    
    # Order 4: Interleaved? Or just random with prime bias
    # Let's do a few random shuffles but ensure primes are in the first half?
    # Actually, just standard random shuffles of the full list, but we run many.
    for _ in range(5):
        r = list(candidates)
        random.shuffle(r)
        orders.append(r)
        
    best_size = 0
    best_result = []
    
    for order in orders:
        if time.time() > deadline - 20:
            break
            
        res = greedy_construct(order)
        if len(res) > best_size:
            best_size = len(res)
            best_result = res
            
        # If we hit the max, we can stop early? 
        # But maybe a different order allows 48 where this one allows 47? 
        # The task says "Maximize cardinality". 48 is the hard cap.
        if best_size == 48:
            break

    # Local Search / Improvement Phase
    # If we didn't reach 48, try to swap elements to allow adding more.
    if best_size < 48 and time.time() < deadline - 10:
        current_best = best_result[:]
        unused = [x for x in range(1, 513) if x not in current_best]
        
        # Try to add more elements directly first
        random.shuffle(unused)
        for x in unused:
            if time.time() > deadline - 2:
                break
            if len(current_best) >= 48:
                break
            if is_valid_addition(current_best, x):
                current_best.append(x)
        
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_result = current_best
        
        # If still not 48, try swaps.
        # This is expensive. Only do a limited number of swaps.
        if len(current_best) < 48:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try swapping one element in current_best with one in unused
                # We want to see if this swap allows adding a new element.
                # To speed up, we can just try random swaps and check if the set size can increase by 1 afterwards.
                
                # Pick a random element to remove
                idx_to_remove = random.randint(0, len(current_best) - 1)
                elem_to_remove = current_best[idx_to_remove]
                
                # Try a few random candidates to replace it
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    if not unused:
                        break
                    elem_to_add = random.choice(unused)
                    
                    # Test swap: Remove elem_to_remove, Add elem_to_add
                    # New set: (current_best - {elem_to_remove}) + {elem_to_add}
                    # Check if this new set is valid.
                    # If valid, can we add ANOTHER element?
                    # Or does the swap itself fix a conflict that prevented addition?
                    
                    # Actually, a simpler local search: 
                    # Just try to add elements to the current set. If stuck, 
                    # maybe removing a "bad" element allows adding two "good" ones?
                    
                    # Let's try a specific heuristic:
                    # Identify elements that are "means" of other subsets? 
                    # We don't track that. 
                    
                    # Let's just try random swaps to see if we can get a set that allows more additions.
                    new_set = [x for x in current_best if x != elem_to_remove]
                    if elem_to_add in new_set:
                        continue
                    new_set.append(elem_to_add)
                    
                    # Is new_set valid?
                    # We can check validity by trying to add each element of new_set to an empty set?
                    # No, that's expensive.
                    # We can check if new_set is valid by verifying all elements.
                    # But is_valid_addition only checks addition.
                    # Let's implement a quick full check for new_set.
                    def is_valid_set(S):
                        if len(S) < 3:
                            return True
                        for a in S:
                            S_excl_a = [x for x in S if x != a]
                            if len(S_excl_a) < 2:
                                continue
                            dp_excl = compute_dp(S_excl_a)
                            m = len(S_excl_a)
                            sum_excl = sum(S_excl_a)
                            for k in range(2, m + 1):
                                target_sum = k * a
                                if target_sum > sum_excl:
                                    break
                                if (dp_excl[k] >> target_sum) & 1:
                                    return False
                        return True

                    if not is_valid_set(new_set):
                        continue
                    
                    # If valid, try to add more elements to new_set
                    potential_set = new_set[:]
                    remaining_unused = [x for x in unused if x != elem_to_add and x not in potential_set]
                    random.shuffle(remaining_unused)
                    
                    for x in remaining_unused:
                        if len(potential_set) >= 48:
                            break
                        if is_valid_addition(potential_set, x):
                            potential_set.append(x)
                    
                    if len(potential_set) > len(current_best):
                        current_best = potential_set
                        best_size = len(current_best)
                        best_result = current_best
                        improved = True
                        # Update unused
                        unused = [x for x in range(1, 513) if x not in current_best]
                        break

    return best_result
# EVOLVE-BLOCK-END
