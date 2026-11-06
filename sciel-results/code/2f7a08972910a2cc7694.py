import time

def construct():
    n = 5040
    target = 4979
    
    # Get divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    
    # Filter out divisors larger than target
    divisors = [d for d in divisors if d <= target]
    
    # Precompute suffix sums for max sum pruning?
    # suffix_sum[i] = sum(divisors[i:])
    # But we need sum of specific counts.
    
    # Search for minimum k
    for k in range(1, 10):
        # Check if it's even possible
        if sum(divisors[:k]) < target:
            continue
        # If min sum of k divisors > target, skip?
        # Min sum of k divisors is sum of last k divisors.
        if sum(divisors[-k:]) > target:
            # Since divisors are sorted desc, last k are smallest.
            # If even the smallest k sum to more than target, no solution.
            continue
            
        found = None
        
        # DFS
        def dfs(idx, remaining, current):
            if remaining == 0:
                return current
            if len(current) == k:
                return None
            if idx >= len(divisors):
                return None
            if remaining < 0:
                return None
                
            # Pruning 1: Max possible sum with remaining slots
            slots = k - len(current)
            if idx + slots > len(divisors):
                return None
            max_possible_sum = sum(divisors[idx : idx + slots])
            if max_possible_sum < remaining:
                return None
                
            # Pruning 2: Min possible sum with remaining slots
            # The smallest 'slots' divisors available from idx onwards
            # are at the end of the divisors list.
            # But we must pick from idx onwards.
            # So the smallest 'slots' are divisors[len(divisors)-slots : len(divisors)]
            # Wait, if idx is large, the available set is smaller.
            # The available set is divisors[idx:].
            # The smallest 'slots' elements in divisors[idx:] are the last 'slots' elements of divisors[idx:].
            # Which are divisors[len(divisors)-slots : len(divisors)]?
            # No, if idx > len(divisors)-slots, then the smallest 'slots' are just the last 'slots' of the whole list?
            # Actually, if we are at index idx, we can pick any subset of size 'slots' from divisors[idx:].
            # The minimum sum is the sum of the 'slots' smallest elements in divisors[idx:].
            # Since divisors is sorted descending, the smallest elements are at the end.
            # So min_sum = sum(divisors[len(divisors)-slots : len(divisors)])
            # BUT this is only true if idx <= len(divisors)-slots.
            # If idx > len(divisors)-slots, then we are forced to pick elements that are larger.
            # In that case, the min sum is sum(divisors[idx : idx + slots])? No, that's max.
            # If idx > len(divisors)-slots, then the available elements are divisors[idx:].
            # The size of available is len(divisors)-idx.
            # We need to pick 'slots' from them.
            # The smallest 'slots' are the last 'slots' of the available list.
            # So min_sum = sum(divisors[len(divisors)-slots : len(divisors)]) is NOT correct if idx is large.
            # Correct min_sum:
            # available = divisors[idx:]
            # min_sum = sum(available[-slots:])
            
            min_possible_sum = sum(divisors[len(divisors)-slots : len(divisors)])
            # Wait, if idx is 50 and slots is 2, available is divisors[50:].
            # len(divisors) is 60.
            # available[-2:] is divisors[58:].
            # But we can only pick from divisors[50:].
            # So the smallest 2 in divisors[50:] are divisors[58] and divisors[59].
            # So sum(divisors[58:60]) is correct.
            # What if idx=59 and slots=2? Impossible, checked earlier.
            # What if idx=58 and slots=2? Available is [58, 59]. Smallest 2 are [58, 59].
            # sum(divisors[58:60]) is correct.
            
            # However, if idx is such that idx + slots > len(divisors), we return None.
            # So we always have at least 'slots' elements available.
            # The smallest 'slots' elements in divisors[idx:] are indeed the last 'slots' elements of the whole list?
            # No. If idx=0, smallest 'slots' are last 'slots'.
            # If idx=10, smallest 'slots' in divisors[10:] are still the last 'slots' of the whole list, provided idx <= len(divisors)-slots.
            # Since we checked `idx + slots <= len(divisors)`, we have `idx <= len(divisors)-slots`.
            # So the last 'slots' elements of the whole list are indeed in the range [idx, len(divisors)).
            # So `min_possible_sum = sum(divisors[len(divisors)-slots : len(divisors)])` is CORRECT.
            
            if min_possible_sum > remaining:
                return None

            # Include divisors[idx]
            res = dfs(idx + 1, remaining - divisors[idx], current + [divisors[idx]])
            if res:
                return res
                
            # Exclude divisors[idx]
            res = dfs(idx + 1, remaining, current)
            if res:
                return res
                
            return None

        found = dfs(0, target, [])
        if found:
            return found
            
    # Fallback
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
