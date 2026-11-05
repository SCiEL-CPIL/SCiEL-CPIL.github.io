import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Helper: LCM calculation
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Helper: count distinct prime factors
    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    # Generate prime powers
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(n)
    prime_powers = set()
    for p in primes:
        power = p
        while power <= n:
            prime_powers.add(power)
            power *= p
            
    # Core set: prime powers + 1
    # This set is known to be free of bad triples internally.
    current_set = set(prime_powers | {1})
    
    # Remaining numbers to potentially add
    remaining = [x for x in range(1, n + 1) if x not in current_set]
    
    # Sort by a heuristic: fewer distinct prime factors first, then by value
    # Numbers with fewer prime factors are "simpler" and less likely to conflict.
    remaining.sort(key=lambda x: (num_distinct_primes(x), x))
    
    # Greedy Addition
    # We try to add each number from 'remaining' to 'current_set' if it doesn't form a bad triple.
    # A bad triple {x, a, b} satisfies lcm(x,a) == lcm(x,b) == lcm(a,b).
    
    for x in remaining:
        if time.time() > deadline - 10.0:
            break
            
        valid = True
        
        # Optimization:
        # For a fixed x, group existing elements a by L = lcm(x, a).
        # If there are two elements a, b in current_set such that lcm(x, a) == lcm(x, b) == L,
        # then we need to check if lcm(a, b) == L.
        # If yes, {x, a, b} is a bad triple.
        
        groups = {}
        for a in current_set:
            L = lcm(x, a)
            if L not in groups:
                groups[L] = []
            groups[L].append(a)
            
        for L, elements in groups.items():
            if len(elements) < 2:
                continue
            
            # Check all pairs in this group
            # If lcm(a, b) == L, then {x, a, b} is bad.
            for i in range(len(elements)):
                for j in range(i + 1, len(elements)):
                    a = elements[i]
                    b = elements[j]
                    if lcm(a, b) == L:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            current_set.add(x)
            
    # Refinement: Local Search / Swaps
    # The greedy addition might get stuck in a local optimum.
    # We can try to remove some elements to allow adding more "better" ones?
    # Or just try to add elements in a different order?
    
    # Let's try a simple local search:
    # Try to remove one element and see if we can add two new ones?
    # Or just try to add elements that were previously rejected, by removing conflicting elements.
    
    # A more effective refinement:
    # 1. Identify "rejected" numbers (those in original remaining but not in current_set).
    # 2. For each rejected number r, find the set of "conflict partners" in current_set.
    #    A partner p is a conflict if adding r would form a bad triple with some other element.
    #    Actually, if r was rejected, there exists a, b in current_set such that {r, a, b} is bad.
    # 3. If we remove one of the conflicting elements, r might become addable.
    #    If removing 1 element allows adding 2 or more rejected elements, it's a net gain.
    
    # Let's implement a simple version:
    # For each element e in current_set:
    #   Tentatively remove e.
    #   Try to add all rejected numbers that were blocked by e (or just try all rejected numbers).
    #   If we can add > 1 new numbers, keep the change.
    
    # To do this efficiently, we need to know why a number was rejected.
    # We can re-run the check for rejected numbers against the modified set.
    
    # Let's collect rejected numbers.
    all_numbers = set(range(1, n + 1))
    rejected = list(all_numbers - current_set)
    
    # Sort rejected by "desirability" (fewer prime factors first)
    rejected.sort(key=lambda x: (num_distinct_primes(x), x))
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try removing each element in current_set
        # To speed up, only try removing elements that are "expensive" (high degree of conflicts)?
        # Or just iterate all.
        
        # We need a fast way to check if a candidate x can be added to a set S.
        # Let's write a helper.
        
        def can_add(x, S):
            # Check if x can be added to S without forming a bad triple.
            # S is a set.
            groups = {}
            for a in S:
                L = lcm(x, a)
                if L not in groups:
                    groups[L] = []
                groups[L].append(a)
                
            for L, elements in groups.items():
                if len(elements) < 2:
                    continue
                for i in range(len(elements)):
                    for j in range(i + 1, len(elements)):
                        a = elements[i]
                        b = elements[j]
                        if lcm(a, b) == L:
                            return False
            return True

        # Iterate over elements in current_set
        # We should iterate over a copy since we might modify the set.
        for e in list(current_set):
            if time.time() > deadline - 5.0:
                break
            
            # Tentatively remove e
            current_set.remove(e)
            
            # Try to add rejected numbers
            added_count = 0
            added_elements = []
            
            # We can try to add multiple numbers.
            # Greedily add rejected numbers that are now valid.
            # We should re-check validity because adding one number might block another.
            
            # To be safe and simple, let's just try to add numbers one by one.
            # If we add one, the set changes, so we need to re-evaluate the next candidate.
            
            # Let's make a copy of the rejected list to iterate?
            # No, we can just iterate through the sorted rejected list.
            
            # Optimization: Only check rejected numbers that might have been blocked by e.
            # But "blocked by e" is hard to determine without re-running the check.
            # So we just check all rejected numbers.
            
            # To make this faster, we can limit the number of attempts or use a heuristic.
            # Let's just try to add as many as possible from the sorted rejected list.
            
            for r in rejected:
                if r in current_set:
                    continue
                if can_add(r, current_set):
                    current_set.add(r)
                    added_count += 1
                    added_elements.append(r)
            
            # If we added more than 1 element, it's a net gain (removed 1, added > 1).
            # If we added exactly 1, it's a neutral swap (removed 1, added 1).
            # If we added 0, it's a loss.
            
            if added_count > 1:
                # Net gain! Keep the change.
                # Update the rejected list (remove added elements)
                rejected = [r for r in rejected if r not in current_set]
                improved = True
            else:
                # No net gain or loss. Restore e.
                current_set.add(e)
                
                # If we added some elements but not enough to offset the removal,
                # we should probably remove them too to restore the state?
                # Yes, we want to restore the state if we didn't improve.
                for r in added_elements:
                    current_set.remove(r)
                    
                # Note: The rejected list is already correct (it didn't include e, and we removed added elements from current_set, so they are effectively rejected again).
                # Wait, if we added r to current_set, and then removed it, r is still in 'rejected' list?
                # 'rejected' was defined as all_numbers - current_set.
                # If we modify current_set, 'rejected' becomes stale.
                # So we should update 'rejected' only if we accept the change.
                # If we reject the change, we restore current_set, so 'rejected' is still valid (assuming it was valid before).
                
    # Final Validation
    # Ensure the set is valid.
    # If not, fall back to a smaller safe set (e.g., prime powers).
    
    # Check for bad triples in current_set
    # This is O(|S|^3). |S| is likely ~80-100. 100^3 = 1M. Fast enough.
    # But we can do it smarter.
    
    valid = True
    S = list(current_set)
    S.sort()
    for i in range(len(S)):
        if time.time() > deadline - 1.0:
            break
        for j in range(i + 1, len(S)):
            lab = lcm(S[i], S[j])
            for k in range(j + 1, len(S)):
                if lcm(S[i], S[k]) == lab and lcm(S[j], S[k]) == lab:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to safe core
        return list(prime_powers | {1})
        
    return list(current_set)
