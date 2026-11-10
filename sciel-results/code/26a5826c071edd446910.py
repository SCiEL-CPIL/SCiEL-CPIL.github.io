import math

def construct():
    N = 256
    
    # Step 1: Identify primes and their max powers <= N
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
                
    prime_to_index = {p: i for i, p in enumerate(primes)}
    
    max_power = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        max_power[p] = power
        
    # Step 2: Generate candidate set
    # Candidates are:
    # 1. Max power of each prime
    # 2. Composites x where x > sum(max_power[p] for p in factors(x))
    # 3. The number 1
    
    candidates = set()
    
    # Add max powers
    for p in primes:
        candidates.add(max_power[p])
        
    # Add 1
    candidates.add(1)
    
    # Add efficient composites
    for x in range(2, N + 1):
        if x in candidates:
            continue
            
        # Factorize x
        temp = x
        factors = set()
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                factors.add(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)
            
        # Check if x is "efficient"
        cost = 0
        valid = True
        for p in factors:
            if p in max_power:
                cost += max_power[p]
            else:
                # p is a prime > N? No, x <= N, so p <= N.
                # But if p > N/2, it's prime.
                # max_power[p] is p if p*p > N.
                # Our max_power dict covers all primes <= N.
                pass
        
        # Note: If x is prime, it's already added as max_power (if it's the max) 
        # or it's a smaller prime power?
        # If x is a prime p, max_power[p] >= p.
        # If max_power[p] > p, then x < cost, so not added.
        # If max_power[p] == p, then x == cost. We add max_power[p] which is x.
        # So primes are handled by the max_power set.
        
        if x > cost:
            candidates.add(x)
            
    # Step 3: Prepare for B&B
    # Map each candidate to (value, prime_mask)
    items = []
    for x in candidates:
        if x == 1:
            items.append((1, 0))
            continue
            
        mask = 0
        temp = x
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                if d in prime_to_index:
                    mask |= (1 << prime_to_index[d])
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            if temp in prime_to_index:
                mask |= (1 << prime_to_index[temp])
                
        items.append((x, mask))
        
    # Sort by value descending
    items.sort(key=lambda t: t[0], reverse=True)
    
    # Precompute suffix sums for pruning
    suffix_sums = [0] * (len(items) + 1)
    for i in range(len(items) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items[i][0]
        
    best_sum = 0
    
    # Branch and Bound
    def solve(idx, current_mask, current_sum):
        nonlocal best_sum
        
        # Prune: if current_sum + max possible remaining <= best_sum
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        if idx == len(items):
            if current_sum > best_sum:
                best_sum = current_sum
            return
            
        # Option 1: Skip item idx
        solve(idx + 1, current_mask, current_sum)
        
        # Option 2: Take item idx if compatible
        val, mask = items[idx]
        if (mask & current_mask) == 0:
            solve(idx + 1, current_mask | mask, current_sum + val)
            
    solve(0, 0, 0)
    
    # Reconstruct the set?
    # The task asks to return a list of ints.
    # The B&B above only finds the max sum. We need the actual set.
    # We need to modify B&B to track the path or re-run to find the set.
    # Given the small size, we can just find the set in a second pass or store it.
    
    # Let's store the best set in the B&B
    best_set = []
    
    def solve_set(idx, current_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        if idx == len(items):
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
            return
            
        # Skip
        solve_set(idx + 1, current_mask, current_sum, current_set)
        
        # Take
        val, mask = items[idx]
        if (mask & current_mask) == 0:
            current_set.append(val)
            solve_set(idx + 1, current_mask | mask, current_sum + val, current_set)
            current_set.pop()
            
    solve_set(0, 0, 0, [])
    
    # Ensure types are Python int
    return [int(x) for x in best_set]

