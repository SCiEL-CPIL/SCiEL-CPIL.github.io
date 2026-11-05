"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 87.0  # seconds, safety margin before 90s limit

def _check_x_is_mean_of_subset(x, S):
    """
    Check if x is the mean of any subset of at least 2 elements in S.
    Uses bitset DP on differences.
    """
    n = len(S)
    if n < 2:
        return False
    
    zs = [y - x for y in S]
    
    # Calculate max absolute sum for offset
    max_sum = sum(abs(z) for z in zs)
    if max_sum == 0:
        return False  # All equal to x, impossible with distinct elements
        
    offset = max_sum
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitset where bit s is set if sum s-offset is achievable with exactly k elements
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            
            dp[k] |= (shifted & mask)
            
            # Early exit: if sum 0 is achievable with k >= 2 elements
            if k >= 2 and (dp[k] & (1 << offset)):
                return True
                
    return False

def _check_existing_violation(x, S):
    """
    Check if adding x to S creates any violation where an element a in S
    is the mean of a subset containing x and at least one other element from S.
    
    For each a in S, we need: x + sum(T) = |T+1| * a for some non-empty T subset of S\{a}
    Equivalent: sum(t - a for t in T) = a - x
    """
    n = len(S)
    if n < 2:
        return False
    
    S_set = set(S)
    
    for a in S:
        target_val = a - x
        U = [t for t in S if t != a]
        if len(U) < 1:
            continue
        
        ws = [t - a for t in U]
        max_abs = sum(abs(w) for w in ws)
        if max_abs == 0:
            continue
        
        offset = max_abs
        mask = (1 << (2 * max_abs + 1)) - 1
        
        dp = [0] * (len(U) + 1)
        dp[0] = 1 << offset
        
        for w in ws:
            for k in range(len(U), 0, -1):
                if dp[k-1] == 0:
                    continue
                if w >= 0:
                    shifted = dp[k-1] << w
                else:
                    shifted = dp[k-1] >> (-w)
                dp[k] |= (shifted & mask)
        
        if 0 <= offset + target_val <= 2 * max_abs:
            for k in range(1, len(U) + 1):
                if dp[k] & (1 << (offset + target_val)):
                    return True
    
    return False

def _quick_pair_check(x, S):
    """
    Quick check for 2-element subsets involving x.
    """
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            if S[i] + S[j] == 2 * x:
                return True
    
    S_set = set(S)
    for a in S:
        b = 2 * a - x
        if b in S_set and b != a:
            return True
    
    return False

def _full_check(C):
    """
    Full verification: check if any element in C is the mean of a subset of 
    at least 2 other elements.
    """
    for i in range(len(C)):
        a = C[i]
        others = C[:i] + C[i+1:]
        if _check_x_is_mean_of_subset(a, others):
            return True
    return False

def _generate_sidon_seed():
    """
    Generate a Sidon sequence (B2 sequence) in [1, 512].
    A Sidon sequence has all pairwise sums distinct.
    We then filter it to remove any pair-mean violations.
    """
    # Greedy Sidon construction
    S = []
    sums = set()
    for x in range(1, 513):
        valid = True
        for a in S:
            s = a + x
            if s in sums:
                valid = False
                break
            sums.add(s)
        if valid:
            S.append(x)
    
    # Filter S to ensure no element is the mean of a pair of others
    valid_S = []
    S_set = set(S)
    for x in S:
        is_mean = False
        for a in S:
            if a == x:
                continue
            b = 2 * x - a
            if b in S_set and b != x:
                is_mean = True
                break
        if not is_mean:
            valid_S.append(x)
            
    return valid_S

def _generate_power_of_two_seed():
    """
    Generate a set based on powers of 2, which naturally avoids mean violations.
    """
    S = []
    val = 1
    while val <= 512:
        S.append(val)
        val *= 2
    # Add some more values that maintain the property
    for x in range(1, 513):
        if x not in S:
            # Check if x can be added without creating a mean violation
            valid = True
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    if S[i] + S[j] == 2 * x:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(x)
    return S[:48]

def _generate_spaced_seed():
    """
    Generate a well-spaced set that avoids arithmetic progressions.
    """
    S = []
    # Use a logarithmic spacing pattern
    for i in range(1, 15):
        val = int(512 * (i / 14.0))
        if 1 <= val <= 512 and val not in S:
            S.append(val)
    
    # Add more values with careful checking
    for x in range(1, 513):
        if len(S) >= 48:
            break
        if x in S:
            continue
        valid = True
        for a in S:
            b = 2 * x - a
            if b in S and b != x:
                valid = False
                break
        if valid:
            S.append(x)
    
    return S

def construct():
    start_time = time.time()
    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Generate diverse orders
    orders = []
    
    # 1. Sidon-based seed order
    # We want to prioritize elements that form a good Sidon set
    sidon_seed = _generate_sidon_seed()
    # Create an order that puts sidon_seed first, then the rest
    # We shuffle the sidon_seed a bit to allow different variations?
    # Or just use it as a priority list.
    rest = [x for x in candidates if x not in sidon_seed]
    orders.append(sidon_seed + rest)
    
    # 2. Structured orders
    orders.append(candidates[:])       # Increasing
    orders.append(candidates[::-1])    # Decreasing
    
    # 3. Spread out orders
    spread_order = []
    lo, hi = 1, 512
    while lo <= hi:
        spread_order.append(hi)
        if lo != hi:
            spread_order.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread_order)
    
    # 4. Power of 2 based seed order
    pow2_seed = _generate_power_of_two_seed()
    rest = [x for x in candidates if x not in pow2_seed]
    orders.append(pow2_seed + rest)
    
    # 5. Spaced seed order
    spaced_seed = _generate_spaced_seed()
    rest = [x for x in candidates if x not in spaced_seed]
    orders.append(spaced_seed + rest)
    
    # 6. Random orders with different seeds
    rng = random.Random(42)
    num_random_orders = 200
    
    for i in range(num_random_orders):
        if time.time() - start_time > _DEADLINE - 3:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
    
    # 7. Biased random orders: weight towards extremes
    for i in range(80):
        if time.time() - start_time > _DEADLINE - 3:
            break
        biased = []
        for _ in range(512):
            r = rng.random()
            if r < 0.3:
                biased.append(rng.randint(1, 50))
            elif r < 0.6:
                biased.append(rng.randint(463, 512))
            else:
                biased.append(rng.randint(1, 512))
        seen = set()
        unique_biased = []
        for v in biased:
            if v not in seen:
                seen.add(v)
                unique_biased.append(v)
        for v in range(1, 513):
            if v not in seen:
                unique_biased.append(v)
        orders.append(unique_biased)
    
    # 8. Prime-based order
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(1, 513) if is_prime(x)]
    non_primes = [x for x in candidates if not is_prime(x)]
    orders.append(primes + non_primes)
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 2:
            break
            
        selected = []
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 2:
                break
            
            if len(selected) >= 48:
                break
                
            # Quick rejection checks first (fastest)
            if _quick_pair_check(x, selected):
                continue
            
            # Then check if x is mean of subset of selected
            if _check_x_is_mean_of_subset(x, selected):
                continue
            
            # Finally check if any existing element becomes mean of subset containing x
            if _check_existing_violation(x, selected):
                continue
            
            selected.append(x)
            
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) >= 48:
                break
    
    if best_set:
        if not _full_check(best_set):
            return best_set
    
    # Fallback: try to build a valid set from scratch with a simple approach
    fallback = []
    for x in range(1, 513):
        if len(fallback) >= 48:
            break
        if _quick_pair_check(x, fallback):
            continue
        if _check_x_is_mean_of_subset(x, fallback):
            continue
        if _check_existing_violation(x, fallback):
            continue
        fallback.append(x)
    
    if fallback and not _full_check(fallback):
        return fallback
    
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
