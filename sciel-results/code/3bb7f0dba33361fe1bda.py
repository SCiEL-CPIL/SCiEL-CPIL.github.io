import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    # Precompute inverse costs
    inv_primes = {p: Fraction(1, p) for p in primes}
    
    # We will use bitmasks to represent coverage.
    # Bit i is 1 if integer i is covered.
    # We need to cover 1..L. 
    # Let's set a working limit for the bitmask. 
    # The score is capped at 10^5, but we likely won't reach that far with sum 1/p <= 1 
    # using small primes. However, to be safe, we should handle larger L.
    # Python ints can be arbitrarily large, so we don't strictly need a fixed limit for the mask itself,
    # but generating patterns for large L is slow.
    # Strategy: Generate patterns up to a moderate limit (e.g., 2048) for fast gain calculation.
    # For finding the exact L, we can scan the bits.
    
    PATTERN_LIMIT = 2048
    
    # Precompute bit patterns for each prime and residue
    # patterns[p][r] is an int bitmask where bit k is 1 if k % p == r and 1 <= k <= PATTERN_LIMIT
    patterns = {}
    for p in primes:
        patterns[p] = {}
        # We can generate patterns for all r, or generate on the fly.
        # On the fly is O(PATTERN_LIMIT/p). Precomputing is better if we reuse.
        # But we only use r = n % p. 
        # Let's precompute all to make candidate evaluation O(1) bitwise OR.
        for r in range(p):
            mask = 0
            # Start from r. If r==0, start from p (since we care about 1..L, and 0 is not in 1..L)
            # Actually, the problem says 0 <= r < p.
            # If r=0, numbers are p, 2p, ...
            # If r>0, numbers are r, r+p, ...
            start = r
            if start == 0:
                start = p
            # Ensure we only set bits for k >= 1
            if start < 1:
                start = p # This case shouldn't happen if r>=1, but if r=0 start=p>=2
            for k in range(start, PATTERN_LIMIT + 1, p):
                mask |= (1 << k)
            patterns[p][r] = mask

    best_L_found = 0
    best_classes_found = []
    
    def find_first_uncovered(mask):
        """Find the smallest n >= 1 such that bit n in mask is 0."""
        # We can use a trick: if mask has bits 1..L set, then (mask & ((1 << (L+1)) - 1)) == ((1 << (L+1)) - 1) - 1? No.
        # Simpler: iterate. Since L is likely < 10000, this is fast.
        # Optimization: Check blocks of bits.
        # But simple loop is fine for L < 10000.
        # Let's try a faster way: 
        # If we know bits 1..L are 1, we want the first 0.
        # We can invert the mask? No, that's complex with infinite bits.
        # Let's just loop.
        n = 1
        while True:
            if (mask >> n) & 1 == 0:
                return n
            n += 1
            if n > 50000: # Safety cap
                return n

    def get_coverage_length(mask):
        """Return L such that 1..L are covered."""
        return find_first_uncovered(mask) - 1

    def dfs(current_mask, remaining_budget, used_primes, current_classes, current_L):
        nonlocal best_L_found, best_classes_found
        
        if time.time() > deadline:
            return
            
        if current_L > best_L_found:
            best_L_found = current_L
            best_classes_found = current_classes.copy()
            
        if current_L >= 1000:
            # We have achieved the target. 
            # We can try to go higher, but diminishing returns.
            # Let's allow a few more steps if time permits, but prioritize returning.
            # Actually, if we hit 1000, that's a very strong signal.
            # The task says "L >= 999 would realize...". 
            # Let's continue searching briefly for higher L, but be careful.
            pass

        n = current_L + 1
        if n > 100000: # Cap score
            return

        # Generate candidates
        candidates = []
        
        # We need to cover n.
        # So r must be n % p.
        
        for p in primes:
            if p in used_primes:
                continue
            
            cost = inv_primes[p]
            if remaining_budget < cost:
                continue
            
            r = n % p
            pattern = patterns[p][r]
            
            # Calculate new mask
            new_mask = current_mask | pattern
            
            # Calculate new L
            # Optimization: We only need to check up to a reasonable extension.
            # But find_first_uncovered is fast.
            new_L = get_coverage_length(new_mask)
            
            # Calculate gain: number of new bits set
            # gain = bin(new_mask ^ current_mask).count('1')
            # This is O(number of bits). Might be slow if mask is large.
            # But PATTERN_LIMIT is 2048, so masks are small.
            # Wait, current_mask might have bits > 2048 if we extended far?
            # No, we only OR with patterns up to 2048. 
            # So bits > 2048 are never set by our patterns?
            # Ah, this is a bug in the pattern precomputation strategy if L > 2048.
            # If L > 2048, we can't use precomputed patterns directly for the full mask.
            
            # Correction:
            # If current_L is small (< 1000), precomputed patterns are fine.
            # If current_L is large, we need a different approach or dynamic pattern generation.
            # Given the budget constraint sum(1/p) <= 1, the maximum L is likely not huge.
            # Let's estimate: sum of reciprocals of primes up to 1000 is ~ log(log(1000)) + M ~ 2 + 0.26 ~ 2.2.
            # We have budget 1. So we can use a subset.
            # Small primes cover a lot. 
            # 2, 3, 5, 7, 11, 13...
            # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31 ...
            # Sum of first 10 primes reciprocals is approx 0.5+0.33+0.2+0.14+0.09+0.077+0.058+0.052+0.043+0.034 = 1.52 > 1.
            # So we can use about 6-7 small primes.
            # With primes 2,3,5,7,11,13, the coverage is very dense.
            # LCM(2,3,5,7,11,13) = 30030.
            # It is possible to cover 1..30030? No, the density is 1 - prod(1-1/p).
            # For p=2..13, density is 1 - (1/2)(2/3)(4/5)(6/7)(10/11)(12/13) = 1 - (1/2 * 2/3 * 4/5 * 6/7 * 10/11 * 12/13)
            # = 1 - (1/13 * 1/7 * 1/5 * 1/11 * ... ) 
            # Actually, the probability a number is NOT covered by p is (p-1)/p.
            # P(not covered by 2,3,5,7,11,13) = (1/2)*(2/3)*(4/5)*(6/7)*(10/11)*(12/13) = 1/13 * 1/7 * 1/5 * 1/11 * ...
            # = 1 / (13*7*5*11 * ... ) 
            # The product is roughly 1 / (13*7*5*11* ... ) 
            # Let's calculate: 0.5 * 0.66 * 0.8 * 0.85 * 0.91 * 0.92 ≈ 0.24.
            # So 24% of numbers are NOT covered.
            # So gaps exist.
            # The first gap might be small.
            # So L is likely small, < 1000.
            # Thus, PATTERN_LIMIT = 2048 is safe.
            
            # To be absolutely safe, if new_L > PATTERN_LIMIT, we should handle it.
            # But if new_L > 2048, our pattern didn't cover bits > 2048, so new_L would be capped by the pattern limit effectively?
            # No, if pattern has 0s after 2048, and current_mask has 1s, the OR keeps 1s.
            # But if current_mask has 0s after 2048, and pattern has 0s, new_mask has 0s.
            # So get_coverage_length would find the first 0.
            # If we haven't set bits > 2048, and current_L < 2048, then new_L < 2048 is likely.
            
            if new_L > current_L:
                gain = bin(new_mask ^ current_mask).count('1')
                candidates.append((new_L, gain, p, r))
                
        if not candidates:
            return
            
        # Sort by new_L descending, then gain descending
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Limit branching factor
        max_branches = 10
        
        for new_L, gain, p, r in candidates[:max_branches]:
            if time.time() > deadline:
                return
            
            # Apply move
            new_mask = current_mask | patterns[p][r]
            new_budget = remaining_budget - inv_primes[p]
            new_used = used_primes | {p}
            new_classes = current_classes + [[p, r]]
            
            dfs(new_mask, new_budget, new_used, new_classes, new_L)
            
    # Initial state
    initial_mask = 0
    initial_budget = Fraction(1)
    initial_used = set()
    initial_classes = []
    initial_L = 0
    
    # Run DFS
    dfs(initial_mask, initial_budget, initial_used, initial_classes, initial_L)
    
    return best_classes_found
