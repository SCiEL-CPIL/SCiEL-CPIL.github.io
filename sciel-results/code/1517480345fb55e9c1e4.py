"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    # Generate primes up to 1000
    def gen_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = gen_primes(1000)
    MAX_L = 100000
    SEARCH_LIMIT = 10000  # Look ahead this far to decide best class
    
    # State
    covered = [False] * (MAX_L + 1)
    used_primes = set()
    classes = []
    budget = Fraction(1, 1)
    
    # Greedy: Maximize coverage of currently uncovered numbers in [1, SEARCH_LIMIT]
    # This is faster and often more effective than just extending the prefix
    for _ in range(len(primes)):
        best_gain = 0
        best_prime = None
        best_residue = None
        
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget - cost < 0:
                continue
            
            max_gain_for_p = 0
            best_r_for_p = 0
            
            for r in range(p):
                gain = 0
                start = p if r == 0 else r
                # Only count up to SEARCH_LIMIT for speed
                for n in range(start, SEARCH_LIMIT + 1, p):
                    if not covered[n]:
                        gain += 1
                
                if gain > max_gain_for_p:
                    max_gain_for_p = gain
                    best_r_for_p = r
            
            if max_gain_for_p > best_gain:
                best_gain = max_gain_for_p
                best_prime = p
                best_residue = best_r_for_p
        
        if best_prime is None or best_gain == 0:
            break
        
        # Apply the best choice
        p = best_prime
        r = best_residue
        used_primes.add(p)
        classes.append([p, r])
        budget -= Fraction(1, p)
        
        # Update coverage for the full range
        start = p if r == 0 else r
        for n in range(start, MAX_L + 1, p):
            covered[n] = True
    
    # Verify and find L
    L = 0
    while L + 1 <= MAX_L and covered[L + 1]:
        L += 1
        
    # Double-check budget
    total = sum(Fraction(1, p) for p, r in classes)
    # The logic ensures budget <= 1
    
    return classes
# EVOLVE-BLOCK-END
