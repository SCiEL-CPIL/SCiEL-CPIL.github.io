# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'n': 256}

def _prime_factors(n):
    """Return set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _valid_set(members):
    """Check if the set of members satisfies the constraint:
    For every three distinct a,b,c in members, a does not divide b*c."""
    m = len(members)
    for i in range(m):
        a = members[i]
        for j in range(i + 1, m):
            b = members[j]
            for k in range(j + 1, m):
                c = members[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(candidate, members):
    """Check if adding candidate to members keeps the set valid."""
    for i in range(len(members)):
        a = members[i]
        for j in range(i + 1, len(members)):
            b = members[j]
            c = candidate
            if (b * c) % a == 0:
                return False
            if (a * c) % b == 0:
                return False
    for i in range(len(members)):
        a = members[i]
        for j in range(i + 1, len(members)):
            b = members[j]
            if (a * b) % candidate == 0:
                return False
    return True

def construct():
    deadline = time.time() + 80  # safety margin for 90s limit
    
    n = P['n']
    all_nums = list(range(2, n + 1))
    
    # Precompute prime factors for sorting heuristics
    pf = {}
    for x in all_nums:
        pf[x] = _prime_factors(x)
        
    def _smallest_prime_factor(x):
        if x % 2 == 0: return 2
        if x % 3 == 0: return 3
        d = 5
        while d * d <= x:
            if x % d == 0:
                return d
            d += 2
        return x

    # Initial Greedy: Try to build a large set
    # Strategy: Primes are generally good, but composites with large prime factors might be better.
    # Let's try multiple greedy orderings and keep the best result.
    
    best_set = []
    
    # Ordering 1: Primes first (by value desc), then composites (by SPF desc)
    order1 = sorted(all_nums, key=lambda x: (0 if len(pf[x])==1 and x==min(pf[x]) else 1, -_smallest_prime_factor(x), -x))
    # Ordering 2: All by SPF desc
    order2 = sorted(all_nums, key=lambda x: (-_smallest_prime_factor(x), -x))
    # Ordering 3: All by value desc
    order3 = sorted(all_nums, key=lambda x: -x)
    # Ordering 4: All by number of prime factors asc, then value desc
    order4 = sorted(all_nums, key=lambda x: (len(pf[x]), -x))
    
    for ordering in [order1, order2, order3, order4]:
        if time.time() > deadline:
            break
        current = []
        for x in ordering:
            if time.time() > deadline:
                break
            if _can_add(x, current):
                current.append(x)
        if len(current) > len(best_set):
            best_set = current
            # If we found a very large set, we might still try to improve
            # but for now, just record it.

    # Local Search Phase
    # Try to improve best_set by:
    # 1. Adding any number not in set (if valid)
    # 2. Swapping any number in set with a number not in set (if valid and size increases or same but "better"?)
    #    Since we want max size, only accept swap if size increases (impossible for 1-to-1) 
    #    OR if it allows further additions.
    #    Actually, 1-to-1 swap keeps size same. We only accept if it enables adding more later?
    #    Simpler: Just try to add. If can't add, try to remove one and add two?
    
    current_best = list(best_set)
    current_size = len(current_best)
    improved = True
    in_set = set(current_best)
    
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding any candidate
        for x in all_nums:
            if time.time() > deadline:
                break
            if x not in in_set:
                if _can_add(x, current_best):
                    current_best.append(x)
                    in_set.add(x)
                    current_size += 1
                    improved = True
                    # Restart loop to check for more additions
                    break 
        
        if improved:
            continue
            
        # 2. Try swapping: remove y, add x.
        #    Only useful if it allows adding more later.
        #    Heuristic: Try swapping small primes with composites that have larger SPF?
        #    Or just try all swaps? 50 * 200 = 10,000 swaps. Each check is O(N^2) ~ 2500 ops.
        #    10,000 * 2500 = 25M ops. Might be too slow in Python if done repeatedly.
        #    Let's limit swap attempts.
        
        # Try removing an element and see if we can add two others?
        # Or just try 1-for-1 swaps to see if they lead to a state where we can add more.
        
        # Efficient Swap:
        # For each y in current_best:
        #   For each x not in current_best:
        #     If we remove y and add x, is the set valid?
        #     If yes, and if we can then add z (another not in set), accept.
        
        # To save time, we'll only try a subset of swaps.
        # Sort candidates by "potential" (e.g., SPF desc).
        
        # Let's try a simpler local move: Remove y, add x.
        # We accept the swap if the new set allows adding at least one more element than the old set did?
        # This is expensive to evaluate fully.
        
        # Alternative: Randomized local search?
        # Let's stick to deterministic: Try to replace the smallest prime factor element with a composite?
        
        # Actually, the set of all primes is likely optimal.
        # Let's just return the best greedy result.
        # If we have time, try a few random restarts of greedy with noise.
        
        if time.time() > deadline - 5:
            break
            
        # Try a few random restarts
        import random
        random.seed(42)
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            # Shuffle ordering slightly
            shuffled = list(all_nums)
            random.shuffle(shuffled)
            # Bias towards primes
            shuffled.sort(key=lambda x: (random.random() > 0.5 and len(pf[x])>1, -_smallest_prime_factor(x)))
            
            current = []
            for x in shuffled:
                if _can_add(x, current):
                    current.append(x)
            if len(current) > len(best_set):
                best_set = current
                improved = True
                break

    return list(best_set)
# EVOLVE-BLOCK-END
