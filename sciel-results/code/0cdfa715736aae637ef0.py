"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    # Also check: for pairs (a, b) in members, candidate as a
    # candidate does not divide a*b for any distinct a,b in members
    for i in range(len(members)):
        a = members[i]
        for j in range(i + 1, len(members)):
            b = members[j]
            if (a * b) % candidate == 0:
                return False
    return True

def construct():
    deadline = time.time() + 50  # safety margin
    
    # Start with all primes in 2..256
    n = P['n']
    primes = []
    for a in range(2, n + 1):
        is_p = True
        for d in range(2, isqrt(a) + 1):
            if a % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(a)
    
    best = list(primes)
    
    # Get all composites in 2..256
    composites = [a for a in range(2, n + 1) if a not in set(primes)]
    
    # For each composite, count how many conflicts it has with the prime set
    # A composite x conflicts with primes p1, p2 if p1|x or p2|x (meaning x divides p1*p2 is impossible for primes,
    # but the real constraint is: for distinct a,b in set, a | b*c)
    # 
    # Let's think more carefully. If we have set S and want to add x:
    # - For any a in S: a must not divide x * b for any b in S, b != a
    # - For any a,b in S: x must not divide a * b
    #
    # For x composite with prime factors {p1, p2, ...}:
    # - If p1 is in S, then p1 | x * p1 (trivially, since p1|x), so p1 divides x*p1.
    #   Wait, we need p1 | x * b where b is another member. If b = p1, that's not distinct.
    #   If b = p2 (another prime in S), then p1 | x * p2. Since p1 | x, yes p1 | x*p2. CONFLICT!
    #
    # So a composite x with prime factor p (where p is in the prime set) will conflict:
    # p | x * q for any other prime q in the set (since p|x).
    #
    # This means: if x has a prime factor p that's in our set, we CANNOT have any other member q
    # such that p | x*q. Since p|x, p always divides x*q. So p conflicts with ALL other members.
    #
    # Wait, that means we can't add ANY composite that has a prime factor in the prime set!
    # Because if x has factor p (prime in set), then for any other member q in set, p | x*q.
    #
    # Unless... we remove some primes. Let me reconsider.
    #
    # Actually, the constraint is: for three DISTINCT a,b,c, a does not divide b*c.
    # So if a=p (prime), b=x (composite with p|x), c=q (another prime), then p | x*q since p|x.
    # This is a violation.
    #
    # So indeed, if we keep all primes, we can't add any composite.
    #
    # To add composites, we need to remove some primes. The question is: what's the optimal trade-off?
    #
    # Strategy: Try different subsets. Let's try a greedy approach where we consider all numbers
    # and use a conflict-based ordering.
    
    # Alternative approach: use a different construction.
    # Consider numbers that are products of large primes only.
    # Or consider: what if we use numbers > 128? Then for a > 128, b*c where b,c <= 256...
    # a | b*c means b*c >= a and a divides b*c. If a > 128 and b,c < a, then b*c could be >= a.
    #
    # Let me try a completely different approach: greedy with conflict counting.
    
    all_nums = list(range(2, n + 1))
    
    # Precompute prime factors for all numbers
    pf = {}
    for x in all_nums:
        pf[x] = _prime_factors(x)
    
    # Greedy: order by number of "conflicts" (how many pairs would be violated)
    # For each candidate x, count how many existing pairs (a,b) would be violated by adding x
    # This is expensive, so let's use a simpler heuristic.
    
    # Heuristic: numbers with fewer distinct prime factors are "safer"
    # because they divide fewer products.
    
    # Let's try: sort by (number of prime factors, value) and greedily add
    # But we need to be smarter.
    
    # Key insight: if x is composite with prime factors {p1, p2}, then:
    # - p1 | x * anything, so if p1 is in the set, conflict
    # - x | a*b only if all prime factors of x divide a*b
    #
    # Best strategy: pick numbers that share NO prime factors with other picked numbers.
    # This is like a matching/packing problem.
    
    # Let's try: group numbers by their prime factor sets, and pick one from each "compatible" group.
    
    # Actually, let me try a direct greedy with a good ordering.
    # Order: prefer numbers with larger smallest prime factor (less likely to divide products)
    # and fewer prime factors.
    
    def _smallest_prime_factor(x):
        if x % 2 == 0: return 2
        if x % 3 == 0: return 3
        d = 5
        while d * d <= x:
            if x % d == 0:
                return d
            d += 2
        return x
    
    # Sort candidates: prefer primes (safe), then composites with large smallest prime factor
    # Actually, let's just do a greedy with a scoring function.
    
    # Score for adding x to current set S:
    # - Penalty: number of (a,b) in S where a|x*b or b|x*a or x|a*b
    # - We want to minimize penalty; only add if penalty = 0
    
    # Let's try multiple orderings and take the best.
    
    orderings = [
        # Ordering 1: by smallest prime factor descending, then value descending
        sorted(all_nums, key=lambda x: (-_smallest_prime_factor(x), -x)),
        # Ordering 2: by number of prime factors ascending, then value descending  
        sorted(all_nums, key=lambda x: (len(pf[x]), -x)),
        # Ordering 3: by value descending
        sorted(all_nums, key=lambda x: -x),
        # Ordering 4: primes first, then composites by smallest prime factor desc
        sorted(all_nums, key=lambda x: (0 if len(pf[x])==1 and x==min(pf[x]) else 1, -_smallest_prime_factor(x), -x)),
    ]
    
    best_set = best  # primes
    
    for ordering in orderings:
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
    
    # Try to improve: remove some elements and add others
    # This is a local search
    
    # For now, return the best found
    return list(best_set)
# EVOLVE-BLOCK-END
