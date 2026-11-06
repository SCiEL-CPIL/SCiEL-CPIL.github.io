"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    deadline = time.time() + 295  # safety margin
    
    # Precompute primes up to N
    primes = _primes(N + 1)
    prime_set = set(primes)
    
    # Strategy: Greedy construction with careful admissibility maintenance
    # We build the set incrementally, ensuring at each step that the set remains admissible.
    
    # Key insight: For a set S of size k, we need for every prime p <= k,
    # there exists a residue class mod p that is NOT in {x % p : x in S}.
    
    # We'll use a greedy approach:
    # 1. Start with empty set
    # 2. At each step, try to add the smallest available number that maintains admissibility
    # 3. Use efficient tracking of which residues are covered for each relevant prime
    
    points = set()
    k = 0
    
    # Track empty residues for each prime
    # empty_res[p] = set of residues mod p NOT covered by points
    # Only need to track primes p <= k (since we only need p <= len(points))
    empty_res = {}
    
    def get_empty_res(p):
        """Compute empty residues for prime p given current points."""
        covered = set()
        for x in points:
            covered.add(x % p)
        return set(range(p)) - covered
    
    def is_valid_addition(x, new_k):
        """Check if adding x to points (making size new_k) maintains admissibility."""
        # Check all primes p <= new_k
        for p in primes:
            if p > new_k:
                break
            
            if p in empty_res:
                # We have cached info
                er = empty_res[p]
                if x % p in er:
                    # Adding x would cover this empty residue
                    new_er = er - {x % p}
                    if len(new_er) == 0:
                        # All residues would be covered - not admissible
                        return False
            else:
                # Need to compute
                er = get_empty_res(p)
                if x % p in er:
                    new_er = er - {x % p}
                    if len(new_er) == 0:
                        return False
        
        return True
    
    def update_after_add(x, new_k):
        """Update empty_res after adding x."""
        for p in primes:
            if p > new_k:
                break
            if p not in empty_res:
                empty_res[p] = get_empty_res(p)
            r = x % p
            if r in empty_res[p]:
                empty_res[p].discard(r)
    
    # Greedy: try to add numbers in increasing order
    # But also try to skip problematic ones
    
    for x in range(N + 1):
        if time.time() > deadline:
            break
        
        if x in points:
            continue
        
        new_k = k + 1
        
        if is_valid_addition(x, new_k):
            points.add(x)
            update_after_add(x, new_k)
            k = new_k
    
    return list(points)
# EVOLVE-BLOCK-END
