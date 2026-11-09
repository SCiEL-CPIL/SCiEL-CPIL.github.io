"""Improved program: greedy pairing to maximize double-coverage length."""
import time

def _get_primes(n):
    """Return all primes <= n in increasing order."""
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(residues, primes, cap=100000):
    """Compute the largest L such that every integer 1..L satisfies >= 2 congruences."""
    # For each prime p with residue r, the covered integers in [1, cap] are r, r+p, r+2p, ...
    # We need count >= 2 for each x in 1..L
    # Use a difference array approach for efficiency
    n = len(primes)
    # Build coverage count array
    # For each (p, r): positions are r, r+p, r+2p, ... up to cap
    # But r could be 0, meaning multiples of p
    # We need to count for each x in 1..cap how many (p,r) have x ≡ r (mod p)
    
    # Efficient approach: for each prime, mark its covered positions
    # Using numpy would be faster but let's use a clever approach
    # 
    # Alternative: iterate x from 1 upward, count coverage, stop when count < 2
    # For each x, count how many primes p have x % p == r_p
    # This is O(L * n) which for L up to 10000 and n=168 is ~1.68M ops, fine
    
    L = 0
    for x in range(1, cap + 1):
        count = 0
        for i in range(n):
            if x % primes[i] == residues[i]:
                count += 1
                if count >= 2:
                    break
        if count >= 2:
            L = x
        else:
            break
    return L

def _greedy_pairing(primes, time_limit=140):
    """
    Greedy approach: pair primes to cover consecutive integers.
    For each target integer t (starting from 1), find two unused primes p1, p2
    such that we can set residues to cover t (and ideally nearby integers).
    """
    n = len(primes)
    residues = [0] * n
    used = [False] * n
    
    # Sort primes: we want to use smaller primes for smaller targets
    # because smaller primes have fewer residue options
    # Actually, let's think differently:
    # For target t, prime p covers t if r_p = t % p
    # We need to assign r_p for each prime
    # Strategy: process targets t = 1, 2, 3, ... and for each, pick 2 unused primes
    # that can cover t, preferring primes where t % p is "natural"
    
    # Better strategy: 
    # Assign residues greedily. For each prime in order, choose a residue that
    # maximizes coverage of as-yet-underserved integers.
    
    # Let me try a different approach: 
    # Pair primes (p_i, p_{i+1}) to cover integers around a target.
    # For target t, use primes p and q: set r_p = t % p, r_q = t % q
    # This covers t. The next covered integers are t+p, t+q, etc.
    
    # Simplest effective approach: 
    # Process primes in increasing order. For prime p_i, assign residue to cover
    # the smallest integer not yet doubly covered.
    
    # Actually, let's use the pairing approach more carefully:
    # We have 168 primes. Pair them: (p_1,p_2), (p_3,p_4), ..., (p_167,p_168)
    # Pair i covers integer t_i. Choose t_i to maximize coverage.
    
    # For pair (p, q) covering target t:
    #   r_p = t % p, r_q = t % q
    #   This covers: t, t+p, t+2p, ... and t, t+q, t+2q, ...
    #   The "doubly covered" integers near t include t itself.
    #   Integers t+1, t+2, ... are covered by this pair only if they match
    #   one of these arithmetic progressions.
    
    # To maximize consecutive double coverage, I want the pairs to cover
    # consecutive integers 1, 2, 3, ...
    # Pair i covers integer i. But then pair (p,q) with target i covers i,
    # and the next integers covered by this pair are i+p and i+q (single coverage).
    # For double coverage of i+1, i+2, etc., we need other pairs.
    
    # Key realization: if I assign pair j to cover integer j, then:
    # - Integer j is covered by pair j (both primes) -> double covered
    # - But integer j+1 might only be covered by pair j+1
    # - So every integer j gets double coverage from its own pair!
    # 
    # Wait, that's not right. Pair j covers integer j with BOTH its primes.
    # So integer j has at least 2 covers (from pair j's two primes).
    # But does it have coverage from other pairs? Maybe, maybe not.
    # The key is: as long as each integer j is the target of exactly one pair,
    # it gets double coverage from that pair.
    #
    # So if I can assign 168 pairs to targets 1, 2, ..., 168, then
    # integers 1 through 168 are all doubly covered!
    #
    # But wait - I need to verify: does assigning r_p = j % p for pair (p,q) 
    # targeting j cause conflicts? No, each prime is in exactly one pair,
    # so each prime gets exactly one residue assignment.
    #
    # So the simple strategy: pair consecutive primes, assign pair i to target i.
    # This gives L >= 168.
    
    # But can we do better? The pairs also cover other integers.
    # Pair (p,q) targeting j also covers j+p, j+q, j+2p, etc.
    # These might help extend coverage beyond 168.
    
    # Let's also consider: can we choose targets more cleverly?
    # If pair (p,q) targets j, it covers j, j+p, j+q, j+2p, j+2q, ...
    # For the "gap" between j and j+min(p,q), only this pair provides coverage.
    # For double coverage in that gap, we'd need another pair to also cover those integers.
    
    # For now, the basic pairing gives L >= 168. Let's implement that and then
    # try to extend by choosing targets optimally.
    
    # Basic: pair i (0-indexed) -> primes[2*i], primes[2*i+1], target = i+1
    for i in range(n // 2):
        p = primes[2 * i]
        q = primes[2 * i + 1]
        t = i + 1  # target integer
        residues[2 * i] = t % p
        residues[2 * i + 1] = t % q
    
    # Now try to extend: check how far the double coverage goes
    # and see if we can adjust targets to extend further
    
    return residues

def _improve_targets(primes, residues, time_limit=140):
    """Try to improve the target assignments to extend double coverage."""
    n = len(primes)
    num_pairs = n // 2
    start_time = time.time()
    
    # Current targets: pair i targets i+1
    # Try shifting some targets to extend coverage
    # 
    # The limiting factor is where double coverage breaks.
    # Let's first compute current L, then try local search.
    
    current_L = _compute_L(residues, primes)
    best_L = current_L
    best_residues = residues[:]
    
    # Local search: try changing individual pair targets
    # For pair i, changing target from t to t' changes residues
    # Try targets in a range around current
    
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        for i in range(num_pairs):
            if time.time() - start_time > time_limit:
                break
            p = primes[2 * i]
            q = primes[2 * i + 1]
            current_t = i + 1
            
            # Try nearby targets
            for dt in range(-5, 6):
                if dt == 0:
                    continue
                new_t = current_t + dt
                if new_t < 1:
                    continue
                # Save old residues
                old_rp = residues[2 * i]
                old_rq = residues[2 * i + 1]
                # Set new residues
                residues[2 * i] = new_t % p
                residues[2 * i + 1] = new_t % q
                # Evaluate
                L = _compute_L(residues, primes)
                if L > best_L:
                    best_L = L
                    best_residues = residues[:]
                    improved = True
                # Restore
                residues[2 * i] = old_rp
                residues[2 * i + 1] = old_rq
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    time_limit = 140  # safety margin before 150s kill
    
    primes = _get_primes(1000)
    n = len(primes)
    
    # Basic greedy pairing
    residues = _greedy_pairing(primes, time_limit)
    
    # Compute initial L
    L = _compute_L(residues, primes)
    
    # Try to improve with local search
    if time.time() - start_time < time_limit - 5:
        residues, L = _improve_targets(primes, residues, time_limit - (time.time() - start_time))
    
    # Final check
    L = _compute_L(residues, primes)
    
    return residues

# EVOLVE-BLOCK-START
# (construct is defined above, this is the entry point)
# EVOLVE-BLOCK-END
