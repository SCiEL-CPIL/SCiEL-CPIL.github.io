"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    # Deadline guard: 150s limit, use 145s for safety
    deadline = time.time() + 145.0
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We need to find residues r_p for each prime p.
    # Score L: largest L such that every integer 1..L has coverage >= 2.
    # Coverage(x) = count of primes p such that x % p == r_p.
    
    # Initial heuristic: 
    # We want small numbers to be covered by many primes.
    # If we set r_p = x mod p for many p, x gets high coverage.
    # Let's start with r_p = 1 mod p for all p. 
    # Then 1 is covered by all 168 primes.
    # 2 is covered by all odd primes (since 1 != 0 mod p for odd p, wait. 
    # If r_p = 1, then x=1 is 1 mod p. x=2 is 2 mod p.
    # So 1 is covered by all. 2 is covered by primes where 2 mod p == 1 => p divides 1 => none? No.
    # Condition: x % p == r_p.
    # If r_p = 1, then x=1 works for all p.
    # x=2 works if 2 % p == 1 => p | 1. No prime divides 1. So coverage(2) = 0.
    # This is bad.
    
    # Better initial state:
    # Assign r_p randomly, but bias towards small residues?
    # Or, use the "stacking" idea:
    # Let's try to make the first few integers have high coverage.
    # For integer k, if we want it covered, we need r_p = k % p.
    # If we set r_p = k % p for all p, then k is covered by all primes.
    # But we can only have one r_p.
    # So we must trade off.
    
    # Strategy: Start with r_p = 1 % p for all p.
    # This gives coverage(1) = 168.
    # coverage(2): 2 % p == 1 ? Only if p=1 (not prime). So 0.
    # This is very poor.
    
    # Let's try r_p = 0 for all p.
    # coverage(x) = number of primes dividing x.
    # 1: 0. 2: 1. 3: 1. 4: 1. 6: 2.
    # L would be 0 because 1 has 0 coverage.
    
    # Let's try a mix.
    # Random initialization is a safe start for SA.
    residues = [random.randint(0, p - 1) for p in primes]
    
    # To speed up coverage calculation, we precompute the "hits" for each residue?
    # No, we update incrementally.
    
    # Coverage array size: We care about L up to maybe 1000 or 2000.
    # The task says L >= 1000 is the goal. 
    # Let's track coverage up to MAX_X = 2000 to be safe.
    MAX_X = 2000
    coverage = [0] * (MAX_X + 1)
    
    def update_coverage(idx, old_r, new_r, sign):
        """
        Update coverage array when prime at index idx changes residue from old_r to new_r.
        sign = +1 for add, -1 for remove.
        """
        p = primes[idx]
        # The numbers x in [1, MAX_X] such that x % p == old_r
        # x = old_r, old_r + p, old_r + 2p, ...
        # Note: if old_r == 0, x = p, 2p, ... (since x>=1, 0 is not in range)
        
        # Remove old
        if old_r > 0:
            start = old_r
        else:
            start = p
            
        x = start
        while x <= MAX_X:
            coverage[x] -= sign # Wait, sign is for the change. 
            # Let's be precise.
            # If we are changing from old to new:
            # We subtract contributions of old, add contributions of new.
            pass # Logic handled outside for clarity in loop below
        
    # Let's write a direct re-calculation for simplicity first, then optimize if slow.
    # 168 primes, MAX_X=2000. 
    # Full recalc: 168 * (2000/p) ops. Sum(2000/p) approx 2000 * log(168) ~ 2000*5 = 10000 ops.
    # 10,000 ops is very fast in Python (< 1ms).
    # So we can just recalculate coverage from scratch on every move?
    # 140 seconds / 1ms per move = 140,000 moves. That's plenty.
    
    def calculate_coverage(residues):
        cov = [0] * (MAX_X + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            x = start
            while x <= MAX_X:
                cov[x] += 1
                x += p
        return cov

    def get_L(cov):
        L = 0
        for i in range(1, MAX_X + 1):
            if cov[i] >= 2:
                L = i
            else:
                break
        return L

    # Initial state
    best_residues = residues[:]
    cov = calculate_coverage(residues)
    best_L = get_L(cov)
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() < deadline and T > T_min:
        # Choose a move
        # 1. Random single change
        # 2. Targeted change: Pick a prime, change its residue to cover the current gap
        #    Or pick a prime that covers the gap, and change it to cover something else? No, that breaks gap.
        #    Better: Identify gap G (first i with cov[i] < 2).
        #    Pick a random prime p.
        #    Set r_p = G % p. This ensures G is covered by p.
        #    This might fix G, but might break other numbers covered by p previously.
        
        gap = best_L + 1
        # If gap > MAX_X, we are done or need to expand MAX_X? 
        # Task says cap at 10^5 for score, but L>=1000 is the target.
        # If we reach 2000, we can try to extend, but let's stick to 2000 for now.
        
        move_type = random.random()
        new_residues = residues[:]
        
        if move_type < 0.5:
            # Random change
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            new_r = random.randint(0, p - 1)
            new_residues[idx] = new_r
        else:
            # Targeted change: Force coverage of the gap
            # Pick a random prime index
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            # Set residue so that gap is covered
            new_r = gap % p
            new_residues[idx] = new_r
            
        # Evaluate new state
        new_cov = calculate_coverage(new_residues)
        new_L = get_L(new_cov)
        
        # Acceptance probability
        delta = new_L - best_L
        if delta > 0:
            # Always accept improvement
            residues = new_residues
            cov = new_cov
            best_L = new_L
            best_residues = residues[:]
        else:
            # Accept worse state with probability exp(delta / T)
            # delta is negative
            prob = math.exp(delta / T)
            if random.random() < prob:
                residues = new_residues
                cov = new_cov
                # Do not update best_residues if it's worse, unless we want to keep track of current?
                # SA usually keeps current state, but we want to return best found.
                # So we track best_residues separately.
                
        T *= alpha

    # Return the best residues found
    return best_residues
# EVOLVE-BLOCK-END
