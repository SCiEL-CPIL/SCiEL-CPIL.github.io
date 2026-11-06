"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

# Constants
MAX_L = 100000
DEADLINE = 115  # seconds, safety margin before 120s kill

def _primes_upto(n):
    """Sieve of Eratosthenes for primes up to n."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _coverage_length(classes, cap=MAX_L):
    """Compute the largest L such that every integer 1..L is covered.
    classes: list of [p, r]
    Returns L (capped at cap)."""
    if not classes:
        return 0
    # Create a coverage boolean array
    covered = bytearray(cap + 1)  # index 0 unused
    for p, r in classes:
        # Mark all numbers n in [1, cap] where n % p == r
        # Start at the smallest n >= 1 with n % p == r
        if r == 0:
            start = p
        else:
            start = r
            if start < 1:
                start += p
        for n in range(start, cap + 1, p):
            covered[n] = 1
    # Find the first gap
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _sum_recip(classes):
    """Sum of 1/p for classes, as Fraction."""
    s = Fraction(0)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def construct():
    start_time = time.time()
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Precompute for each prime p and residue r, the bitset of coverage in [1, MAX_L]
    # This is expensive, so we'll do it lazily or use a different approach.
    
    # Strategy: Greedy set cover with bitsets
    # Represent coverage as a bitset (Python int) for [1, MAX_L]
    # For each (p, r), compute the bitset of numbers covered.
    
    # Precompute bitsets for all (p, r) pairs
    # This is 168 primes * average ~500 residues = ~84000 bitsets of 100000 bits
    # That's about 84000 * 12.5KB = ~1GB. Too much.
    
    # Alternative: For each prime p, precompute the coverage pattern as a periodic function.
    # Use numpy for speed.
    
    import numpy as np
    
    # Precompute coverage arrays for each prime
    # For prime p, the coverage for residue r is: positions where i % p == r
    # We can represent this as a boolean array of length MAX_L+1
    
    # But storing all is too much. Instead, compute on the fly during greedy.
    
    # Greedy approach:
    # 1. Start with empty set of classes.
    # 2. At each step, for each unused prime p, try all residues r and pick the one
    #    that covers the most currently uncovered integers in [1, current_L+1].
    # 3. Add that class, update coverage.
    # 4. Stop when sum of reciprocals would exceed 1, or no improvement.
    
    # For efficiency, track coverage as a boolean numpy array.
    
    covered = np.zeros(MAX_L + 1, dtype=bool)
    selected = []  # list of (p, r)
    used_primes = set()
    budget = Fraction(1)
    
    # Precompute primes sorted by size (smaller primes cover more, so prefer them)
    primes_sorted = sorted(primes)
    
    # For greedy, we need to evaluate all (p, r) for unused primes.
    # This is expensive. Let's limit to a reasonable number of iterations.
    
    max_iterations = 100
    
    for it in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            break
        
        best_gain = -1
        best_p = None
        best_r = None
        best_cost = None
        
        for p in primes_sorted:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget - cost < 0:
                continue
            
            # For this prime, find the residue that maximizes coverage of uncovered numbers
            # The uncovered numbers are those where covered[i] == False
            # We want to maximize count of i in [1, MAX_L] where not covered[i] and i % p == r
            
            # Efficiently: for each residue r, count uncovered numbers with i % p == r
            # We can do this by iterating over uncovered numbers and tallying by residue.
            
            # Get uncovered indices
            uncovered_idx = np.where(~covered[1:MAX_L+1])[0] + 1
            if len(uncovered_idx) == 0:
                break
            
            # For each uncovered number, compute its residue mod p
            residues = uncovered_idx % p
            # Count occurrences of each residue
            if len(residues) > 0:
                # Use bincount for speed
                counts = np.bincount(residues, minlength=p)
                # Find the residue with max count
                best_r_local = int(np.argmax(counts))
                gain = int(counts[best_r_local])
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = best_r_local
                    best_cost = cost
        
        if best_p is None or best_gain <= 0:
            break
        
        # Add the best class
        selected.append((best_p, best_r))
        used_primes.add(best_p)
        budget -= best_cost
        
        # Update coverage
        p, r = best_p, best_r
        if r == 0:
            start = p
        else:
            start = r
            if start < 1:
                start += p
        for n in range(start, MAX_L + 1, p):
            covered[n] = True
    
    # Local search: try to swap residues to improve coverage
    # For each selected class, try changing its residue and see if coverage improves
    
    if time.time() - start_time < DEADLINE:
        # Try local search
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            for idx in range(len(selected)):
                if time.time() - start_time > DEADLINE:
                    break
                p, r = selected[idx]
                # Try all other residues
                for r_new in range(p):
                    if r_new == r:
                        continue
                    # Compute the change in coverage
                    # Old coverage: numbers with i % p == r
                    # New coverage: numbers with i % p == r_new
                    # Net change: numbers covered by new but not old, minus numbers covered by old but not new
                    # A number is "lost" if it's covered only by this class (no other class covers it)
                    # A number is "gained" if it's not covered at all currently
                    
                    # This is complex. Simplify: just check if the new residue covers more uncovered numbers
                    # and loses fewer covered-only-by-this-class numbers.
                    
                    # For simplicity, just check if gain > loss where:
                    # gain = count of i with i%p==r_new and not covered[i] (before removal)
                    # loss = count of i with i%p==r and not covered[i] (after removal, i.e., only this class covers them)
                    
                    # This requires knowing which numbers are covered only by this class.
                    # Too expensive. Skip local search for now.
                    pass
            # Skip complex local search due to time constraints
    
    # Compute final coverage length
    L = _coverage_length([[p, r] for p, r in selected])
    
    # Convert to required format
    classes = [[p, r] for p, r in selected]
    
    # Verify constraints
    assert len(set(p for p, r in classes)) == len(classes), "Duplicate primes"
    for p, r in classes:
        assert p <= 1000, f"Prime {p} > 1000"
        assert 0 <= r < p, f"Residue {r} not in [0, {p})"
        assert p in primes, f"{p} is not prime"
    total = _sum_recip(classes)
    assert total <= 1, f"Sum of reciprocals {total} > 1"
    
    return classes

# EVOLVE-BLOCK-END
