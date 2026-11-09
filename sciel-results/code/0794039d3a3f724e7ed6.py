import time
import math

N = 5000

def _primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

# Precompute primes up to N once
_PRIMES = _primes(N)

def construct():
    start = time.time()
    DEADLINE = start + 280.0  # safety margin under 300s

    # Start with all integers in [0, N]
    points = list(range(N + 1))

    # Greedy Repair Phase
    for p in _PRIMES:
        if time.time() > DEADLINE:
            break
        L = len(points)
        if p > L:
            break
        # Count residues mod p
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        if min(counts) > 0:
            # All residues present -> remove least-populated class
            rmin = min(range(p), key=lambda r: counts[r])
            points = [v for v in points if v % p != rmin]

    # Polish Phase: Try to add back elements
    L = len(points)
    points_set = set(points)
    missing_elements = [x for x in range(N + 1) if x not in points_set]
    
    # Initialize missing residue sets for current L
    # We need to track which residues are missing for each prime p <= L
    # missing_sets[p] = set of residues r in [0, p-1] such that no v in points has v % p == r
    missing_sets = {}
    for p in _PRIMES:
        if p > L:
            break
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        missing_sets[p] = {r for r in range(p) if counts[r] == 0}

    # Sort missing elements to try adding back.
    # Strategy: Try adding elements that are "least likely" to cause conflicts?
    # Or just iterate. Randomization might help escape local minima, but deterministic is safer for reproducibility unless we have time for multiple restarts.
    # Let's try increasing order first.
    
    for x in missing_elements:
        if time.time() > DEADLINE:
            break
        
        # Check if adding x is valid
        valid = True
        # We need to check primes p <= L + 1
        # Because L will increase by 1, so the new constraint applies to p <= L + 1
        for p in _PRIMES:
            if p > L + 1:
                break
            
            # If p is not in missing_sets, it means p > old L, so it's a new prime constraint
            # But wait, if p <= L, it is in missing_sets.
            # If p == L + 1, it might be new.
            
            if p in missing_sets:
                r = x % p
                # If r is missing, and it's the ONLY missing one, adding x fills it -> Invalid
                if r in missing_sets[p] and len(missing_sets[p]) == 1:
                    valid = False
                    break
            else:
                # p is a new prime (p == L + 1 and prime)
                # We need to check if adding x creates a "full" set for this prime?
                # No, for a new prime p, the current set `points` has size L < p.
                # So it's impossible to have all p residues present.
                # So any x is valid with respect to this new prime p.
                # We just need to initialize the missing set for p if we add x.
                # But we only update missing_sets if we actually add x.
                pass

        if valid:
            # Add x to points
            points.append(x)
            points_set.add(x)
            L += 1
            
            # Update missing_sets
            for p in _PRIMES:
                if p > L:
                    break
                
                if p in missing_sets:
                    r = x % p
                    if r in missing_sets[p]:
                        missing_sets[p].discard(r)
                else:
                    # Initialize missing set for this new prime p
                    # This happens if p == L (the new length) and p is prime
                    # Or if we skipped it before? No, we iterate primes.
                    # If p was not in missing_sets, it means p > old L.
                    # Now p <= L.
                    # We need to build the missing set for p from scratch?
                    # That's expensive.
                    # But this only happens when L increases to a prime number.
                    # How often does that happen? ~ L / log L times.
                    # Cost to build: O(L * p).
                    # Total cost sum over primes p <= L of L*p?
                    # This might be okay.
                    
                    counts = [0] * p
                    for v in points:
                        counts[v % p] += 1
                    missing_sets[p] = {r for r in range(p) if counts[r] == 0}

    return points
