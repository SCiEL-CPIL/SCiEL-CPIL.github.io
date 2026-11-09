import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    # Step 1: Generate primes up to 1000
    def _primes_up_to(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, v in enumerate(sieve) if v]
    
    primes = _primes_up_to(1000)
    prime_set = set(primes)
    
    # Step 2: Greedy set-cover to select prime-residue classes
    # We want to cover 1..L maximally. Use a coverage array.
    MAX_L = 100000
    covered = bytearray(MAX_L + 1)  # 1-indexed, 0 means uncovered
    
    def compute_prefix(covered_arr):
        """Find largest L such that 1..L all covered."""
        L = 0
        for i in range(1, len(covered_arr)):
            if covered_arr[i]:
                L = i
            else:
                break
        return L
    
    # Budget tracking with exact rationals
    budget = Fraction(1, 1)
    selected = []  # list of (p, r)
    
    # Greedy: at each step, try each unused prime p and each residue r,
    # pick the (p, r) that maximizes the new covered prefix, subject to budget.
    # But trying all residues for all primes each step is expensive.
    # Optimization: for each prime p, the best residue is the one that covers
    # the first uncovered number. So we only need to check one candidate per prime.
    
    unused_primes = set(primes)
    
    while True:
        if time.time() > DEADLINE - 5.0:
            break
        
        best_gain = 0
        best_p = None
        best_r = None
        current_prefix = compute_prefix(covered)
        
        for p in unused_primes:
            if budget < Fraction(1, p):
                continue
            
            # Find the first uncovered number >= 1
            # The best residue for p is r such that r ≡ first_uncovered (mod p)
            # i.e., r = first_uncovered % p
            # This will cover first_uncovered and then every p-th number after.
            
            # But we need to find the first uncovered number
            # For efficiency, track it
            first_uncovered = -1
            for i in range(1, min(current_prefix + 1, MAX_L)):
                if not covered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                # All covered up to current_prefix, check beyond
                for i in range(current_prefix + 1, MAX_L + 1):
                    if not covered[i]:
                        first_uncovered = i
                        break
                if first_uncovered == -1:
                    break
            
            r = first_uncovered % p
            
            # Simulate adding (p, r): count how many new positions in 1..MAX_L get covered
            # and compute new prefix
            new_covered_count = 0
            new_prefix = current_prefix
            # Mark positions: r, r+p, r+2p, ... in 1..MAX_L
            # But we need to compute new prefix efficiently
            # For now, just estimate gain by checking how far the prefix extends
            # Actually, let's do a quick simulation for the prefix only
            
            # Check if first_uncovered is covered (it is, by construction)
            # Then check subsequent numbers
            temp_prefix = first_uncovered
            i = first_uncovered + 1
            while i <= MAX_L:
                # Check if i is covered by existing covered array OR by new class (p, r)
                if covered[i] or (i % p == r):
                    temp_prefix = i
                    i += 1
                else:
                    break
            
            gain = temp_prefix - current_prefix
            if gain > best_gain:
                best_gain = gain
                best_p = p
                best_r = r
        
        if best_p is None or best_gain <= 0:
            break
        
        # Commit the best choice
        p, r = best_p, best_r
        selected.append((p, r))
        budget -= Fraction(1, p)
        unused_primes.remove(p)
        
        # Update covered array
        for i in range(r if r > 0 else p, MAX_L + 1, p):
            covered[i] = 1
    
    # Step 3: Residue optimization pass
    # For each selected prime, try shifting its residue to maximize the covered prefix.
    # Use incremental coverage: for each prime, try all residues and pick the best.
    
    def recompute_prefix_from_classes(classes):
        """Compute covered prefix given list of (p, r) classes."""
        cov = bytearray(MAX_L + 1)
        for p, r in classes:
            start = r if r > 0 else p
            for i in range(start, MAX_L + 1, p):
                cov[i] = 1
        return compute_prefix(cov)
    
    # Try to improve by adjusting residues
    for _round in range(3):  # multiple passes
        if time.time() > DEADLINE - 3.0:
            break
        improved = False
        for idx in range(len(selected)):
            if time.time() > DEADLINE - 3.0:
                break
            p, old_r = selected[idx]
            best_r = old_r
            best_prefix = -1
            
            # Try all residues for this prime
            for r in range(p):
                # Temporarily change residue
                selected[idx] = (p, r)
                prefix = recompute_prefix_from_classes(selected)
                if prefix > best_prefix:
                    best_prefix = prefix
                    best_r = r
                if time.time() > DEADLINE - 3.0:
                    break
            
            selected[idx] = (p, best_r)
            if best_r != old_r:
                improved = True
        
        if not improved:
            break
    
    # Final: return as [[p, r], ...]
    return [[p, r] for p, r in selected]
