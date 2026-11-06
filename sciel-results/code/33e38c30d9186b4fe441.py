"""Structural approach: greedy gap-closure with fast numpy coverage + local search."""
import time
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    n_primes = len(primes)  # 168
    primes_arr = np.array(primes)
    
    MAX_L = 20000  # coverage array size
    
    # Initialize residues
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Coverage array: count[x] = number of congruences satisfied by x
    coverage = np.zeros(MAX_L + 1, dtype=np.int32)
    
    def find_first_gap(cov):
        """Find the first position x >= 1 where cov[x] < 2."""
        # Use numpy to find first index where cov < 2
        deficient = np.where(cov[1:] < 2)[0]
        if len(deficient) == 0:
            return MAX_L  # all covered up to MAX_L
        return deficient[0] + 1
    
    def add_congruence(cov, p, r):
        """Add coverage for x ≡ r (mod p) to cov array."""
        if r == 0:
            start = p
        else:
            start = r
        if start <= MAX_L:
            cov[start::p] += 1
    
    def remove_congruence(cov, p, r):
        """Remove coverage for x ≡ r (mod p) from cov array."""
        if r == 0:
            start = p
        else:
            start = r
        if start <= MAX_L:
            cov[start::p] -= 1
    
    # Phase 1: Greedy gap-closure
    # Process primes in order of increasing size (small primes cover more positions)
    # For each prime, choose the residue that best helps close the current gap
    
    # Sort primes by size for processing order (small first)
    order = np.argsort(primes_arr)
    
    for idx in order:
        if time.time() > deadline - 60:
            break
        
        p = primes[idx]
        
        # Find current gap
        gap = find_first_gap(coverage)
        
        # If gap is beyond MAX_L, we're doing great
        if gap >= MAX_L:
            residues[idx] = 0
            add_congruence(coverage, p, 0)
            continue
        
        # For this prime, try to pick the residue that covers the gap
        # and maximizes coverage in the critical region [gap, gap + p + 50]
        
        best_r = 0
        best_score = -1
        
        # The gap position: we want x = gap to be covered
        # gap % p gives us a residue that covers gap
        target_r = gap % p
        
        # Try a focused set of candidates around target_r
        # and also try 0 (covers multiples of p)
        candidates = set()
        candidates.add(target_r)
        candidates.add(0)
        
        # Also try residues that cover nearby deficient positions
        for dx in range(-3, 4):
            x = gap + dx
            if 1 <= x <= MAX_L:
                candidates.add(x % p)
        
        # For small primes, try all residues
        if p <= 50:
            candidates = set(range(p))
        
        for r in candidates:
            # Simulate adding this residue
            # Count how many positions in [gap, gap + p + 100] would improve
            if r == 0:
                start = p
            else:
                start = r
            
            if start > MAX_L:
                score = 0
            else:
                # Get the positions this residue covers in the critical window
                end = min(MAX_L, gap + p + 100)
                # Positions: start, start+p, start+2p, ... up to end
                n_pos = (end - start) // p + 1 if start <= end else 0
                
                if n_pos <= 0:
                    score = 0
                else:
                    positions = np.arange(start, start + n_pos * p, p)
                    positions = positions[positions <= MAX_L]
                    
                    if len(positions) == 0:
                        score = 0
                    else:
                        # Score: prioritize positions that are currently 1 (become 2)
                        # and the gap position itself
                        cov_vals = coverage[positions]
                        
                        # Big bonus if gap is covered
                        if (gap % p == r) or (r == 0 and gap % p == 0):
                            # gap is covered by this residue
                            gap_bonus = 1000
                        else:
                            gap_bonus = 0
                        
                        # Count 1->2 transitions
                        one_to_two = np.sum(cov_vals == 1)
                        # Count 0->1 transitions (less important)
                        zero_to_one = np.sum(cov_vals == 0)
                        
                        score = gap_bonus + one_to_two * 10 + zero_to_one
                        
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = best_r
        add_congruence(coverage, p, best_r)
    
    # Phase 2: Local search - try changing individual residues to extend L
    current_L = find_first_gap(coverage) - 1
    if current_L < 1:
        current_L = 0
    
    improved = True
    max_iterations = 20
    iteration = 0
    
    while improved and time.time() < deadline - 30 and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Find the first gap
        gap = find_first_gap(coverage)
        if gap >= MAX_L:
            break
        
        # Try each prime: can we change its residue to help cover the gap?
        # Process primes in order of how many positions they cover in the critical region
        # (smaller primes cover more, so try them first)
        
        for idx in order:
            if time.time() > deadline - 30:
                break
            
            p = primes[idx]
            old_r = residues[idx]
            
            # The residue that would cover the gap
            target_r = gap % p
            
            # Try a few candidates
            candidates = {target_r, 0}
            # Add neighbors
            for dx in range(-2, 3):
                candidates.add((target_r + dx) % p)
            
            # Remove old_r to avoid no-op
            candidates.discard(old_r)
            
            best_r = old_r
            best_L = current_L
            
            for new_r in candidates:
                if time.time() > deadline - 30:
                    break
                
                # Try the change
                remove_congruence(coverage, p, old_r)
                add_congruence(coverage, p, new_r)
                
                new_L = find_first_gap(coverage) - 1
                if new_L < 0:
                    new_L = 0
                
                if new_L > best_L:
                    best_L = new_L
                    best_r = new_r
                else:
                    # Revert
                    remove_congruence(coverage, p, new_r)
                    add_congruence(coverage, p, old_r)
            
            if best_r != old_r:
                # Apply the best change
                if best_r != old_r:
                    # We already reverted in the loop if it wasn't best
                    # Need to apply best_r
                    remove_congruence(coverage, p, old_r)
                    add_congruence(coverage, p, best_r)
                    residues[idx] = best_r
                current_L = best_L
                improved = True
    
    # Phase 3: Simulated annealing / random restarts for local minima escape
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        
        best_L = current_L
        best_residues = residues.copy()
        
        T = 50.0  # initial temperature
        cooling = 0.95
        
        while time.time() < deadline - 5:
            # Pick a random prime and change its residue
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Apply
            remove_congruence(coverage, p, old_r)
            add_congruence(coverage, p, new_r)
            residues[idx] = new_r
            
            new_L = find_first_gap(coverage) - 1
            if new_L < 0:
                new_L = 0
            
            if new_L >= best_L:
                best_L = new_L
                best_residues = residues.copy()
                T = max(T * cooling, 1.0)
            else:
                # Accept with probability exp(-(best_L - new_L) / T)
                delta = best_L - new_L
                if delta > 0 and random.random() < np.exp(-delta / max(T, 0.1)):
                    T = max(T * cooling, 0.5)
                else:
                    # Revert
                    remove_congruence(coverage, p, new_r)
                    add_congruence(coverage, p, old_r)
                    residues[idx] = old_r
                    T = max(T * cooling, 0.5)
    
    # Restore best solution
    if best_L > current_L:
        # Rebuild coverage from best_residues
        coverage = np.zeros(MAX_L + 1, dtype=np.int32)
        for idx in range(n_primes):
            p = primes[idx]
            r = best_residues[idx]
            add_congruence(coverage, p, r)
        residues = best_residues
    
    # Final: return as list of ints
    return [int(r) for r in residues]
