import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Helper to compute covered prefix length for a given set of (p, r) pairs
    def compute_L(classes, max_check=10000):
        """Compute the largest L such that 1..L is all covered."""
        if not classes:
            return 0
        # Use a bitset approach for efficiency
        # We'll check up to max_check
        covered = [False] * (max_check + 1)
        for p, r in classes:
            start = r if r > 0 else p
            for k in range(start, max_check + 1, p):
                covered[k] = True
        
        L = 0
        for i in range(1, max_check + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    # Better: use numpy for fast coverage checking
    import numpy as np
    
    def compute_L_fast(classes, max_check=50000):
        """Fast computation using numpy."""
        if not classes:
            return 0
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            start = r if r > 0 else p
            if start <= max_check:
                covered[start-1::p] = True  # 0-indexed, so start-1
        
        # Find first False
        false_indices = np.where(~covered)[0]
        if len(false_indices) == 0:
            return max_check
        return false_indices[0]  # This is 0-indexed, so L = false_indices[0]

    # Generate candidate prime sets
    def build_prime_set(base_primes, fill_strategy='smallest_first'):
        """Build a prime set starting from base_primes, filling remaining budget."""
        s = sum(Fraction(1, p) for p in base_primes)
        if s > 1:
            return []
        final_set = list(base_primes)
        base_set = set(base_primes)
        
        if fill_strategy == 'smallest_first':
            candidates = [p for p in primes if p not in base_set]
        else:
            candidates = [p for p in primes if p not in base_set]
        
        for p in candidates:
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        
        return final_set

    # Candidate prime sets
    candidate_sets = []
    
    # Strategy 1: Include small primes for high density
    candidate_sets.append(build_prime_set([2]))
    candidate_sets.append(build_prime_set([3]))
    candidate_sets.append(build_prime_set([2, 3]))
    candidate_sets.append(build_prime_set([2, 5]))
    candidate_sets.append(build_prime_set([2, 7]))
    candidate_sets.append(build_prime_set([3, 5]))
    candidate_sets.append(build_prime_set([5, 7]))
    
    # Strategy 2: Skip some small primes to allow more medium primes
    candidate_sets.append(build_prime_set([2, 11]))
    candidate_sets.append(build_prime_set([2, 13]))
    candidate_sets.append(build_prime_set([3, 7]))
    candidate_sets.append(build_prime_set([5, 11]))
    
    # Strategy 3: No small primes, many medium/large
    candidate_sets.append(build_prime_set([]))
    
    # Remove empty sets
    candidate_sets = [s for s in candidate_sets if s]
    
    best_L = 0
    best_classes = []
    
    # For each prime set, optimize residues using hill-climbing
    for prime_set in candidate_sets:
        if time.time() > deadline - 20:
            break
            
        P = sorted(prime_set)
        n_primes = len(P)
        
        # Initial residues: random or based on covering 1..k
        # Start with residues that cover 1..n_primes
        residues = [0] * n_primes
        for i, p in enumerate(P):
            # Try to cover small numbers
            residues[i] = (i + 1) % p
        
        # Evaluate initial
        classes = [[P[i], residues[i]] for i in range(n_primes)]
        L = compute_L_fast(classes, max_check=10000)
        
        if L > best_L:
            best_L = L
            best_classes = classes
        
        # Hill-climbing: try changing one residue at a time
        improved = True
        max_iterations = 500
        iteration = 0
        
        while improved and iteration < max_iterations and time.time() < deadline - 15:
            improved = False
            iteration += 1
            
            # Try each prime's residue
            for i in range(n_primes):
                if time.time() > deadline - 15:
                    break
                    
                p = P[i]
                old_r = residues[i]
                
                # Try a few candidate residues
                # Focus on residues that might help cover gaps
                # Try residues that are small (to cover small numbers)
                candidates = set()
                candidates.add(old_r)
                
                # Try residues 0 to min(p-1, 50)
                for r in range(min(p, 50)):
                    candidates.add(r)
                
                # Also try random residues
                for _ in range(10):
                    candidates.add(random.randint(0, p - 1))
                
                best_r = old_r
                best_L_i = L
                
                for r in candidates:
                    if r == old_r:
                        continue
                    
                    residues[i] = r
                    classes = [[P[j], residues[j]] for j in range(n_primes)]
                    L_i = compute_L_fast(classes, max_check=10000)
                    
                    if L_i > best_L_i:
                        best_L_i = L_i
                        best_r = r
                    elif L_i == best_L_i and random.random() < 0.1:
                        # Occasionally accept equal to escape local minima
                        best_r = r
                        best_L_i = L_i
                
                if best_r != old_r:
                    residues[i] = best_r
                    L = best_L_i
                    improved = True
                    
                    if L > best_L:
                        best_L = L
                        best_classes = [[P[j], residues[j]] for j in range(n_primes)]
        
        # Also try a few random restarts for this prime set
        for restart in range(3):
            if time.time() > deadline - 10:
                break
                
            # Random initial residues
            residues = [random.randint(0, P[i] - 1) for i in range(n_primes)]
            classes = [[P[i], residues[i]] for i in range(n_primes)]
            L = compute_L_fast(classes, max_check=10000)
            
            if L > best_L:
                best_L = L
                best_classes = classes
            
            # Quick hill-climb
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                i = random.randint(0, n_primes - 1)
                p = P[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                
                residues[i] = new_r
                classes = [[P[j], residues[j]] for j in range(n_primes)]
                L_new = compute_L_fast(classes, max_check=10000)
                
                if L_new >= L:
                    L = L_new
                    if L > best_L:
                        best_L = L
                        best_classes = classes
                else:
                    residues[i] = old_r
    
    # Final verification with larger max_check
    if best_classes:
        final_L = compute_L_fast(best_classes, max_check=100000)
        if final_L > best_L:
            best_L = final_L
    
    return best_classes
