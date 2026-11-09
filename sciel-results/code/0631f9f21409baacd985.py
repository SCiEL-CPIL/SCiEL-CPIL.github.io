import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    L_target = 5000  # Search space upper bound for evaluation
    
    def evaluate(residues):
        """
        Returns the largest L such that every integer 1..L is covered by at least 2 primes.
        """
        counts = np.zeros(L_target + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            # Add 1 to all indices x in [1, L_target] such that x % p == r
            # If r == 0, indices are p, 2p, ... (and 0, which we ignore)
            # If r > 0, indices are r, r+p, ...
            counts[r::p] += 1
        
        # Find the first index >= 1 where count < 2
        bad_indices = np.where(counts[1:] < 2)[0]
        if len(bad_indices) == 0:
            return L_target
        return bad_indices[0]

    # Initialization
    random.seed(42)
    residues = [random.randint(0, primes[i] - 1) for i in range(n_primes)]
    
    current_L = evaluate(residues)
    best_L = current_L
    best_residues = residues[:]

    no_improve_steps = 0
    max_no_improve = 2000 # Threshold for aggressive restart

    while time.time() < deadline - 2.0:
        if best_L >= L_target:
            break
            
        x_bad = best_L + 1
        
        # Try to fix x_bad by changing one prime's residue
        # We try a few random primes
        found_improvement = False
        for _ in range(10): # Try up to 10 random primes per iteration
            if time.time() > deadline - 2.0:
                break
                
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            
            # Set residue to cover x_bad
            new_r = x_bad % p
            
            if new_r == old_r:
                continue
                
            residues[idx] = new_r
            L = evaluate(residues)
            
            if L > best_L:
                best_L = L
                best_residues = residues[:]
                found_improvement = True
                no_improve_steps = 0
                break
            else:
                # Revert
                residues[idx] = old_r
                
        if found_improvement:
            continue
            
        no_improve_steps += 1
        
        if no_improve_steps > max_no_improve:
            # Aggressive perturbation: change several random residues
            num_changes = random.randint(3, 8)
            for _ in range(num_changes):
                idx = random.randint(0, n_primes - 1)
                residues[idx] = random.randint(0, primes[idx] - 1)
            
            L = evaluate(residues)
            if L > best_L:
                best_L = L
                best_residues = residues[:]
            else:
                # Revert to best known state
                residues = best_residues[:]
            
            no_improve_steps = 0

    return best_residues
