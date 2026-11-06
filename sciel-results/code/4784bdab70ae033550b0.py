import numpy as np
import time
import math

def construct():
    # Generate primes <= 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Search range for L
    N = 5000
    
    best_residues = None
    best_score = 0
    
    start_time = time.time()
    max_time = 110 # seconds
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.1
    alpha = 0.9995
    
    while time.time() - start_time < max_time:
        # Initialize residues
        residues = np.array([np.random.randint(0, p) for p in primes])
        
        # Calculate initial counts
        counts = np.zeros(N, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
            
        # Calculate initial score
        def get_score(counts):
            # We care about x=1..N
            # Index 0 in counts corresponds to x=0, which is not in 1..L
            # But x=0 mod p is r=0.
            # The problem asks for integers 1..L.
            # So we check indices 1..N-1 (since counts is 0..N-1)
            # Wait, counts[r::p] updates indices r, r+p, ...
            # If r=0, it updates 0, p, 2p...
            # We need x=1..L.
            # So we should look at counts[1:N]
            failures = np.nonzero(counts[1:N] < 2)[0]
            if len(failures) == 0:
                return N
            return failures[0] + 1
        
        current_score = get_score(counts)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        # Annealing loop
        while time.time() - start_time < max_time:
            # Pick a random prime
            i = np.random.randint(n_primes)
            p = primes[i]
            r_old = residues[i]
            
            # Pick a new random residue
            r_new = np.random.randint(0, p)
            if r_old == r_new:
                continue
                
            # Update counts
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            
            new_score = get_score(counts)
            delta = new_score - current_score
            
            # Acceptance probability
            if delta > 0 or np.random.rand() < math.exp(delta / T):
                current_score = new_score
                residues[i] = r_new
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
                
    return best_residues.tolist()
