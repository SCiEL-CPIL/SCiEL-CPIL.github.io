import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    num_primes = len(primes)
    
    # We check up to M=100000 as per the problem statement cap
    M = 100000
    
    # State
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    counts = np.zeros(M + 1, dtype=np.int64)
    
    # Initialize counts
    for i, p in enumerate(primes):
        r = residues[i]
        # x % p == r
        # Indices in 0..M (representing 1..M+1? No, let's use 1-based indexing for x)
        # If r == 0, x = p, 2p, ...
        # If r != 0, x = r, r+p, ...
        # numpy slice: start::step
        # We want x in [1, M] such that x % p == r.
        # If r == 0: start = p (since x>=1, x=p is first)
        # If r != 0: start = r (if r>=1)
        
        if r == 0:
            start = p
        else:
            start = r
            
        if start <= M:
            counts[start::p] += 1
            
    def get_L():
        # Find first x in 1..M where counts[x] < 2
        # If all are >= 2, L = M
        # counts[0] is unused (x=0). We care about 1..M.
        # We want the first index i in 1..M where counts[i] < 2.
        # mask = counts[1:M+1] < 2
        # if not any(mask): return M
        # else: return index of first True + 1? 
        # Let's use argmax on the boolean array.
        # argmax returns 0 if no True. 
        # So if no failure, L=M.
        # If failure at index k (0-based in counts[1:]), then x = k+1.
        # L = x - 1 = k.
        
        # Optimization: check if all >= 2 first? 
        # np.all is fast.
        if np.all(counts[1:M+1] >= 2):
            return M
        
        # Find first False in (counts >= 2)
        # Which is first True in (counts < 2)
        mask = counts[1:M+1] < 2
        first_fail_idx = np.argmax(mask)
        # first_fail_idx is 0-based index in the slice [1..M]
        # So the failing x is first_fail_idx + 1
        # L is x - 1 = first_fail_idx
        return int(first_fail_idx)
    
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing Parameters
    T = 50.0
    T_min = 1.0
    cooling_rate = 0.9995
    
    # Main Loop
    while time.time() < deadline:
        # Pick a random prime index
        i = random.randint(0, num_primes - 1)
        p = primes[i]
        r_old = residues[i]
        
        # Pick a new random residue
        r_new = random.randint(0, p - 1)
        if r_new == r_old:
            continue
            
        # Update counts
        # Remove coverage of r_old
        if r_old == 0:
            start_old = p
        else:
            start_old = r_old
            
        if start_old <= M:
            counts[start_old::p] -= 1
            
        # Add coverage of r_new
        if r_new == 0:
            start_new = p
        else:
            start_new = r_new
            
        if start_new <= M:
            counts[start_new::p] += 1
            
        # Calculate new L
        new_L = get_L()
        
        # Acceptance criterion
        if new_L >= current_L:
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = current_L - new_L
            # If delta is large, exp is tiny.
            # We can skip calculation if delta > T * 20 (approx)
            if delta < 50 * T: 
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_L = new_L
                    residues[i] = r_new
                else:
                    # Revert
                    if start_new <= M:
                        counts[start_new::p] -= 1
                    if start_old <= M:
                        counts[start_old::p] += 1
                    residues[i] = r_old
            else:
                # Revert
                if start_new <= M:
                    counts[start_new::p] -= 1
                if start_old <= M:
                    counts[start_old::p] += 1
                residues[i] = r_old
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    return [int(r) for r in best_residues]
