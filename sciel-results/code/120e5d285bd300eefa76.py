"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return below_two[0]

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    
    # Deadline guard: 150s wall-clock limit, use 145s for safety
    deadline = time.time() + 145
    
    # Start with the seed solution
    best_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_score = _score(best_residues, primes)
    
    # Local search: greedy hill-climbing
    improved = True
    round_num = 0
    max_rounds = 200
    
    while improved and round_num < max_rounds and time.time() < deadline:
        improved = False
        round_num += 1
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            p = primes[i]
            current_r = best_residues[i]
            
            # Try strategic candidates: neighbors and a few randoms
            candidates = set()
            for delta in range(-3, 4):
                new_r = (current_r + delta) % p
                candidates.add(new_r)
            # Add some random candidates for larger primes
            if p > 10:
                for _ in range(5):
                    candidates.add(np.random.randint(0, p))
            
            for new_r in candidates:
                if new_r == current_r:
                    continue
                
                test_residues = list(best_residues)
                test_residues[i] = new_r
                
                s = _score(test_residues, primes)
                
                if s > best_score:
                    best_residues = test_residues
                    best_score = s
                    improved = True
                    break  # Move to next i
    
    return best_residues
# EVOLVE-BLOCK-END
