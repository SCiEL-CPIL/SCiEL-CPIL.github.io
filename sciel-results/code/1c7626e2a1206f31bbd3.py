import time
import random
import math
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def _find_first_gap(residues, primes, limit=100000):
    """
    Find the smallest x >= 1 such that coverage < 2.
    Returns x if found, else limit + 1.
    Uses vectorized NumPy operations for speed.
    """
    residues_np = np.array(residues, dtype=np.int64)
    primes_np = np.array(primes, dtype=np.int64)
    
    chunk_size = 10000
    for start in range(1, limit + 1, chunk_size):
        end = min(start + chunk_size, limit + 1)
        xs = np.arange(start, end, dtype=np.int64)
        
        # Calculate differences
        # xs shape: (chunk_size, 1)
        # residues shape: (1, P)
        diff = xs[:, None] - residues_np[None, :]
        
        # Modulo
        mods = diff % primes_np[None, :]
        
        # Check equality
        matches = (mods == 0)
        
        # Sum matches
        counts = np.sum(matches, axis=1)
        
        # Find gaps
        gaps = np.where(counts < 2)[0]
        if len(gaps) > 0:
            return start + int(gaps[0])
            
    return limit + 1

def _find_best_residue_for_gap(x, residues, primes, indices, current_score, eval_limit):
    # Kept for interface compatibility, but not used in the new SA strategy
    return residues[indices[0]], current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    best_residues = None
    best_score = -1
    
    # Try a few random restarts
    for seed in range(20):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        
        # Initial score
        gap = _find_first_gap(residues, primes, 100000)
        current_score = gap - 1
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues[:]
            
        # Simulated Annealing
        T = 100.0
        cooling_rate = 0.995
        min_T = 1.0
        
        while T > min_T and time.time() < deadline - 5:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
                
            residues[i] = new_r
            gap = _find_first_gap(residues, primes, 100000)
            new_score = gap - 1
            
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
            else:
                # Avoid division by zero or underflow
                if T > 0:
                    prob = math.exp(delta / T)
                else:
                    prob = 0.0
                    
                if random.random() < prob:
                    current_score = new_score
                else:
                    residues[i] = old_r
            
            T *= cooling_rate
            
            # If we reached the cap, break
            if current_score >= 100000:
                break

    return best_residues
