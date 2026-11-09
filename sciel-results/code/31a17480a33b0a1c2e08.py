import time
import random

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def _find_first_gap(residues, primes, limit=10000):
    """
    Find the smallest x >= 1 such that coverage < 2.
    Returns x if found, else limit + 1.
    """
    # residues is a list of ints
    # primes is a list of ints
    for x in range(1, limit + 1):
        count = 0
        for i, p in enumerate(primes):
            if x % p == residues[i]:
                count += 1
                if count >= 2:
                    break
        if count < 2:
            return x
    return limit + 1

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # We can use a larger limit for evaluation now because early exit is fast
    eval_limit = 50000 
    
    def evaluate(residues):
        gap = _find_first_gap(residues, primes, eval_limit)
        return gap - 1

    best_residues = None
    best_score = -1
    
    # Initialization with multiple random restarts
    # We do a quick hill climb for each start
    for seed in range(20):
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        
        # Simple Hill Climbing for this start
        current_residues = residues[:]
        current_score = evaluate(current_residues)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try to improve by changing one prime
            # Random order to avoid bias
            indices = list(range(n_primes))
            random.shuffle(indices)
            for i in indices:
                old_r = current_residues[i]
                # Try a few random changes, or all?
                # All is too slow. Let's try a few.
                # Or, just one random change per pass?
                # Let's try the "best of 10" random changes for this prime
                best_new_r = old_r
                best_new_score = current_score
                for _ in range(10):
                    new_r = random.randint(0, primes[i] - 1)
                    if new_r == old_r: continue
                    current_residues[i] = new_r
                    score = evaluate(current_residues)
                    if score > best_new_score:
                        best_new_score = score
                        best_new_r = new_r
                if best_new_r != old_r:
                    current_residues[i] = best_new_r
                    current_score = best_new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = current_residues[:]
                        # Update eval_limit if we are doing well?
                        # Keep it fixed for simplicity, 50000 is good.
        
        # After hill climbing, if we have time, maybe do some SA?
        # For now, just keep the best from these restarts.

    return best_residues
