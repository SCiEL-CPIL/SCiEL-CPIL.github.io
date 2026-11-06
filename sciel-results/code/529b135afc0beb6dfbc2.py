"""Rewritten program: multi-start greedy hill-climbing + plateau-walking for residue selection."""
import time
import random
import numpy as np

# Precompute primes up to 1000
def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, v in enumerate(sieve) if v]

PRIMES = _primes_up_to(1000)
N_PRIMES = len(PRIMES)
MAX_L = 10**5

def _compute_L(residues):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    # For each prime p_i with residue r_i, the covered residues in [1, MAX_L] are:
    # r_i, r_i + p_i, r_i + 2*p_i, ... (where r_i >= 1, or if r_i == 0 then p_i, 2*p_i, ...)
    # Actually x = r_i (mod p_i) means x ≡ r_i (mod p_i).
    # For x in 1..MAX_L, x satisfies the congruence if x mod p_i == r_i.
    
    # Build coverage count array
    counts = np.zeros(MAX_L + 1, dtype=np.int16)
    for i, (p, r) in enumerate(zip(PRIMES, residues)):
        # Find all x in 1..MAX_L such that x ≡ r (mod p)
        # x = r + k*p for k = 0, 1, 2, ...
        # Need x >= 1 and x <= MAX_L
        if r == 0:
            # x = p, 2p, 3p, ...
            start = p
        else:
            # x = r, r+p, r+2p, ... but need x >= 1
            start = r
            if start < 1:
                start = r + p  # shouldn't happen since r >= 1 when r != 0
        if start > MAX_L:
            continue
        # Generate all values: start, start+p, start+2p, ... <= MAX_L
        # Use numpy for speed
        k_max = (MAX_L - start) // p
        if k_max >= 0:
            values = start + np.arange(k_max + 1, dtype=np.int64) * p
            counts[values] += 1
    
    # Find the largest L such that counts[1..L] >= 2
    # We want the first position where count < 2, then L is one less than that
    # But we need every integer 1..L to have count >= 2
    # So L is the largest value such that min(counts[1:L+1]) >= 2
    
    # Find first index i >= 1 where counts[i] < 2
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return MAX_L
    else:
        first_fail = below_two[0] + 1  # 1-indexed
        return first_fail - 1

def _compute_L_fast(residues, current_L):
    """Faster computation that can use current_L as a hint."""
    # For efficiency, we can check if we can extend beyond current_L
    # But for now, use the same method
    return _compute_L(residues)

def _objective(residues):
    """Return the score (largest L)."""
    return _compute_L(residues)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_residues = None
    best_score = 0
    
    # Initial solution from the seed
    init_residues = [((i + 2) // 2) % p for i, p in enumerate(PRIMES)]
    best_residues = init_residues[:]
    best_score = _objective(best_residues)
    
    random.seed(42)
    
    # Multi-start greedy hill-climbing with plateau walking
    num_starts = 20
    
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
        
        if start_idx == 0:
            current = best_residues[:]
        else:
            # Random initialization with some bias toward good values
            current = [(random.randint(0, p - 1)) for p in PRIMES]
        
        current_score = _objective(current)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = current[:]
        
        # Hill climbing with plateau walking
        no_improve_count = 0
        max_no_improve = 5000
        plateau_steps = 0
        max_plateau_steps = 200
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Pick a random prime to modify
            idx = random.randint(0, N_PRIMES - 1)
            p = PRIMES[idx]
            
            if p == 2:
                # Only one alternative
                new_val = 1 - current[idx]
            else:
                new_val = random.randint(0, p - 1)
                if new_val == current[idx]:
                    continue
            
            old_val = current[idx]
            current[idx] = new_val
            
            new_score = _objective(current)
            
            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
                plateau_steps = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = current[:]
            elif new_score == current_score:
                # Plateau: keep the move
                no_improve_count += 1
                plateau_steps += 1
                if plateau_steps > max_plateau_steps:
                    # Revert to break plateau
                    current[idx] = old_val
                    plateau_steps = 0
            else:
                # Reject
                current[idx] = old_val
                no_improve_count += 1
                plateau_steps = 0
        
        # After hill climbing, try a local search focused on the failure point
        if time.time() < deadline:
            # Get the current L
            L = _objective(current)
            # Focus on residues that affect coverage near L
            # Try perturbing residues of small primes more aggressively
            for _ in range(200):
                if time.time() > deadline:
                    break
                # Pick a prime, bias toward smaller primes
                idx = random.randint(0, min(N_PRIMES - 1, 50))
                p = PRIMES[idx]
                old_val = current[idx]
                if p > 2:
                    new_val = random.randint(0, p - 1)
                else:
                    new_val = 1 - old_val
                if new_val == old_val:
                    continue
                current[idx] = new_val
                new_score = _objective(current)
                if new_score >= _objective(current if new_score < L else current):
                    # This is getting complicated, let's simplify
                    pass
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current[:]
                elif new_score < L:
                    current[idx] = old_val
    
    return best_residues
