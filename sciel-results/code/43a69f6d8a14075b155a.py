import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(2, n+1) if sieve[i]]

def _compute_L(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L has >= 2 covers."""
    # For each x in 1..max_check, count how many congruences it satisfies
    # x satisfies x ≡ r_p (mod p) iff x % p == r_p
    # Vectorized approach: build a coverage array
    L = max_check
    # We'll check in chunks for efficiency
    # For each prime, mark which positions it covers
    # But 168 primes × 100000 = 16.8M operations, doable with numpy
    
    # More efficient: for each x, count covers
    # Use the fact that x ≡ r (mod p) means x = r + kp
    # So for prime p with residue r, it covers r, r+p, r+2p, ...
    
    # Build coverage count array
    cover = np.zeros(max_check + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        if r == 0:
            positions = np.arange(p, max_check + 1, p)
        else:
            positions = np.arange(r, max_check + 1, p)
        cover[positions] += 1
    
    # Find the largest L where all 1..L have cover >= 2
    # Find first position where cover < 2
    bad = np.where(cover[1:] < 2)[0]
    if len(bad) == 0:
        return max_check
    return bad[0]

def _compute_L_fast(residues, primes, max_check=100000):
    """Faster version using direct marking."""
    cover = np.zeros(max_check + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        if r == 0:
            positions = np.arange(p, max_check + 1, p)
        else:
            positions = np.arange(r, max_check + 1, p)
        cover[positions] += 1
    
    bad = np.where(cover[1:] < 2)[0]
    if len(bad) == 0:
        return max_check
    return bad[0]

def _greedy_init(primes, max_check=5000):
    """Greedy: for each prime, pick residue that maximizes coverage of undercovered positions."""
    n = len(primes)
    residues = [0] * n
    cover = np.zeros(max_check + 1, dtype=np.int32)
    
    for i, p in enumerate(primes):
        best_r = 0
        best_score = -1
        for r in range(p):
            if r == 0:
                positions = np.arange(p, max_check + 1, p)
            else:
                positions = np.arange(r, max_check + 1, p)
            # Score: how many positions become newly covered (or more covered)
            # We want to help positions that are currently undercovered
            # Score = sum of (1 if cover[x] < 2 else 0) for x in positions
            # But also consider positions that go from 1 to 2
            score = np.sum(cover[positions] < 2)
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        # Update cover
        if best_r == 0:
            positions = np.arange(p, max_check + 1, p)
        else:
            positions = np.arange(best_r, max_check + 1, p)
        cover[positions] += 1
    
    return residues

def _hill_climb(residues, primes, max_check=100000, max_iter=500, deadline=None):
    """Local search: try changing each residue, accept if it improves L."""
    residues = list(residues)
    n = len(primes)
    current_L = _compute_L_fast(residues, primes, max_check)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Shuffle order for diversity
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            p = primes[i]
            old_r = residues[i]
            best_r = old_r
            best_L = current_L
            # Try all possible residues for this prime
            for r in range(p):
                if r == old_r:
                    continue
                residues[i] = r
                new_L = _compute_L_fast(residues, primes, max_check)
                if new_L > best_L:
                    best_L = new_L
                    best_r = r
            residues[i] = best_r
            if best_r != old_r:
                current_L = best_L
                improved = True
        if not improved:
            break
    
    return residues, current_L

def _simulated_annealing(residues, primes, max_check=100000, deadline=None, initial_temp=100.0, cooling=0.999):
    """SA to escape local minima."""
    residues = list(residues)
    n = len(primes)
    current_L = _compute_L_fast(residues, primes, max_check)
    best_L = current_L
    best_residues = list(residues)
    
    temp = initial_temp
    step = 0
    while temp > 0.01:
        if deadline and time.time() > deadline:
            break
        # Pick a random prime to change
        i = random.randint(0, n - 1)
        p = primes[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        residues[i] = new_r
        new_L = _compute_L_fast(residues, primes, max_check)
        
        delta = new_L - current_L
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = list(residues)
        else:
            residues[i] = old_r
        
        temp *= cooling
        step += 1
        # Periodic hill climb
        if step % 200 == 0 and deadline and time.time() < deadline - 5:
            residues, current_L = _hill_climb(residues, primes, max_check, max_iter=20, deadline=deadline)
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    primes = _primes_up_to(1000)
    n = len(primes)  # Should be 168
    
    # Phase 1: Greedy initialization
    residues = _greedy_init(primes, max_check=5000)
    current_L = _compute_L_fast(residues, primes, 100000)
    
    # Phase 2: Hill climbing
    if time.time() < deadline - 30:
        residues, current_L = _hill_climb(residues, primes, 100000, max_iter=100, deadline=deadline)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 10:
        residues, current_L = _simulated_annealing(residues, primes, 100000, deadline=deadline, initial_temp=50.0, cooling=0.9995)
    
    # Phase 4: Final hill climb polish
    if time.time() < deadline - 2:
        residues, current_L = _hill_climb(residues, primes, 100000, max_iter=50, deadline=deadline)
    
    return residues
# EVOLVE-BLOCK-END
