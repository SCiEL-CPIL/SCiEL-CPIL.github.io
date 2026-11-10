"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _primes_up_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [int(p) for p in np.where(sieve)[0]]

def _build_count(residues, primes, L):
    """Build count array for positions 1..L."""
    counts = np.zeros(L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > L:
            continue
        # x = start, start+p, start+2p, ... <= L
        k_max = (L - start) // p
        if k_max >= 0:
            x_vals = start + np.arange(k_max + 1, dtype=np.int32) * p
            counts[x_vals] += 1
    return counts

def _find_prefix_L(counts):
    """Find largest L such that counts[1..L] all >= 2."""
    if len(counts) <= 1:
        return 0
    # Find first position where count < 2
    bad = np.where(counts[1:] < 2)[0]
    if len(bad) == 0:
        return len(counts) - 1
    return bad[0]

def _score(residues, primes, L_max=100000):
    """Find the largest L such that every integer 1..L has count >= 2."""
    L = 2000
    while L <= L_max:
        counts = _build_count(residues, primes, L)
        prefix = _find_prefix_L(counts)
        if prefix >= L:
            L = min(L * 2, L_max + 1)
        else:
            return prefix
    counts = _build_count(residues, primes, L_max)
    return _find_prefix_L(counts)

def _seed_residues(primes):
    """Create a reasonable initial assignment."""
    n = len(primes)
    residues = [0] * n
    # Heuristic: try to spread residues to cover early integers
    for i, p in enumerate(primes):
        # Try residue that hits a small integer
        # For prime p, residue r covers r, r+p, r+2p, ...
        # We want to cover integers 1..~200
        # Simple heuristic: r = (i+1) % p or similar
        residues[i] = (i + 1) % p
    return residues

def _greedy_repair(residues, primes, time_limit=100.0):
    """
    Structural change: gap-repair approach.
    Instead of optimizing score directly, find the first gap and try to fix it.
    """
    n = len(primes)
    start_time = time.time()
    residues = residues[:]
    
    # Work with a window that grows as we make progress
    L_window = 500
    
    while time.time() - start_time < time_limit - 5:
        counts = _build_count(residues, primes, L_window)
        prefix = _find_prefix_L(counts)
        
        if prefix >= L_window:
            # Extend window
            L_window = min(L_window * 2, 5000)
            continue
        
        # Find the gap position
        gap_pos = prefix + 1  # 1-indexed
        
        # Try to fix this gap by changing one prime's residue
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit - 5:
                break
            p = primes[i]
            old_r = residues[i]
            
            # What residue would make prime i cover gap_pos?
            target_r = gap_pos % p
            if target_r == 0:
                target_r = p  # but residue must be < p, so 0
                target_r = 0
            
            if target_r == old_r:
                # Already covers gap_pos, but maybe not enough other coverage
                # Try nearby residues
                candidates = [(target_r + d) % p for d in range(-3, 4) if d != 0]
            else:
                candidates = [target_r]
            
            for r in candidates:
                if r == old_r:
                    continue
                residues[i] = r
                new_counts = _build_count(residues, primes, L_window)
                new_prefix = _find_prefix_L(new_counts)
                if new_prefix > prefix:
                    improved = True
                    prefix = new_prefix
                    break
                else:
                    residues[i] = old_r
            
            if improved:
                break
        
        if not improved:
            # Try 2-flip: change two primes simultaneously
            two_flip_improved = False
            for i in range(n):
                if time.time() - start_time > time_limit - 5:
                    break
                if two_flip_improved:
                    break
                p_i = primes[i]
                old_r_i = residues[i]
                target_r_i = gap_pos % p_i
                if target_r_i == 0:
                    target_r_i = 0
                
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit - 5:
                        break
                    if two_flip_improved:
                        break
                    p_j = primes[j]
                    old_r_j = residues[j]
                    target_r_j = gap_pos % p_j
                    if target_r_j == 0:
                        target_r_j = 0
                    
                    # Try changing both to cover gap_pos
                    residues[i] = target_r_i
                    residues[j] = target_r_j
                    new_counts = _build_count(residues, primes, L_window)
                    new_prefix = _find_prefix_L(new_counts)
                    if new_prefix > prefix:
                        two_flip_improved = True
                        prefix = new_prefix
                    else:
                        residues[i] = old_r_i
                        residues[j] = old_r_j
            
            if not two_flip_improved:
                # Random perturbation to escape local optimum
                for _ in range(3):
                    idx = np.random.randint(0, n)
                    residues[idx] = np.random.randint(0, primes[idx])
                counts = _build_count(residues, primes, L_window)
                prefix = _find_prefix_L(counts)
    
    return residues

def construct():
    primes = _primes_up_to(1000)
    n = len(primes)
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with seeded residues
    residues = _seed_residues(primes)
    
    # Greedy repair
    residues = _greedy_repair(residues, primes, time_limit=120.0)
    
    # Check time and return
    if time.time() > deadline:
        return residues
    
    # Final local polish: try single flips to improve
    L_window = 2000
    counts = _build_count(residues, primes, L_window)
    prefix = _find_prefix_L(counts)
    
    while time.time() < deadline - 10:
        improved = False
        for i in range(n):
            if time.time() > deadline - 10:
                break
            p = primes[i]
            old_r = residues[i]
            # Try a few random residues
            for _ in range(3):
                r = np.random.randint(0, p)
                if r == old_r:
                    continue
                residues[i] = r
                new_counts = _build_count(residues, primes, L_window)
                new_prefix = _find_prefix_L(new_counts)
                if new_prefix > prefix:
                    improved = True
                    prefix = new_prefix
                    break
                else:
                    residues[i] = old_r
            if improved:
                break
        
        if not improved:
            # Try to extend window
            if prefix >= L_window and L_window < 5000:
                L_window = min(L_window * 2, 5000)
            else:
                break
    
    return residues
# EVOLVE-BLOCK-END
