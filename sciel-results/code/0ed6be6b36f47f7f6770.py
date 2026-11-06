import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    # Generate primes up to 1000
    primes = []
    is_prime = [True] * 1001
    is_prime[0] = is_prime[1] = False
    for i in range(2, 1001):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 1001, i):
                is_prime[j] = False
    
    n_primes = len(primes) # 168
    MAX_L = 2000 # We aim to find L up to 2000
    
    # State
    residues = [0] * n_primes
    counts = [0] * (MAX_L + 1)
    
    def apply_residue(p_idx, r, delta):
        p = primes[p_idx]
        # x = r, r+p, r+2p ... < MAX_L
        # Note: r is in [0, p-1].
        # If r == 0, x can be 0? The problem says integers 1..L.
        # Congruence x = r (mod p).
        # If r=0, x is multiple of p.
        # We only care about x >= 1.
        start = r
        if start == 0:
            start = p # First positive multiple
        # Wait, if r=0, x=0 is not in 1..L. x=p, 2p...
        # If r>0, x=r, r+p...
        
        # Actually, simpler:
        # x in [1, MAX_L] such that x % p == r.
        # First x >= 1:
        # if r == 0: x = p (if p <= MAX_L)
        # if r > 0: x = r (if r <= MAX_L)
        
        x = r
        if x == 0:
            x = p
        while x <= MAX_L:
            counts[x] += delta
            x += p
            
    def get_L():
        # Find largest L such that counts[x] >= 2 for all 1 <= x <= L
        # If counts[1] < 2, L=0? 
        # Problem says "every integer 1..L".
        # If counts[1] < 2, L is 0? Or undefined?
        # Score is largest L. If 1 is not covered twice, L=0.
        for x in range(1, MAX_L + 1):
            if counts[x] < 2:
                return x - 1
        return MAX_L

    def score():
        return get_L()

    # Initialization Strategy
    # Try to cover small numbers well.
    # A good heuristic: assign residues to cover 1, 2, 3...
    # But we have 168 primes.
    
    best_residues = None
    best_L = 0
    
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Initialize with a few random restarts
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        residues = [random.randint(0, p-1) for p in primes]
        
        # Apply initial residues
        counts = [0] * (MAX_L + 1)
        for i, p in enumerate(primes):
            apply_residue(i, residues[i], 1)
            
        current_L = get_L()
        
        # Hill Climbing
        no_improve_count = 0
        max_no_improve = 500 # Restart if stuck
        
        while time.time() - start_time < time_limit:
            if no_improve_count > max_no_improve:
                break
                
            # Pick a random prime to change
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            
            # Pick a new residue
            # Sometimes pick a "good" one?
            # For now, random.
            new_r = random.randint(0, p-1)
            
            if new_r == old_r:
                continue
                
            # Apply change
            apply_residue(i, old_r, -1)
            apply_residue(i, new_r, 1)
            
            new_L = get_L()
            
            if new_L > current_L:
                current_L = new_L
                residues[i] = new_r
                no_improve_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
            else:
                # Revert
                apply_residue(i, new_r, -1)
                apply_residue(i, old_r, 1)
                no_improve_count += 1
                
    # If best_residues is None, return something valid (e.g., all 0)
    if best_residues is None:
        best_residues = [0] * n_primes
        # Calculate score for this just in case, though it's likely low
        counts = [0] * (MAX_L + 1)
        for i, p in enumerate(primes):
            apply_residue(i, best_residues[i], 1)
        best_L = get_L()

    return best_residues
# EVOLVE-BLOCK-END
