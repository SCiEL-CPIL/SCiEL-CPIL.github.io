import numpy as np
import time
import random

def construct():
    # Primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    num_primes = len(primes)
    
    # Target length to search for
    # The score is capped at 10^5, but we don't need to allocate that much.
    # L >= 1000 is the target. Let's aim for 5000 to be safe.
    N = 5000
    
    x = np.arange(1, N + 1)
    
    # Precompute x % p for each prime to speed up mask generation
    # mods[i] is an array of length N containing x % primes[i]
    mods = []
    for p in primes:
        mods.append(x % p)
        
    # Initialize residues to 0
    residues = np.zeros(num_primes, dtype=np.int32)
    
    # Initialize counts
    # counts[x] is the number of primes p such that x == r_p (mod p)
    # Note: x is 1-indexed in our array, but np.arange(1, N+1) makes index 0 correspond to x=1.
    # So counts[0] corresponds to x=1.
    counts = np.zeros(N, dtype=np.int32)
    
    # Build initial counts
    for i, p in enumerate(primes):
        r = residues[i]
        counts[mods[i] == r] += 1
        
    def get_L(counts_arr):
        # Find the first index where count < 2
        # If none, L = N
        # We care about x=1..N, which are indices 0..N-1
        bad = np.where(counts_arr < 2)[0]
        if len(bad) == 0:
            return N
        return bad[0] # Index of first bad x. If x=1 (idx 0) is bad, L=0. If x=2 (idx 1) is bad, L=1.
        
    current_L = get_L(counts)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s buffer
    
    # --- Phase 1: Greedy Repair ---
    # Iterate through x, fix deficiencies
    for idx in range(N):
        if time.time() > deadline:
            break
        x_val = idx + 1
        if counts[idx] < 2:
            # Find candidates: primes that do NOT currently cover x_val
            # Current residue r_p covers x if mods[i][idx] == residues[i]
            # We want mods[i][idx] != residues[i]
            
            best_score = -1
            best_i = -1
            best_new_r = -1
            best_mask_old = None
            best_mask_new = None
            
            # To save time, we might not check all 168 primes if we find a good one quickly?
            # But 168 is small.
            for i in range(num_primes):
                if mods[i][idx] == residues[i]:
                    continue # This prime already covers x, so changing it won't help cover x (unless we change to another value that also covers x? No, residue is unique mod p).
                
                old_r = residues[i]
                new_r = x_val % primes[i]
                
                # If new_r == old_r, skip (won't happen if mods[i][idx] != old_r, because new_r is defined by x_val)
                # Actually, if mods[i][idx] != old_r, then new_r != old_r is guaranteed?
                # x_val % p = new_r.
                # mods[i][idx] = x_val % p.
                # So new_r = mods[i][idx].
                # Condition was mods[i][idx] != residues[i] (old_r).
                # So new_r != old_r.
                
                mask_old = (mods[i] == old_r)
                mask_new = (mods[i] == new_r)
                
                # Calculate score
                # Waste: elements that were count==2 and become count==1
                waste = np.sum((counts == 2) & mask_old)
                # Gain: elements that were count<2 and become count>=2 (or just count increases)
                # We specifically want to help those < 2
                gain = np.sum((counts < 2) & mask_new)
                
                score = gain - waste
                
                if score > best_score:
                    best_score = score
                    best_i = i
                    best_new_r = new_r
                    best_mask_old = mask_old
                    best_mask_new = mask_new
            
            if best_i != -1:
                # Apply change
                counts -= best_mask_old
                counts += best_mask_new
                residues[best_i] = best_new_r
                current_L = get_L(counts)
                
    # --- Phase 2: Local Search (Simulated Annealing) ---
    T = 5.0
    min_T = 0.1
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick random prime
        i = random.randint(0, num_primes - 1)
        
        # Pick random new residue
        p = primes[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        mask_old = (mods[i] == old_r)
        mask_new = (mods[i] == new_r)
        
        # Tentative update
        counts -= mask_old
        counts += mask_new
        
        new_L = get_L(counts)
        
        # Acceptance criterion
        if new_L > current_L:
            current_L = new_L
            residues[i] = new_r
        else:
            # Calculate delta
            delta = new_L - current_L # negative or 0
            if delta < 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    residues[i] = new_r
                    # current_L stays same? No, new_L is the score.
                    # But if we accept a worse move, current_L should update to new_L?
                    # Yes, the state is now worse.
                    current_L = new_L
                else:
                    # Revert
                    counts += mask_old
                    counts -= mask_new
                    # residues[i] unchanged
            
        # Cool
        T = max(min_T, T * cooling)
        
    return residues.tolist()

# EVOLVE-BLOCK-END
