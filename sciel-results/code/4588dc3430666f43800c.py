import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
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
    p_arr = np.array(primes, dtype=int)
    
    # We want to maximize L such that every integer 1..L is covered by at least 2 residues.
    # Let's try to push L as high as possible.
    # The current seed gets L=168.
    # A simple strategy is to assign residues such that small numbers are covered many times.
    # For example, if we set r_p = 1 for all p, then 1 is covered by all 168 primes.
    # But we need every integer up to L to be covered at least twice.
    
    # Let's use a local search approach but more aggressive.
    # We can also try to "stack" coverage on small numbers.
    
    # Initial guess: random residues
    residues = np.random.randint(0, p_arr)
    
    N = 2000  # We can look ahead a bit, but the score is capped at 10^5. Let's check up to 1000 or more.
    # The task says "L >= 1000 would realize the property...".
    # Let's set N to 10000 to be safe, but computing counts for 10000 * 168 is fast.
    # Actually, we only need to find the first failure.
    
    x = np.arange(1, N + 1)
    
    def compute_counts(res):
        counts = np.zeros(N, dtype=int)
        for i, p in enumerate(primes):
            r = res[i]
            counts += (x % p == r)
        return counts
    
    counts = compute_counts(residues)
    
    def get_L(c):
        # Find first index where c < 2
        # x is 1..N. Index 0 is x=1.
        # We want largest L such that c[0:L] >= 2.
        # So find first i where c[i] < 2. L = i.
        fails = np.where(c < 2)[0]
        if len(fails) == 0:
            return N
        return fails[0]
    
    current_L = get_L(counts)
    best_residues = residues.copy()
    best_L = current_L
    
    # Local search
    # Strategy: Find the first failing integer x_fail.
    # Try to fix it by changing a residue of a prime p that currently does NOT cover x_fail.
    # We want to increase coverage of x_fail.
    # Changing r_p to x_fail % p adds coverage to x_fail.
    # But it removes coverage from x_fail - p, x_fail - 2p, etc. (or x_fail + p, etc.)
    # Specifically, it removes coverage from all y such that y % p == old_r_p.
    # We must ensure that no y < x_fail has its count drop below 2.
    
    while time.time() < deadline:
        if current_L >= N:
            break
            
        # Find first failure
        fail_idx = current_L  # 0-indexed, so x = fail_idx + 1
        x_fail = fail_idx + 1
        
        # Find primes that do NOT cover x_fail
        # current_r_i = residues[i]
        # covers if x_fail % p_i == residues[i]
        # We want to pick one and change it.
        
        # To be efficient, we can pick a random prime that doesn't cover x_fail.
        # Or iterate through them.
        
        # Let's try all candidates for x_fail and pick the one that maximizes new L?
        # That might be expensive.
        # Let's just try random moves that fix x_fail.
        
        # Candidates: i such that x_fail % primes[i] != residues[i]
        # new_r = x_fail % primes[i]
        
        # We can precompute which primes cover x_fail?
        # x_fail % p == r_i
        
        # Let's just iterate through a random permutation of indices
        # and try to fix.
        
        improved = False
        order = np.random.permutation(num_primes)
        
        for i in order:
            if time.time() > deadline:
                break
                
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            if old_r == new_r:
                continue # Already covers it
            
            # Check if move is safe
            # We remove coverage from positions where x % p == old_r
            # We add coverage from positions where x % p == new_r
            
            # Danger: any x < x_fail where count[x] == 2 and x % p == old_r
            # If we remove that cover, count becomes 1.
            
            # Mask for removed coverage in prefix [1, x_fail-1]
            # x is 1..N. Indices 0..fail_idx-1 correspond to 1..x_fail-1.
            
            # We can check:
            # prefix_x = x[:fail_idx]
            # mask_removed = (prefix_x % p == old_r)
            # if np.any(mask_removed & (counts[:fail_idx] == 2)):
            #    continue
            
            # This check is O(N). If we do it for every i, it's O(N * 168).
            # N=2000, 168 primes. 336,000 ops per pass.
            # Passes might be many.
            
            # Optimization:
            # We can precompute for each p, the positions covered?
            # Or just do the check.
            
            # Let's try to optimize the check.
            # Instead of checking all x < x_fail, we can just check if the move breaks the prefix.
            
            # Actually, we can just apply the move, compute new L, and if it's worse, revert.
            # Computing new L is O(N).
            # Applying move is O(N) to update counts.
            # So O(N) per move.
            # 168 moves per pass. 168 * 2000 = 336,000 ops.
            # If we do 1000 passes, 336M ops. Might be slow in Python/Numpy.
            
            # Let's stick to the safety check which is also O(N) but allows us to skip bad moves early.
            # Actually, the safety check `np.any(mask_removed & (counts[:fail_idx] == 2))` is very fast with Numpy.
            
            if fail_idx > 0:
                prefix_x = x[:fail_idx]
                mask_removed = (prefix_x % p == old_r)
                if np.any(mask_removed & (counts[:fail_idx] == 2)):
                    continue
            
            # Apply move
            residues[i] = new_r
            
            # Update counts
            # counts -= (x % p == old_r)
            # counts += (x % p == new_r)
            
            # We can do this efficiently
            mask_old = (x % p == old_r)
            mask_new = (x % p == new_r)
            
            counts -= mask_old
            counts += mask_new
            
            new_L = get_L(counts)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                improved = True
            else:
                # Revert
                residues[i] = old_r
                counts += mask_old
                counts -= mask_new
                
        if not improved:
            # Stuck at local minimum.
            # Try a random restart or perturbation?
            # Or just break?
            # Let's try a few random restarts.
            # But we have time budget.
            
            # Perturbation: change a few random residues
            for _ in range(5):
                idx = np.random.randint(0, num_primes)
                old_r = residues[idx]
                new_r = np.random.randint(0, primes[idx])
                residues[idx] = new_r
                
                p = primes[idx]
                mask_old = (x % p == old_r)
                mask_new = (x % p == new_r)
                counts -= mask_old
                counts += mask_new
                
                new_L = get_L(counts)
                if new_L > current_L:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                    break
                else:
                    # Revert
                    residues[idx] = old_r
                    counts += mask_old
                    counts -= mask_new
                    
            if time.time() > deadline - 5:
                break
            
            # If still stuck, maybe restart completely?
            # Let's just continue the loop, the random order might help next time?
            # No, if we are stuck, we are stuck.
            # Let's try a full random restart if we have time.
            
            # Actually, let's just break out of the inner loop and continue outer.
            # The outer loop will pick a new random order.
            # But if the state is the same, it will fail again.
            # So perturbation is key.
            
            # If perturbation failed, we are in a deep local minimum.
            # Let's try a more aggressive perturbation: random restart of a subset?
            # Or just full random restart?
            
            # Let's check time.
            if time.time() > deadline - 10:
                break
            
            # Full random restart?
            # residues = np.random.randint(0, p_arr)
            # counts = compute_counts(residues)
            # current_L = get_L(counts)
            # if current_L > best_L:
            #    best_L = current_L
            #    best_residues = residues.copy()
            
            # Random restarts are slow to improve. Local search is better.
            # Let's just break and return best found.
            # Or keep trying with different random seeds?
            
            # Let's try one more time with a fresh random order.
            # If it fails again, we break.
            
            # To avoid infinite loop on stuck state:
            # We can count consecutive failures.
            # But for simplicity, let's just break.
            
            break
            
    return [int(r) for r in best_residues]
