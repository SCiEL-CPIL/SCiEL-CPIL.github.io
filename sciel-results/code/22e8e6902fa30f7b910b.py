import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin
    
    # Generate primes <= 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We want to maximize L. The evaluator checks up to 10^5, but we can
    # realistically only push L to a few thousand with 168 primes.
    # Let's set a working window W. If we reach W, we can extend it, but 
    # for simplicity and speed, we'll work with a fixed large W.
    W = 20000 
    
    best_L_found = 0
    best_residues = None
    
    def compute_L(cov):
        """Find the largest L such that cov[1..L] >= 2"""
        if cov[0] < 2: # cov[0] corresponds to x=0, we care about x=1..
             # Actually cov is indexed 0..W. x=1 is cov[1].
             pass
        # We need cov[1] >= 2, cov[2] >= 2, ...
        # Find first index i >= 1 where cov[i] < 2
        # If cov[1] < 2, L=0.
        # If cov[1..k] >= 2 and cov[k+1] < 2, L=k.
        
        # Using numpy to find first False in (cov[1:] >= 2)
        mask = cov[1:] < 2
        if not mask.any():
            return W
        first_bad_idx = np.argmax(mask)
        # first_bad_idx is 0-based relative to cov[1:]
        # So the bad position is 1 + first_bad_idx
        # The last good position is (1 + first_bad_idx) - 1 = first_bad_idx
        return first_bad_idx

    def run_restart():
        # Random initialization
        residues = np.array([random.randint(0, p-1) for p in primes], dtype=np.int64)
        
        # Compute initial coverage
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r > 0 else p
            if start <= W:
                cov[start:W+1:p] += 1
        
        current_L = compute_L(cov)
        local_best_L = current_L
        local_best_res = residues.copy()
        
        # Greedy Improvement Phase
        # We try to fix the first gap.
        max_greedy_steps = 500
        for step in range(max_greedy_steps):
            if time.time() > deadline - 10:
                return local_best_L, local_best_res
            
            if current_L >= W:
                break
                
            # Find the first bad position
            bad_pos = current_L + 1
            
            # For each prime, calculate the impact of setting r_p = bad_pos % p
            # We want to minimize the number of positions in [1, bad_pos] that 
            # drop from coverage >= 2 to < 2.
            
            best_prime_idx = -1
            best_new_r = -1
            min_damage = float('inf')
            max_gain = -1
            
            # Optimization: Only consider primes where changing the residue 
            # actually covers bad_pos. 
            # Actually, ALL primes can cover bad_pos with r = bad_pos % p.
            # But changing a prime that doesn't cover bad_pos currently might be costly.
            
            # Let's sample a subset of primes if n_primes is large? 
            # 168 is small enough to check all.
            
            for i in range(n_primes):
                p = primes_arr[i]
                old_r = residues[i]
                new_r = bad_pos % p
                
                if new_r == old_r:
                    # This prime already covers bad_pos. 
                    # If cov[bad_pos] < 2, this prime is already doing its job.
                    # We need OTHER primes to cover it.
                    continue
                
                # Calculate damage: positions in [1, bad_pos] covered by old_r 
                # that will NOT be covered by new_r and have cov == 2.
                
                # Positions covered by old_r: old_r, old_r+p, ... <= bad_pos
                # Positions covered by new_r: new_r, new_r+p, ... <= bad_pos
                
                # We only care about positions <= bad_pos because we want to 
                # maintain the prefix property.
                
                # Efficient check:
                # old_positions = np.arange(old_r, bad_pos+1, p) if old_r > 0 else np.arange(p, bad_pos+1, p)
                # new_positions = np.arange(new_r, bad_pos+1, p) if new_r > 0 else np.arange(p, bad_pos+1, p)
                
                # Instead of creating arrays, we can check specific points.
                # But we need to count how many points in old_positions have cov == 2 
                # and are NOT in new_positions.
                
                # Since p is prime and we are looking at a small range relative to W,
                # the number of such points is small (bad_pos/p).
                
                old_start = old_r if old_r > 0 else p
                new_start = new_r if new_r > 0 else p
                
                # List of positions to check
                # We iterate through old positions
                damage = 0
                gain = 0
                
                # Check positions covered by old_r but not new_r
                # And positions covered by new_r but not old_r
                
                # Since we are only looking at [1, bad_pos], and p can be small,
                # this loop might be slow if done naively for every prime.
                # However, bad_pos is likely small initially.
                
                # Let's approximate damage by just checking if bad_pos is covered.
                # Actually, a simpler heuristic:
                # Damage = (cov[bad_pos] < 2) ? 0 : (cov[bad_pos] == 2 ? 1 : 0) 
                # This is not quite right.
                
                # Let's do a quick check for the most critical position: bad_pos.
                # If we change r_p, does bad_pos remain covered?
                # Yes, by definition new_r = bad_pos % p.
                # So bad_pos gains +1 coverage from this prime.
                
                # The risk is that we lose coverage at some other position j < bad_pos.
                # If cov[j] was 2, and we remove old_r, it becomes 1.
                
                # To speed this up, we can just pick a random few primes to try?
                # Or pick primes where p is large? Large p means fewer collisions in [1, bad_pos].
                
                # Let's just try all primes but use a fast damage estimate.
                # Damage estimate: 
                # If old_r == new_r, skip.
                # If old_r != new_r:
                #   The positions lost are {old_r + k*p <= bad_pos}
                #   The positions gained are {new_r + k*p <= bad_pos}
                
                # We can check the coverage of these specific points.
                
                # To avoid slow Python loops, let's limit the check to 
                # a few specific points or just accept the risk.
                
                # Heuristic: Prefer primes where p > bad_pos / 2.
                # If p > bad_pos/2, then there is at most 1 position in [1, bad_pos]
                # covered by any residue.
                # If p > bad_pos, there is 0 or 1 position.
                
                # Let's prioritize large primes.
                if p > bad_pos:
                    # Only one position is affected: old_r (if <= bad_pos) and new_r (if <= bad_pos)
                    # Actually, if p > bad_pos, then r is the only position in [1, bad_pos]
                    # Wait, if r=0, position is p. If p > bad_pos, no position in [1, bad_pos].
                    # If r > 0, position is r.
                    
                    old_pos = old_r if old_r > 0 else -1
                    new_pos = new_r if new_r > 0 else -1
                    
                    d = 0
                    g = 0
                    if old_pos > 0 and old_pos <= bad_pos:
                        if cov[old_pos] == 2:
                            d += 1
                    if new_pos > 0 and new_pos <= bad_pos:
                        if cov[new_pos] == 1:
                            g += 1
                    
                    # Score: minimize damage, maximize gain
                    score = d - g
                    if score < min_damage:
                        min_damage = score
                        best_prime_idx = i
                        best_new_r = new_r
                        
                else:
                    # Small prime, more complex.
                    # We'll skip these in the first pass to find easy wins.
                    pass
            
            if best_prime_idx == -1:
                # No easy win with large primes. Try all primes with a more careful check?
                # Or just break?
                # Let's try a random prime to escape.
                i = random.randint(0, n_primes - 1)
                p = primes_arr[i]
                new_r = bad_pos % p
                if new_r == residues[i]:
                    break
                best_prime_idx = i
                best_new_r = new_r
                min_damage = 0 # Just accept it
            
            # Apply change
            i = best_prime_idx
            p = primes_arr[i]
            old_r = residues[i]
            new_r = best_new_r
            
            old_start = old_r if old_r > 0 else p
            new_start = new_r if new_r > 0 else p
            
            if old_start <= W:
                cov[old_start:W+1:p] -= 1
            if new_start <= W:
                cov[new_start:W+1:p] += 1
            
            residues[i] = new_r
            
            new_L = compute_L(cov)
            
            if new_L > local_best_L:
                local_best_L = new_L
                local_best_res = residues.copy()
            
            current_L = new_L
            
            # If we made no progress, we might be stuck.
            # The greedy loop above always tries to fix bad_pos.
            # If it fails to improve L, it might oscillate.
            # We can add a counter for no-improvement steps.
            
        # Local Search Phase (Simple Hill Climbing)
        # Try to improve further by random moves that don't break the prefix.
        # This is risky and slow. Given the time budget, we might skip this 
        # and rely on more restarts.
        
        return local_best_L, local_best_res

    # Main Loop: Multiple Restarts
    restarts = 0
    while time.time() < deadline - 5:
        L, res = run_restart()
        if L > best_L_found:
            best_L_found = L
            best_residues = res
            # print(f"New best: {L}", file=sys.stderr)
        
        restarts += 1
        # If we are doing well, maybe do more restarts?
        # If we are stuck, try more.
        
    if best_residues is None:
        # Fallback: all 1s
        return [1] * n_primes
        
    return [int(r) for r in best_residues]
