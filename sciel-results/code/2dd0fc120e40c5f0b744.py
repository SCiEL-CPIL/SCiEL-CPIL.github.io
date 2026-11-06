import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
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
    
    # Working window. We want to maximize L.
    # If we find a solution with L > 1000, we are good.
    # Let's set W to 50000 to be safe, but updates are O(W/p).
    # Sum(1/p) ~ 2.2, so total updates per step is ~ 2.2 * W / W? No.
    # Each update touches W/p elements.
    # Total elements touched per full pass of primes is W * sum(1/p) ~ 2.2 W.
    # With W=50000, that's 110,000 operations per full coverage update.
    # But we only update one prime at a time.
    # So one move costs O(W/p_i). For small p, this is large.
    # p=2 -> 25000 ops. p=1000 -> 50 ops.
    # Average cost is manageable.
    
    W = 50000
    
    best_L_found = 0
    best_residues = None
    
    def compute_L(cov):
        """Find the largest L such that cov[1..L] >= 2"""
        # We care about 1..L
        # cov is 0-indexed, cov[0] is x=0.
        # We need cov[1] >= 2, cov[2] >= 2, ...
        mask = cov[1:] < 2
        if not mask.any():
            return W
        first_bad_idx = np.argmax(mask)
        # first_bad_idx is 0-based in cov[1:]
        # So bad position is 1 + first_bad_idx
        # Last good position is (1 + first_bad_idx) - 1 = first_bad_idx
        return first_bad_idx

    def run_local_search(residues, cov, current_L, max_steps=10000):
        local_best_L = current_L
        local_best_res = residues.copy()
        stuck_counter = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            if current_L >= W:
                break
                
            # Find first gap
            bad_pos = current_L + 1
            
            # Try to fix bad_pos
            # We want to change one r_i to bad_pos % p_i
            # Evaluate all i? 168 primes.
            # For each i, calculate damage.
            
            best_i = -1
            best_new_r = -1
            best_score = -float('inf')
            
            # To speed up, we can sample primes if it's too slow, but 168 is small.
            for i in range(n_primes):
                p = primes_arr[i]
                old_r = residues[i]
                new_r = bad_pos % p
                
                if new_r == old_r:
                    # Already covers bad_pos? 
                    # If cov[bad_pos] < 2, and this prime covers it, 
                    # then we need ANOTHER prime to cover it.
                    # Changing this prime to the same value does nothing.
                    continue
                
                # Calculate impact on [1, bad_pos-1]
                # We only care about positions < bad_pos because bad_pos is currently bad.
                # Actually, we care about maintaining the prefix [1, bad_pos-1].
                
                # Positions affected:
                # Old: old_r, old_r+p, ... < bad_pos
                # New: new_r, new_r+p, ... < bad_pos
                
                # We can compute the change in coverage for these positions.
                # But doing this for all 168 primes every step might be slow if bad_pos is large.
                # However, bad_pos is likely small in the beginning.
                
                # Heuristic: 
                # If p > bad_pos, only 1 position is affected in [1, bad_pos-1].
                # If p is small, many positions.
                
                # Let's approximate damage by just looking at the count of positions 
                # where coverage drops from 2 to 1.
                
                # Efficient check:
                # We can create a temporary delta array? No, too slow.
                # Let's just check the specific points.
                
                damage = 0
                # Iterate through positions covered by old_r in [1, bad_pos-1]
                # and check if they are NOT covered by new_r.
                
                # To avoid Python loops over ranges, we can use numpy if the range is large,
                # but for small ranges, Python loops are fine.
                
                # Let's try a simpler heuristic first:
                # Just pick the prime with the largest p (least interference)
                # or random.
                
                # Let's stick to the "minimize damage" but calculate it fast.
                # Damage is number of x in [1, bad_pos-1] such that:
                # x % p == old_r AND x % p != new_r AND cov[x] == 2.
                
                # If p > bad_pos:
                if p > bad_pos:
                    old_pos = old_r if old_r > 0 else -1
                    new_pos = new_r if new_r > 0 else -1
                    
                    # Check old_pos
                    if 0 < old_pos < bad_pos:
                        if cov[old_pos] == 2:
                            damage += 1
                    # Check new_pos (gain)
                    if 0 < new_pos < bad_pos:
                        if cov[new_pos] == 1:
                            damage -= 1 # Gain reduces damage effectively
                
                else:
                    # Small prime, skip detailed damage calc for speed?
                    # Or do it?
                    # Let's do it for correctness.
                    # Number of points is bad_pos / p.
                    # If bad_pos is 1000 and p=2, 500 points.
                    # 168 primes * 500 points = 84,000 checks.
                    # This might be too slow for 10,000 steps.
                    
                    # Optimization: Only do detailed check for a subset of primes?
                    # Or break early if damage is already high?
                    
                    # Let's assume damage is high for small primes and prefer large primes.
                    # Score = -damage + (1 if p > bad_pos else 0)
                    
                    # For now, let's just estimate damage as bad_pos / p * 0.5
                    damage = int(bad_pos / p)
                
                # Score: Prefer moves that fix bad_pos (all do) and minimize damage.
                # Also prefer moves that don't break existing coverage.
                score = -damage
                
                if score > best_score:
                    best_score = score
                    best_i = i
                    best_new_r = new_r
            
            if best_i == -1:
                # No valid move found? (Shouldn't happen unless all cover bad_pos already and L is wrong)
                # Or maybe we are stuck.
                stuck_counter += 1
                if stuck_counter > 10:
                    # Random kick
                    i = random.randint(0, n_primes - 1)
                    p = primes_arr[i]
                    new_r = random.randint(0, p-1)
                    old_r = residues[i]
                    if new_r != old_r:
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
                        stuck_counter = 0
                    continue
                else:
                    break
            
            # Apply best move
            i = best_i
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
                stuck_counter = 0
            else:
                stuck_counter += 1
                # If we didn't improve L, we might be stuck in a local minimum.
                # The "damage" heuristic might be too conservative.
                # If stuck for a while, do a random kick.
                if stuck_counter > 50:
                    # Random kick
                    kick_i = random.randint(0, n_primes - 1)
                    kick_p = primes_arr[kick_i]
                    kick_new_r = random.randint(0, kick_p-1)
                    kick_old_r = residues[kick_i]
                    if kick_new_r != kick_old_r:
                        ks_old = kick_old_r if kick_old_r > 0 else kick_p
                        ks_new = kick_new_r if kick_new_r > 0 else kick_p
                        if ks_old <= W:
                            cov[ks_old:W+1:kick_p] -= 1
                        if ks_new <= W:
                            cov[ks_new:W+1:kick_p] += 1
                        residues[kick_i] = kick_new_r
                        new_L = compute_L(cov)
                        if new_L > local_best_L:
                            local_best_L = new_L
                            local_best_res = residues.copy()
                        current_L = new_L
                        stuck_counter = 0
                    else:
                        continue
            
            current_L = new_L
        
        return local_best_L, local_best_res

    # Main Loop: Multiple Restarts
    restarts = 0
    while time.time() < deadline - 10:
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
        
        L, res = run_local_search(residues, cov, current_L)
        
        if L > best_L_found:
            best_L_found = L
            best_residues = res
            # print(f"New best: {L}", file=sys.stderr)
        
        restarts += 1
        
    if best_residues is None:
        # Fallback: all 1s
        return [1] * n_primes
        
    return [int(r) for r in best_residues]
