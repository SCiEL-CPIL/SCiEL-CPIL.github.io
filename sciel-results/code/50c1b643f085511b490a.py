import random
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 135.0  # 15s safety margin before 150s kill
    
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    W = 50000  # window for coverage tracking
    
    def compute_coverage(residues):
        """Compute coverage array for positions 1..W"""
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r >= 1 else p
            if start <= W:
                cov[start:W+1:p] += 1
        return cov
    
    def get_max_L(cov):
        """Largest L such that cov[1..L] all >= 2"""
        if W < 1:
            return 0
        if cov[1] < 2:
            return 0
        bad = np.where(cov[1:W+1] < 2)[0]
        if len(bad) == 0:
            return W
        return bad[0]
    
    def smart_init():
        """Initialize residues to maximize coverage of early positions"""
        residues = np.zeros(n_primes, dtype=np.int64)
        # For each prime, pick residue that covers position 1 if possible, else minimize loss
        for i in range(n_primes):
            p = primes_arr[i]
            # Try to cover position 1: need r such that 1 % p == r, i.e., r = 1 % p
            r1 = 1 % p
            # Also consider covering position 2, 3, etc.
            # For now, just use r=1 if it helps, else random
            residues[i] = r1
        return residues
    
    def repair_pass(residues, cov, current_L, max_iters=100):
        """
        Iterative repair: find first gap, try to fix it with scored single-residue changes.
        """
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if current_L >= W:
                break
            
            target = current_L + 1
            if target > W:
                break
            
            # Find primes that can cover target
            candidates = []
            for i in range(n_primes):
                p = primes_arr[i]
                needed_r = target % p
                if needed_r != residues[i]:
                    candidates.append((i, needed_r))
            
            if not candidates:
                # No single change can fix this gap; try random perturbation
                break
            
            # Score each candidate change
            best_score = -1e18
            best_change = None
            
            for i, new_r in candidates:
                p = primes_arr[i]
                old_r = residues[i]
                
                old_start = old_r if old_r >= 1 else p
                new_start = new_r if new_r >= 1 else p
                
                # Compute the effect of this change
                # Positions affected: those in [1, W] that are in the arithmetic progression
                # We need to count how the coverage changes at each position
                
                # Simplified scoring: 
                # Gain: positions in [1, current_L] that go from cov==1 to cov==2
                # Loss: positions in [1, current_L] that go from cov==2 to cov==1
                # Bonus: covering the target position
                # Penalty: breaking coverage beyond current_L
                
                score = 0
                score += 1000  # Base bonus for covering the target
                
                # Check if this change helps or hurts the prefix
                # We'll do a quick check by looking at a few key positions
                
                # For positions that the old residue covered but new doesn't:
                if old_start <= W:
                    old_positions = np.arange(old_start, W + 1, p)
                    mask_prefix = old_positions <= current_L
                    if np.any(mask_prefix):
                        pos_prefix = old_positions[mask_prefix]
                        # These positions will lose one coverage
                        # Count how many were exactly 2 (will become 1 - bad)
                        exactly_2 = np.sum(cov[pos_prefix] == 2)
                        score -= exactly_2 * 50
                
                # For positions that the new residue covers but old didn't:
                if new_start <= W:
                    new_positions = np.arange(new_start, W + 1, p)
                    mask_prefix = new_positions <= current_L
                    if np.any(mask_prefix):
                        pos_prefix = new_positions[mask_prefix]
                        # These positions will gain one coverage
                        # Count how many were exactly 1 (will become 2 - good)
                        exactly_1 = np.sum(cov[pos_prefix] == 1)
                        score += exactly_1 * 50
                
                # Also check if this helps build coverage beyond current_L
                if new_start <= W:
                    new_positions_beyond = np.arange(new_start, min(W, current_L + p * 10) + 1, p)
                    mask_beyond = (new_positions_beyond > current_L) & (new_positions_beyond <= W)
                    if np.any(mask_beyond):
                        pos_beyond = new_positions_beyond[mask_beyond]
                        beyond_0 = np.sum(cov[pos_beyond] == 0)
                        beyond_1 = np.sum(cov[pos_beyond] == 1)
                        score += beyond_0 * 5 + beyond_1 * 10
                
                if score > best_score:
                    best_score = score
                    best_change = (i, new_r)
            
            if best_change is None:
                break
            
            i, new_r = best_change
            p = primes_arr[i]
            old_r = residues[i]
            
            # Apply the change
            old_start = old_r if old_r >= 1 else p
            new_start = new_r if new_r >= 1 else p
            
            if old_start <= W:
                cov[old_start:W+1:p] -= 1
            if new_start <= W:
                cov[new_start:W+1:p] += 1
            residues[i] = new_r
            
            new_L = get_max_L(cov)
            
            if new_L > current_L:
                current_L = new_L
            else:
                # Revert
                if new_start <= W:
                    cov[new_start:W+1:p] -= 1
                if old_start <= W:
                    cov[old_start:W+1:p] += 1
                residues[i] = old_r
                # Try random perturbation
                if random.random() < 0.3:
                    rand_i = random.randint(0, n_primes - 1)
                    rand_p = primes_arr[rand_i]
                    rand_new_r = random.randint(0, rand_p - 1)
                    old_r2 = residues[rand_i]
                    if rand_new_r != old_r2:
                        old_start2 = old_r2 if old_r2 >= 1 else rand_p
                        new_start2 = rand_new_r if rand_new_r >= 1 else rand_p
                        if old_start2 <= W:
                            cov[old_start2:W+1:rand_p] -= 1
                        if new_start2 <= W:
                            cov[new_start2:W+1:rand_p] += 1
                        residues[rand_i] = rand_new_r
                        # Don't revert this random change; let it be
                        # Recalculate L
                        current_L = get_max_L(cov)
        
        return residues, cov, current_L
    
    # Phase 1: Smart initialization
    residues = smart_init()
    cov = compute_coverage(residues)
    current_L = get_max_L(cov)
    best_L = current_L
    best_residues = residues.copy()
    
    # Phase 2: Iterative repair
    residues, cov, current_L = repair_pass(residues, cov, current_L, max_iters=200)
    if current_L > best_L:
        best_L = current_L
        best_residues = residues.copy()
    
    # Phase 3: Multiple restarts with different initializations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            # All zeros
            residues_rand = np.zeros(n_primes, dtype=np.int64)
        elif restart % 3 == 1:
            # Random
            residues_rand = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
        else:
            # Cover position 2
            residues_rand = np.array([2 % p for p in primes], dtype=np.int64)
        
        cov_rand = compute_coverage(residues_rand)
        L_rand = get_max_L(cov_rand)
        
        residues_new, cov_new, L_new = repair_pass(residues_rand, cov_rand, L_rand, max_iters=100)
        
        if L_new > best_L:
            best_L = L_new
            best_residues = residues_new.copy()
            residues = residues_new.copy()
            cov = cov_new.copy()
            current_L = L_new
    
    # Phase 4: Final targeted repair on best solution
    if time.time() < deadline - 5:
        residues_final, cov_final, L_final = repair_pass(
            best_residues.copy(), compute_coverage(best_residues), best_L, max_iters=150
        )
        if L_final > best_L:
            best_L = L_final
            best_residues = residues_final.copy()
    
    # Phase 5: One more pass if time allows
    if time.time() < deadline - 2:
        residues_final2, cov_final2, L_final2 = repair_pass(
            best_residues.copy(), compute_coverage(best_residues), best_L, max_iters=100
        )
        if L_final2 > best_L:
            best_L = L_final2
            best_residues = residues_final2.copy()
    
    return [int(r) for r in best_residues]
