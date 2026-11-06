import random
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    W = 20000  # window for coverage tracking
    
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
    
    def greedy_set_cover_pass(residues_init, cov_init, L_init):
        """
        Greedy set-cover: process primes in increasing order.
        For each prime, try ALL residues and pick the one that maximizes
        the number of positions in [1, L_init] that go from cov==1 to cov==2.
        Also penalize breaking positions that were cov==2.
        """
        residues = residues_init.copy()
        cov = cov_init.copy()
        current_L = L_init
        
        for i in range(n_primes):
            if time.time() > deadline:
                break
            p = primes_arr[i]
            old_r = residues[i]
            
            best_r = old_r
            best_score = -1e18
            
            # Try all residues for this prime
            for r in range(p):
                start = r if r >= 1 else p
                if start > W:
                    # This residue doesn't cover anything in [1, W]
                    # Score: just check if it's worse than current
                    # No new coverage, so score is 0 (unless old also had no coverage)
                    score = 0
                    if score > best_score:
                        best_score = score
                        best_r = r
                    continue
                
                # Compute the coverage positions for this residue
                positions = np.arange(start, W + 1, p)
                
                # Count how many positions in [1, current_L] have cov == 1 (will become 2)
                # and how many have cov == 2 (will become 3, which is fine)
                # and how many have cov == 0 (will become 1, not helpful yet)
                
                # For positions in [1, current_L]:
                mask_in_L = positions <= current_L
                if np.any(mask_in_L):
                    pos_in_L = positions[mask_in_L]
                    cov_at_pos = cov[pos_in_L]
                    # Gain: positions that are exactly 1 (become 2)
                    gain = np.sum(cov_at_pos == 1)
                    # Neutral: positions that are >= 2 (stay >= 2)
                    neutral = np.sum(cov_at_pos >= 2)
                    # Loss: positions that are 0 (become 1, not immediately helpful but not harmful)
                    loss = np.sum(cov_at_pos == 0)
                else:
                    gain = 0
                    neutral = 0
                    loss = 0
                
                # For positions in [current_L+1, W]: we want to build up coverage there too
                mask_beyond = positions > current_L
                if np.any(mask_beyond):
                    pos_beyond = positions[mask_beyond]
                    cov_at_pos_b = cov[pos_beyond]
                    # Building coverage beyond L is slightly positive
                    beyond_gain = np.sum(cov_at_pos_b == 0) * 0.1 + np.sum(cov_at_pos_b == 1) * 0.5
                else:
                    beyond_gain = 0
                
                # Total score: heavily weight fixing single-covered positions in prefix
                score = gain * 100 + beyond_gain - loss * 1
                
                if score > best_score:
                    best_score = score
                    best_r = r
            
            # Apply the best residue
            new_r = best_r
            if new_r != old_r:
                old_start = old_r if old_r >= 1 else p
                new_start = new_r if new_r >= 1 else p
                if old_start <= W:
                    cov[old_start:W+1:p] -= 1
                if new_start <= W:
                    cov[new_start:W+1:p] += 1
                residues[i] = new_r
            
            # Update current_L
            current_L = get_max_L(cov)
        
        return residues, cov, current_L
    
    # Phase 1: Greedy set-cover initialization
    # Start with all residues = 0 (or 1) and run greedy pass
    residues_init = np.ones(n_primes, dtype=np.int64)
    cov_init = compute_coverage(residues_init)
    L_init = get_max_L(cov_init)
    
    residues, cov, current_L = greedy_set_cover_pass(residues_init, cov_init, L_init)
    best_L = current_L
    best_residues = residues.copy()
    
    # Phase 2: Multiple greedy passes with different initializations
    # Try several random initializations and keep the best
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        residues_rand = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
        cov_rand = compute_coverage(residues_rand)
        L_rand = get_max_L(cov_rand)
        
        residues_new, cov_new, L_new = greedy_set_cover_pass(residues_rand, cov_rand, L_rand)
        
        if L_new > best_L:
            best_L = L_new
            best_residues = residues_new.copy()
            residues = residues_new.copy()
            cov = cov_new.copy()
            current_L = L_new
    
    # Phase 3: Local search - targeted improvement
    # Try to extend the prefix by adjusting individual residues
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            if current_L >= W:
                break
            
            target = current_L + 1
            if target > W:
                break
            
            # For each prime, check if changing its residue to cover `target` helps
            for i in range(n_primes):
                if time.time() > deadline - 1:
                    break
                p = primes_arr[i]
                old_r = residues[i]
                new_r = target % p
                if new_r == old_r:
                    continue
                
                old_start = old_r if old_r >= 1 else p
                new_start = new_r if new_r >= 1 else p
                
                # Apply change
                if old_start <= W:
                    cov[old_start:W+1:p] -= 1
                if new_start <= W:
                    cov[new_start:W+1:p] += 1
                residues[i] = new_r
                
                new_L = get_max_L(cov)
                
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                else:
                    # Revert
                    if new_start <= W:
                        cov[new_start:W+1:p] -= 1
                    if old_start <= W:
                        cov[old_start:W+1:p] += 1
                    residues[i] = old_r
    
    # Phase 4: Second greedy pass on the best solution
    if time.time() < deadline - 1:
        residues_final, cov_final, L_final = greedy_set_cover_pass(
            best_residues, compute_coverage(best_residues), best_L
        )
        if L_final > best_L:
            best_L = L_final
            best_residues = residues_final.copy()
    
    return [int(r) for r in best_residues]
