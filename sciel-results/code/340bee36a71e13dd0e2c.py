import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    W = 30000  # working window
    
    def compute_coverage(residues):
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r >= 1 else p
            if start <= W:
                cov[start:W+1:p] += 1
        return cov
    
    def get_max_L(cov):
        if cov[1] < 2:
            return 0
        bad = np.where(cov[1:W+1] < 2)[0]
        if len(bad) == 0:
            return W
        return bad[0]
    
    # Phase 1: Smart greedy initialization
    # Process primes in increasing order. For each prime, pick residue that
    # best covers positions where current coverage is 0 or 1.
    residues = np.zeros(n_primes, dtype=np.int64)
    cov = np.zeros(W + 1, dtype=np.int32)
    
    # Precompute: for each prime p, the positions it covers for each residue
    # We'll use a scoring function: sum of weights over positions covered
    # Weight: high if cov[pos] == 0, medium if cov[pos] == 1, low if cov[pos] >= 2
    
    for idx in range(n_primes):
        p = primes_arr[idx]
        best_r = 0
        best_score = -1e18
        
        # For this prime, evaluate all residues (p <= 1000, so at most 1000 options)
        # But we can be smart: focus on residues that hit positions 1..min(p, 500)
        # because those are the critical small positions
        
        # Quick scoring: try residues that cover positions 1 through min(p, 300)
        # r = pos % p for pos in 1..min(p, 300)
        candidates = set()
        max_target = min(p, 400)
        for target in range(1, max_target + 1):
            candidates.add(target % p)
        
        # Add a few random candidates
        for _ in range(20):
            candidates.add(random.randint(0, p - 1))
        
        for r in candidates:
            start = r if r >= 1 else p
            if start > W:
                score = 0
            else:
                # Score: how many positions with cov==0 get covered (weight 10)
                # how many with cov==1 get covered (weight 5)
                # how many with cov>=2 get covered (weight 0.1)
                score = 0.0
                for pos in range(start, W + 1, p):
                    if cov[pos] == 0:
                        score += 10.0
                    elif cov[pos] == 1:
                        score += 5.0
                    else:
                        score += 0.1
                # Bonus for covering very small positions
                if start <= 100:
                    score += 2.0
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = best_r
        p = primes_arr[idx]
        r = best_r
        start = r if r >= 1 else p
        if start <= W:
            cov[start:W+1:p] += 1
    
    current_L = get_max_L(cov)
    best_L = current_L
    best_residues = residues.copy()
    
    # Phase 2: Pairwise coordinate descent
    # At each step, find the first gap, then try pairs of primes to fix it.
    # A pair move: change residue of prime i to cover gap_pos, and adjust
    # prime j to compensate for any lost coverage.
    
    max_iters = 200000
    it = 0
    no_improve_count = 0
    
    while it < max_iters and time.time() < deadline - 5:
        if current_L >= W:
            break
        
        gap_pos = current_L + 1
        if gap_pos > W:
            break
        
        # Find primes that could cover gap_pos
        # Prime i covers gap_pos if residues[i] == gap_pos % primes[i]
        # We want to find a pair (i, j) where:
        #   - setting residues[i] = gap_pos % p_i covers the gap
        #   - adjusting residues[j] maintains or improves coverage elsewhere
        
        # Strategy: pick i such that changing its residue to cover gap_pos
        # causes minimal damage, and pick j to repair damage.
        
        # First, find candidate i's: primes where gap_pos % p_i != residues[i]
        # and where the damage from changing is small.
        
        candidates_i = []
        for i in range(n_primes):
            p_i = primes_arr[i]
            needed_r = gap_pos % p_i
            if needed_r == residues[i]:
                continue
            old_start = residues[i] if residues[i] >= 1 else p_i
            new_start = needed_r if needed_r >= 1 else p_i
            
            # Estimate damage: positions in [1, gap_pos] that lose coverage
            damage = 0
            if old_start <= W:
                for pos in range(old_start, gap_pos + 1, p_i):
                    if cov[pos] == 2:
                        damage += 1  # this position would drop to 1
            # Estimate gain: gap_pos gets covered
            gain = 1  # gap_pos now has +1 coverage
            
            candidates_i.append((damage, gain, i, needed_r, old_start, new_start))
        
        if not candidates_i:
            # No single change can help; try random perturbation
            i = random.randint(0, n_primes - 1)
            p = primes_arr[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                it += 1
                continue
            old_start = old_r if old_r >= 1 else p
            new_start = new_r if new_r >= 1 else p
            if old_start <= W:
                cov[old_start:W+1:p] -= 1
            if new_start <= W:
                cov[new_start:W+1:p] += 1
            new_L = get_max_L(cov)
            if new_L >= current_L or random.random() < 0.15:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                if new_start <= W:
                    cov[new_start:W+1:p] -= 1
                if old_start <= W:
                    cov[old_start:W+1:p] += 1
                no_improve_count += 1
            it += 1
            continue
        
        # Sort by damage (ascending) and try the best ones
        candidates_i.sort(key=lambda x: x[0])
        
        applied = False
        for damage, gain, i, needed_r, old_start, new_start in candidates_i[:10]:
            if time.time() > deadline - 3:
                break
            
            p_i = primes_arr[i]
            
            # Apply change to i
            if old_start <= W:
                cov[old_start:W+1:p_i] -= 1
            if new_start <= W:
                cov[new_start:W+1:p_i] += 1
            residues[i] = needed_r
            
            new_L = get_max_L(cov)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                    no_improve_count = 0
                applied = True
                it += 1
                break
            else:
                # Revert i
                if new_start <= W:
                    cov[new_start:W+1:p_i] -= 1
                if old_start <= W:
                    cov[old_start:W+1:p_i] += 1
                residues[i] = residues[i]  # already set, revert
                # Actually we need to revert properly
                residues[i] = old_r if 'old_r' in dir() else residues[i]
                # Let me redo this more carefully
                pass
        
        # If single change didn't help, try a two-prime move
        if not applied:
            # Re-apply the best single change and look for a compensating j
            if candidates_i:
                damage, gain, i, needed_r, old_start, new_start = candidates_i[0]
                p_i = primes_arr[i]
                old_r_i = residues[i]
                
                # Apply change to i
                if old_start <= W:
                    cov[old_start:W+1:p_i] -= 1
                if new_start <= W:
                    cov[new_start:W+1:p_i] += 1
                residues[i] = needed_r
                
                # Find positions that now have cov == 1 (were 2, lost one)
                # Try to fix them by adjusting another prime j
                damaged_positions = []
                if old_start <= W:
                    for pos in range(old_start, W + 1, p_i):
                        if cov[pos] == 1:
                            damaged_positions.append(pos)
                
                if damaged_positions:
                    # Pick a damaged position and try to fix it with prime j
                    # Try a few random damaged positions
                    for dpos in damaged_positions[:5]:
                        for j in random.sample(range(n_primes), min(n_primes, 20)):
                            if j == i:
                                continue
                            p_j = primes_arr[j]
                            old_r_j = residues[j]
                            new_r_j = dpos % p_j
                            if new_r_j == old_r_j:
                                continue
                            
                            old_start_j = old_r_j if old_r_j >= 1 else p_j
                            new_start_j = new_r_j if new_r_j >= 1 else p_j
                            
                            # Check if this helps: does it cover dpos?
                            # And doesn't cause too much damage
                            if new_start_j <= W and dpos >= new_start_j and (dpos - new_start_j) % p_j == 0:
                                # Apply j's change
                                if old_start_j <= W:
                                    cov[old_start_j:W+1:p_j] -= 1
                                if new_start_j <= W:
                                    cov[new_start_j:W+1:p_j] += 1
                                residues[j] = new_r_j
                                
                                new_L = get_max_L(cov)
                                
                                if new_L > current_L:
                                    current_L = new_L
                                    if current_L > best_L:
                                        best_L = current_L
                                        best_residues = residues.copy()
                                        no_improve_count = 0
                                    applied = True
                                    break
                                else:
                                    # Revert j
                                    if new_start_j <= W:
                                        cov[new_start_j:W+1:p_j] -= 1
                                    if old_start_j <= W:
                                        cov[old_start_j:W+1:p_j] += 1
                                    residues[j] = old_r_j
                        
                        if applied:
                            break
                
                if not applied:
                    # Revert i
                    if new_start <= W:
                        cov[new_start:W+1:p_i] -= 1
                    if old_start <= W:
                        cov[old_start:W+1:p_i] += 1
                    residues[i] = old_r_i
                    no_improve_count += 1
                else:
                    it += 1
                    continue
        
        if not applied:
            # Random perturbation to escape local optimum
            i = random.randint(0, n_primes - 1)
            p = primes_arr[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                it += 1
                continue
            old_start = old_r if old_r >= 1 else p
            new_start = new_r if new_r >= 1 else p
            if old_start <= W:
                cov[old_start:W+1:p] -= 1
            if new_start <= W:
                cov[new_start:W+1:p] += 1
            new_L = get_max_L(cov)
            if new_L >= current_L or random.random() < 0.1:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                if new_start <= W:
                    cov[new_start:W+1:p] -= 1
                if old_start <= W:
                    cov[old_start:W+1:p] += 1
                no_improve_count += 1
        
        it += 1
        
        # Periodic full recomputation to fix any drift
        if it % 5000 == 0:
            cov = compute_coverage(residues)
            current_L = get_max_L(cov)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        
        # If stuck for too long, do a bigger random restart of some residues
        if no_improve_count > 5000:
            # Reset 20% of residues to random values
            for _ in range(n_primes // 5):
                i = random.randint(0, n_primes - 1)
                p = primes_arr[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r != old_r:
                    old_start = old_r if old_r >= 1 else p
                    new_start = new_r if new_r >= 1 else p
                    if old_start <= W:
                        cov[old_start:W+1:p] -= 1
                    if new_start <= W:
                        cov[new_start:W+1:p] += 1
                    residues[i] = new_r
            cov = compute_coverage(residues)
            current_L = get_max_L(cov)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
            no_improve_count = 0
    
    # Phase 3: Final targeted polish
    if time.time() < deadline - 2:
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            if current_L >= W:
                break
            target = current_L + 1
            for i in range(n_primes):
                if time.time() > deadline - 1:
                    break
                p = primes_arr[i]
                new_r = target % p
                if new_r == residues[i]:
                    continue
                old_r = residues[i]
                old_start = old_r if old_r >= 1 else p
                new_start = new_r if new_r >= 1 else p
                if old_start <= W:
                    cov[old_start:W+1:p] -= 1
                if new_start <= W:
                    cov[new_start:W+1:p] += 1
                new_L = get_max_L(cov)
                if new_L > current_L:
                    residues[i] = new_r
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                else:
                    if new_start <= W:
                        cov[new_start:W+1:p] -= 1
                    if old_start <= W:
                        cov[old_start:W+1:p] += 1
                    residues[i] = old_r
    
    return [int(r) for r in best_residues]
