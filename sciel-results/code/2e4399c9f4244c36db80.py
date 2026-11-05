import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
        
    primes = get_primes(1000)
    n_primes = len(primes)
    
    MAX_L = 100000
    # Use a smaller window for fast greedy construction
    GREEDY_WINDOW = 2000
    
    # --- Phase 1: Greedy Construction ---
    # We build residues forward. For each prime, we pick the residue that best
    # covers the currently deficient positions in the early range.
    residues = [0] * n_primes
    counts_greedy = np.zeros(GREEDY_WINDOW + 1, dtype=np.int32)
    
    for idx in range(n_primes):
        p = primes[idx]
        
        # Find deficient positions in the early window
        # We care about positions 1..GREEDY_WINDOW
        # A position is deficient if counts_greedy[pos] < 2
        # We want to pick r such that many deficient positions x satisfy x % p == r
        
        # Optimization: Instead of iterating all positions, we can accumulate weights
        # weight[r] = sum of weights for positions x where x%p == r and counts[x] < 2
        # Weight can be higher for counts[x]==0 (more critical)
        
        best_r = 0
        max_score = -1
        
        # We only need to check residues that actually cover some deficient spot
        # But p can be up to 997. Iterating p is fine.
        
        # Vectorized approach for scoring residues:
        # Create an array of positions 1..GREEDY_WINDOW
        positions = np.arange(1, GREEDY_WINDOW + 1)
        
        # Filter to deficient positions only
        deficient_mask = counts_greedy[1:GREEDY_WINDOW + 1] < 2
        if not np.any(deficient_mask):
            # No deficiencies, pick residue 1 (or any) to be safe, or maximize coverage
            # Actually, if no deficiencies, any residue works for the prefix.
            # Let's pick r=1 to avoid r=0 edge cases if possible, or just 0.
            # But r=0 means x ≡ 0 mod p.
            # Let's just pick r=1.
            best_r = 1
        else:
            def_pos = positions[deficient_mask]
            
            # For each residue r in 0..p-1, count how many def_pos match
            # We can use bincount
            mods = def_pos % p
            # counts of each mod
            res_counts = np.bincount(mods, minlength=p)
            
            # Boost score for positions where count was 0 (more urgent)
            # Let's just use the count for now, it's a good heuristic.
            # If we want to weight by urgency:
            # urgency = 2 - counts_greedy[def_pos]  # 2 if count=0, 1 if count=1
            # We can't easily bincount with weights in one go without extra array.
            # Simple count is fast.
            
            best_r = int(np.argmax(res_counts))
            max_score = res_counts[best_r]
            
            # Tie-breaking: if multiple residues have same max score, pick one that covers more total positions?
            # Or just first one.
            
        residues[idx] = best_r
        
        # Update counts_greedy
        r = best_r
        r_eff = r if r > 0 else p
        if r_eff <= GREEDY_WINDOW:
            counts_greedy[r_eff::p] += 1
            
    # --- Phase 2: Simulated Annealing Refinement ---
    
    # Build full counts array for the greedy solution
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1
            
    def calc_L(counts, current_L):
        if current_L > 0:
            # Check if current prefix is still valid
            prefix = counts[1:current_L+1]
            if np.min(prefix) < 2:
                bad = np.where(prefix < 2)[0]
                return bad[0]
            # Else, try to extend
            L = current_L
            if L < MAX_L:
                rest = counts[L+1:]
                bad_idx = np.where(rest < 2)[0]
                if len(bad_idx) == 0:
                    return MAX_L
                else:
                    return L + bad_idx[0]
        else:
            bad_idx = np.where(counts[1:] < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            else:
                return bad_idx[0]
                
    current_L = calc_L(counts, 0)
    best_L = current_L
    best_residues = residues.copy()
    
    # SA Parameters
    # Start with a temperature that allows some exploration
    T = 5.0
    cooling = 0.99995
    
    # To speed up SA, we can pre-compute indices for each prime?
    # counts[start::p] is already quite fast in numpy.
    
    # Local search improvement: 
    # Instead of random restarts, we do hill climbing first to get close to optimum,
    # then SA to escape.
    
    # Hill Climbing Phase
    hc_deadline = start_time + 40.0 # Spend 40s on HC
    improved = True
    while improved and time.time() < hc_deadline:
        improved = False
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            
            # Try all residues for this prime?
            # p can be up to 997. 
            # Trying all is O(p * (MAX_L/p)) = O(MAX_L) per prime.
            # Total O(n_primes * MAX_L) = 168 * 100000 = 16.8M ops.
            # This is fast enough for one pass.
            
            best_r_local = old_r
            best_L_local = current_L
            
            # We need to efficiently evaluate L for each candidate r.
            # Removing old_r and adding new_r.
            
            # Pre-compute the "background" counts (without prime i)
            # This is expensive if done inside the loop.
            # Better: just remove old, try new, revert.
            
            start_old = old_r if old_r > 0 else p
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
            
            # Now counts represents all primes except i.
            
            # For each candidate r in 0..p-1:
            # Add r, calc L, remove r.
            # This is O(p * cost_add_remove + p * cost_calc_L)
            # cost_add_remove is O(MAX_L/p)
            # cost_calc_L is O(MAX_L) in worst case, but often less.
            
            # Optimization: 
            # We can find the "bottleneck" positions.
            # Let's just iterate candidates.
            
            for new_r in range(p):
                if new_r == old_r:
                    continue
                    
                start_new = new_r if new_r > 0 else p
                if start_new <= MAX_L:
                    counts[start_new::p] += 1
                    
                new_L = calc_L(counts, current_L)
                
                if new_L > best_L_local:
                    best_L_local = new_L
                    best_r_local = new_r
                    
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                    
            # Restore old_r
            if start_old <= MAX_L:
                counts[start_old::p] += 1
                
            if best_r_local != old_r:
                # Apply best
                residues[i] = best_r_local
                current_L = best_L_local
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                improved = True
            # else: no improvement for this prime in this pass
            
    # SA Phase
    # Use remaining time
    T = 1.0
    cooling = 0.9999
    
    # To make SA faster, we can limit the search to a subset of primes or use adaptive steps.
    # But standard SA is robust.
    
    while time.time() < deadline:
        # Pick random prime
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        
        # Propose new residue
        # If p is small, pick random. If p is large, maybe pick near old_r?
        # Random is fine.
        new_r = np.random.randint(p)
        
        if new_r == old_r:
            continue
            
        # Apply change
        start_old = old_r if old_r > 0 else p
        if start_old <= MAX_L:
            counts[start_old::p] -= 1
            
        start_new = new_r if new_r > 0 else p
        if start_new <= MAX_L:
            counts[start_new::p] += 1
            
        # Calculate new L
        new_L = calc_L(counts, current_L)
        
        # Acceptance
        if new_L >= current_L:
            # Accept
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Reject or Accept with probability
            delta = new_L - current_L
            # If delta is very large negative, exp is 0
            if delta > -100 * T: # Avoid underflow
                prob = np.exp(delta / T)
            else:
                prob = 0.0
                
            if np.random.random() < prob:
                residues[i] = new_r
                current_L = new_L
            else:
                # Revert
                residues[i] = old_r
                # Revert counts
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1
                # current_L remains unchanged
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_residues
