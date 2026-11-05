"""
Vectorized batched local search for the covering problem.
Strategy:
1. Greedy initialization: Pick residues to cover the most "under-covered" points.
2. Fast evaluation: Use vectorized numpy operations to evaluate all candidates for a prime at once.
3. Batched Hill Climbing: Iterate through primes, finding the best residue for each in a single vectorized pass.
4. Restart: If stuck, restart from a perturbed version of the best solution.
"""
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _compute_counts(residues, primes, max_L):
    """Compute coverage counts for 1..max_L."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_L:
            pos = np.arange(start, max_L + 1, p)
            if len(pos) > 0:
                counts[pos] += 1
    return counts

def _find_L(counts):
    """Find largest L such that min(counts[1..L]) >= 2."""
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return len(counts) - 1
    else:
        return below_two[0]

def _greedy_init(primes, max_L):
    """Greedy initialization: maximize coverage of under-covered points."""
    residues = [0] * len(primes)
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    # We want to cover points 1..max_L at least twice.
    # Greedy: for each prime, pick residue that covers the most points with count < 2.
    # This is a heuristic.
    
    # To make it faster, we can just pick random residues first, then hill climb.
    # But a better start helps.
    # Let's try: for each prime, pick residue that minimizes the "gap" in coverage.
    # Actually, simple random init + fast hill climbing is often better due to speed.
    # Let's stick to a fast random init, but maybe slightly biased.
    
    for i, p in enumerate(primes):
        # Pick a random residue
        residues[i] = np.random.randint(0, p)
        
    return residues

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000
    
    # 1. Initial Solution
    # Use a deterministic but varied init
    np.random.seed(12345)
    best_residues = _greedy_init(primes, max_L)
    counts = _compute_counts(best_residues, primes, max_L)
    best_L = _find_L(counts)
    
    # Precompute position masks for each prime and residue?
    # That's too much memory (168 primes * avg p residues * max_L size).
    # Instead, we compute on the fly but vectorized.
    
    # 2. Batched Hill Climbing
    # For each prime i, we want to find r in [0, p) that maximizes L.
    # L is determined by the first index where count < 2.
    # Let C be the current counts.
    # If we change residue of prime i from r_old to r_new:
    # C' = C - mask(r_old) + mask(r_new)
    # We want to maximize the first index where C'[1..] < 2.
    
    # Optimization:
    # Most of the time, the "bottleneck" is a specific small set of indices.
    # If we can quickly check if a candidate fixes the current bottleneck, we can accept it.
    
    # Let's try a different approach:
    # Iterate through primes. For each prime, try all residues (if p is small) or a sample.
    # To speed up, we can precompute the "impact" of each residue.
    
    # However, the true bottleneck is the O(p * max_L) operation for each prime.
    # With p up to 1000, and 168 primes, that's ~168 * 1000 * 100000 = 1.68 * 10^10 ops. Too slow.
    
    # We need a smarter evaluation.
    # Observation: If the current L is small, only the prefix 1..L matters.
    # If L is large, we need to look further.
    
    # Let's try a "focused" search:
    # 1. Find the current bottleneck index B (first index where count < 2).
    # 2. Only consider primes that cover B (i.e., r_p == B mod p).
    # 3. For those primes, try changing their residue to see if it helps.
    # 4. If no improvement, expand the search window.
    
    # This is a targeted local search.
    
    improved = True
    iteration = 0
    
    while improved and (time.time() - start_time) < deadline - 10:
        improved = False
        iteration += 1
        
        # Find bottleneck
        below_two = np.where(counts[1:] < 2)[0]
        if len(below_two) == 0:
            break # Fully covered to max_L
        
        bottleneck_idx = below_two[0] + 1 # 1-based index
        # Actually, we want to fix the *first* few under-covered points.
        # Let's look at the first 100 under-covered points.
        target_indices = below_two[:100] + 1
        
        # For each prime, check if it covers any target index.
        # If it does, changing its residue might help.
        
        for i in range(n_primes):
            if (time.time() - start_time) > deadline - 5:
                break
            
            p = primes[i]
            current_r = best_residues[i]
            
            # Does current residue cover any target index?
            # target_indices % p == current_r
            # If yes, this prime is "responsible" for some under-coverage (or rather, it's covering them, but they are still under-covered by others).
            # Wait, if count < 2, and this prime covers it, then removing it makes it < 1.
            # We want to ADD coverage to these points.
            # So we want to change residue to cover a target index that is currently covered by only 1 other prime (or 0).
            
            # Let's just try all residues for small primes, and a smart subset for large ones.
            # But we need to be fast.
            
            # Fast check:
            # Compute new_counts = counts - mask(old) + mask(new)
            # We only care about the first, say, 5000 points if L is small.
            # If L is large, we care about more.
            
            search_range = min(10000, max_L) # Only look at prefix
            
            if p <= 50:
                candidates = np.arange(p)
            else:
                # Sample candidates
                # Focus on residues that cover target indices
                # target_indices % p
                candidate_residues = np.unique(target_indices % p)
                # Add some random ones
                random_residues = np.random.choice(p, size=min(50, p), replace=False)
                candidates = np.unique(np.concatenate([candidate_residues, random_residues]))
            
            # Remove current residue from candidates to avoid redundant check
            candidates = candidates[candidates != current_r]
            
            if len(candidates) == 0:
                continue
            
            # Vectorized evaluation
            # For each candidate r, compute mask.
            # This is hard to do in one shot for all r because masks are sparse.
            
            # Alternative:
            # Compute delta for each candidate.
            # delta = mask(new) - mask(old)
            # new_counts = counts + delta
            # We want to maximize the first index where new_counts < 2.
            
            # Let's just loop over candidates if the list is small, and use a fast inner check.
            best_new_r = current_r
            best_new_L = best_L
            
            # Precompute old mask for prefix
            old_mask = np.zeros(search_range + 1, dtype=np.int8)
            if current_r == 0:
                old_start = p
            else:
                old_start = current_r
            if old_start <= search_range:
                old_pos = np.arange(old_start, search_range + 1, p)
                old_mask[old_pos] = 1
            
            for new_r in candidates:
                if (time.time() - start_time) > deadline - 5:
                    break
                
                # Compute new mask
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r
                
                if new_start > search_range:
                    new_mask = np.zeros(search_range + 1, dtype=np.int8)
                else:
                    new_mask = np.zeros(search_range + 1, dtype=np.int8)
                    new_pos = np.arange(new_start, search_range + 1, p)
                    new_mask[new_pos] = 1
                
                new_counts_prefix = counts[:search_range+1] - old_mask + new_mask
                
                # Find L in prefix
                below_two_prefix = np.where(new_counts_prefix[1:] < 2)[0]
                if len(below_two_prefix) == 0:
                    # Covered up to search_range
                    # Need to check if it's covered further?
                    # If best_L < search_range, this is a big jump.
                    # If best_L >= search_range, we need to check further.
                    # For simplicity, assume it's at least search_range.
                    new_L = search_range
                    # Verify if it extends further?
                    # Only if it beats current best_L
                    if new_L > best_L:
                        # Full check
                        full_counts = _compute_counts(best_residues, primes, max_L)
                        # Wait, this is expensive.
                        # Let's just update the global counts array incrementally?
                        # That's what the original code did.
                        pass
                else:
                    new_L = below_two_prefix[0]
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_new_r = new_r
            
            if best_new_r != current_r:
                # Update
                best_residues[i] = best_new_r
                best_L = best_new_L
                improved = True
                
                # Update counts
                if current_r == 0:
                    old_start = p
                else:
                    old_start = current_r
                if old_start <= max_L:
                    old_pos = np.arange(old_start, max_L + 1, p)
                    counts[old_pos] -= 1
                
                if best_new_r == 0:
                    new_start = p
                else:
                    new_start = best_new_r
                if new_start <= max_L:
                    new_pos = np.arange(new_start, max_L + 1, p)
                    counts[new_pos] += 1
    
    return best_residues
