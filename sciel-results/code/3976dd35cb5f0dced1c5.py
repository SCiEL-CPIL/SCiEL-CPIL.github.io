"""Pairwise joint search solver for the covering problem."""
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

def _count_covered(residues, primes, max_L):
    """
    Count how many integers in 1..max_L are covered at least twice.
    Returns the largest L such that every integer 1..L is covered at least twice.
    """
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        pos = np.arange(start, max_L + 1, p)
        if len(pos) > 0:
            counts[pos] += 1
    
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def _compute_counts(residues, primes, max_L):
    """Compute coverage counts array for given residues."""
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

def _find_L(counts, max_L):
    """Find largest L such that all positions 1..L have count >= 2."""
    below_two = np.where(counts[1:max_L+1] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000
    
    # Initial solution: targeted initialization focusing on coverage
    # Use a greedy approach to initialize
    initial_residues = [0] * n_primes
    for i, p in enumerate(primes):
        # Try to pick residue that covers the most currently deficient positions
        best_r = 0
        best_score = -1
        # For initialization, just use a simple heuristic
        best_r = (i + 1) % p
        initial_residues[i] = best_r
    
    best_residues = initial_residues[:]
    counts = _compute_counts(best_residues, primes, max_L)
    best_L = _find_L(counts, max_L)
    
    # Phase 1: Single-prime hill climbing to get a good starting point
    improved = True
    while improved and (time.time() - start_time) < deadline - 30:
        improved = False
        for i in range(n_primes):
            if (time.time() - start_time) > deadline - 30:
                break
            p = primes[i]
            current_r = best_residues[i]
            
            # Get positions for current residue
            if current_r == 0:
                old_start = p
            else:
                old_start = current_r
            if old_start <= max_L:
                old_pos = np.arange(old_start, max_L + 1, p)
            else:
                old_pos = np.array([], dtype=np.int32)
            
            # Generate targeted candidates: focus on residues that cover deficient positions
            # Find positions with count == 1 (deficient but fixable)
            deficient_pos = np.where((counts >= 1) & (counts < 2))[0]
            
            # For each candidate residue, compute how many deficient positions it would cover
            candidates = []
            if p <= 300:
                candidates = list(range(p))
            else:
                # Sample residues, but prioritize those that cover deficient positions
                np.random.seed(42 + i)
                candidates = list(np.random.choice(p, size=min(p, 200), replace=False))
            
            if current_r in candidates:
                candidates.remove(current_r)
            
            best_new_r = current_r
            best_new_L = best_L
            
            old_mask = np.zeros(max_L + 1, dtype=np.int8)
            if len(old_pos) > 0:
                old_mask[old_pos] = 1
            
            for new_r in candidates:
                if (time.time() - start_time) > deadline - 30:
                    break
                
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r
                
                if new_start > max_L:
                    new_counts = counts - old_mask
                else:
                    new_pos = np.arange(new_start, max_L + 1, p)
                    new_mask = np.zeros(max_L + 1, dtype=np.int8)
                    if len(new_pos) > 0:
                        new_mask[new_pos] = 1
                    new_counts = counts - old_mask + new_mask
                
                new_L = _find_L(new_counts, max_L)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_new_r = new_r
            
            if best_new_r != current_r:
                best_residues[i] = best_new_r
                best_L = best_new_L
                improved = True
                
                if len(old_pos) > 0:
                    counts[old_pos] -= 1
                
                if best_new_r == 0:
                    new_start = p
                else:
                    new_start = best_new_r
                if new_start <= max_L:
                    new_pos = np.arange(new_start, max_L + 1, p)
                    if len(new_pos) > 0:
                        counts[new_pos] += 1
    
    # Phase 2: Pairwise joint search
    # For pairs of primes, jointly optimize their residues
    # Focus on pairs where at least one prime has small modulus (more candidates)
    
    # Precompute pair priority: pairs that jointly cover many deficient positions
    deficient_pos = np.where((counts >= 1) & (counts < 2))[0]
    
    # Create a list of prime indices sorted by prime value (smaller primes first)
    prime_indices = list(range(n_primes))
    
    # For each pair, estimate potential improvement
    pair_scores = []
    for i in range(n_primes):
        for j in range(i + 1, n_primes):
            pi = primes[i]
            pj = primes[j]
            # Estimate: how many deficient positions are in the union of possible covers
            # This is expensive, so use a heuristic: prioritize pairs with smaller primes
            score = 1.0 / (pi * pj)
            pair_scores.append((score, i, j))
    
    # Sort pairs by priority
    pair_scores.sort(reverse=True)
    
    # Process top pairs
    max_pairs_to_try = 5000
    pairs_processed = 0
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        for score, i, j in pair_scores:
            if pairs_processed >= max_pairs_to_try:
                break
            if (time.time() - start_time) > deadline - 5:
                break
            pairs_processed += 1
            
            pi = primes[i]
            pj = primes[j]
            ri = best_residues[i]
            rj = best_residues[j]
            
            # Get old positions for both primes
            if ri == 0:
                old_start_i = pi
            else:
                old_start_i = ri
            if old_start_i <= max_L:
                old_pos_i = np.arange(old_start_i, max_L + 1, pi)
            else:
                old_pos_i = np.array([], dtype=np.int32)
            
            if rj == 0:
                old_start_j = pj
            else:
                old_start_j = rj
            if old_start_j <= max_L:
                old_pos_j = np.arange(old_start_j, max_L + 1, pj)
            else:
                old_pos_j = np.array([], dtype=np.int32)
            
            old_mask_i = np.zeros(max_L + 1, dtype=np.int8)
            if len(old_pos_i) > 0:
                old_mask_i[old_pos_i] = 1
            
            old_mask_j = np.zeros(max_L + 1, dtype=np.int8)
            if len(old_pos_j) > 0:
                old_mask_j[old_pos_j] = 1
            
            # Generate candidate residues for both primes
            # For small primes, try all; for large, sample
            if pi <= 100:
                cand_i = list(range(pi))
            else:
                np.random.seed(1000 + i)
                cand_i = list(np.random.choice(pi, size=min(pi, 50), replace=False))
            
            if pj <= 100:
                cand_j = list(range(pj))
            else:
                np.random.seed(2000 + j)
                cand_j = list(np.random.choice(pj, size=min(pj, 50), replace=False))
            
            # Remove current residues from candidates to avoid no-op
            if ri in cand_i:
                cand_i.remove(ri)
            if rj in cand_j:
                cand_j.remove(rj)
            
            # If either list is empty, skip
            if len(cand_i) == 0 or len(cand_j) == 0:
                continue
            
            best_new_ri = ri
            best_new_rj = rj
            best_new_L = best_L
            
            # Joint search: try combinations
            # To keep it tractable, limit the number of combinations
            max_combos = 1000
            combo_count = 0
            
            for new_ri in cand_i:
                if (time.time() - start_time) > deadline - 5:
                    break
                if new_ri == 0:
                    new_start_i = pi
                else:
                    new_start_i = new_ri
                
                if new_start_i > max_L:
                    new_mask_i = np.zeros(max_L + 1, dtype=np.int8)
                else:
                    new_pos_i = np.arange(new_start_i, max_L + 1, pi)
                    new_mask_i = np.zeros(max_L + 1, dtype=np.int8)
                    if len(new_pos_i) > 0:
                        new_mask_i[new_pos_i] = 1
                
                for new_rj in cand_j:
                    if combo_count >= max_combos:
                        break
                    if (time.time() - start_time) > deadline - 5:
                        break
                    combo_count += 1
                    
                    if new_rj == 0:
                        new_start_j = pj
                    else:
                        new_start_j = new_rj
                    
                    if new_start_j > max_L:
                        new_mask_j = np.zeros(max_L + 1, dtype=np.int8)
                    else:
                        new_pos_j = np.arange(new_start_j, max_L + 1, pj)
                        new_mask_j = np.zeros(max_L + 1, dtype=np.int8)
                        if len(new_pos_j) > 0:
                            new_mask_j[new_pos_j] = 1
                    
                    new_counts = counts - old_mask_i - old_mask_j + new_mask_i + new_mask_j
                    new_L = _find_L(new_counts, max_L)
                    
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_new_ri = new_ri
                        best_new_rj = new_rj
            
            if best_new_ri != ri or best_new_rj != rj:
                # Update the solution
                # Remove old coverage for both primes
                if len(old_pos_i) > 0:
                    counts[old_pos_i] -= 1
                if len(old_pos_j) > 0:
                    counts[old_pos_j] -= 1
                
                best_residues[i] = best_new_ri
                best_residues[j] = best_new_rj
                best_L = best_new_L
                improved = True
                
                # Add new coverage
                if best_new_ri == 0:
                    new_start_i = pi
                else:
                    new_start_i = best_new_ri
                if new_start_i <= max_L:
                    new_pos_i = np.arange(new_start_i, max_L + 1, pi)
                    if len(new_pos_i) > 0:
                        counts[new_pos_i] += 1
                
                if best_new_rj == 0:
                    new_start_j = pj
                else:
                    new_start_j = best_new_rj
                if new_start_j <= max_L:
                    new_pos_j = np.arange(new_start_j, max_L + 1, pj)
                    if len(new_pos_j) > 0:
                        counts[new_pos_j] += 1
                
                # Update deficient positions for next iteration
                deficient_pos = np.where((counts >= 1) & (counts < 2))[0]
    
    return best_residues
