import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def _vectorized_score(residues, primes, limit=2000):
    """
    Returns the largest L such that every integer 1..L satisfies at least two congruences.
    Uses vectorized NumPy operations for speed.
    """
    # Convert to numpy arrays for speed
    residues = np.array(residues)
    primes = np.array(primes)
    
    # Create a range of integers 1..limit
    x_vals = np.arange(1, limit + 1)
    
    # We want to count how many primes p satisfy x % p == r_p for each x.
    # This is a bit tricky to vectorize directly over primes because p varies.
    # However, we can iterate over primes and accumulate counts.
    # Since limit is small (2000) and number of primes is 168, this is fast.
    
    counts = np.zeros(limit + 1, dtype=np.int32)
    
    for i in range(len(primes)):
        p = primes[i]
        r = residues[i]
        # Indices in x_vals where x % p == r
        # x is in [1, limit].
        # We can generate the arithmetic progression: r, r+p, r+2p, ...
        # But we need to handle the case where r=0 (multiples of p) and r>0.
        # Also x starts at 1.
        
        # Find the first x >= 1 such that x % p == r
        # If r == 0, first is p (if p <= limit).
        # If r > 0, first is r (if r <= limit).
        
        # Actually, simpler:
        # mask = (x_vals % p) == r
        # counts[mask] += 1
        
        # Using modulo on the whole array is fast in numpy.
        mask = (x_vals % p) == r
        counts[mask] += 1
        
    # Find the first index where count < 2
    # We want the largest L such that counts[1..L] >= 2.
    # If counts[i] < 2, then L < i.
    # So L is the first i-1 where counts[i] < 2.
    # If all counts >= 2, L = limit.
    
    # Find indices where count < 2
    # Note: x_vals starts at 1, so index 0 in counts corresponds to x=1?
    # No, x_vals = np.arange(1, limit+1).
    # mask has same shape as x_vals.
    # counts has shape limit+1. counts[0] is unused?
    # Let's align indices.
    # x_vals[0] = 1.
    # mask[0] corresponds to x=1.
    # counts[1] should hold the count for x=1.
    # My loop does counts[mask] += 1.
    # mask is boolean array of length limit.
    # So counts[0] corresponds to x=1? No.
    # np.arange(1, limit+1) has indices 0..limit-1.
    # mask has indices 0..limit-1.
    # counts has length limit+1.
    # So counts[mask] adds to indices 0..limit-1.
    # This means counts[0] holds count for x=1.
    # counts[1] holds count for x=2.
    # ...
    # counts[limit-1] holds count for x=limit.
    # counts[limit] is unused (0).
    
    # We need to find the first index i (0-based) where counts[i] < 2.
    # If found at index i, then x = i+1 is the first gap.
    # So L = (i+1) - 1 = i.
    # If no such index, L = limit.
    
    bad_indices = np.where(counts < 2)[0]
    if len(bad_indices) == 0:
        return limit
    else:
        first_bad_idx = bad_indices[0]
        return first_bad_idx

def construct():
    start_time = time.time()
    deadline = start_time + 115 # 5s safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Use a larger limit for the internal search to push the score higher
    # The evaluator caps at 10^5, but we can't search that far efficiently.
    # Let's use 5000 as a good balance.
    search_limit = 5000
    
    best_residues = None
    best_score = -1
    
    # Initial Random Search
    # Try a few seeds to get a decent starting point
    for seed in [42, 123, 999, 2024, 555]:
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        score = _vectorized_score(residues, primes, search_limit)
        if score > best_score:
            best_score = score
            best_residues = residues[:]
            
    # Local Search
    current_residues = best_residues[:]
    current_score = best_score
    
    stagnant_count = 0
    stagnation_threshold = 200
    max_iter = 100000
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Find the first gap using the vectorized score logic, but we need the location.
        # _vectorized_score returns L. The gap is at L+1.
        # We can re-derive the gap location quickly or just trust the score.
        # To find the specific gap for targeted repair, we need the x value.
        
        # Let's create a helper to find the first gap x
        def _find_first_gap_idx(residues, primes, limit):
            residues_np = np.array(residues)
            primes_np = np.array(primes)
            x_vals = np.arange(1, limit + 1)
            counts = np.zeros(limit + 1, dtype=np.int32)
            for i in range(len(primes_np)):
                p = primes_np[i]
                r = residues_np[i]
                mask = (x_vals % p) == r
                counts[mask] += 1
            bad_indices = np.where(counts < 2)[0]
            if len(bad_indices) == 0:
                return limit + 1
            return bad_indices[0] + 1 # 1-based x
        
        gap_x = _find_first_gap_idx(current_residues, primes, search_limit)
        
        if gap_x > search_limit:
            # We've covered the search limit. Try to extend?
            # For now, just do random moves to see if we can improve further or just stop.
            # Since the score is capped by search_limit in our internal metric,
            # we should probably increase search_limit or accept the current best.
            # Let's try a few random moves to see if we break through.
            if i % 50 == 0:
                idx = random.randint(0, n_primes - 1)
                old_r = current_residues[idx]
                new_r = random.randint(0, primes[idx] - 1)
                current_residues[idx] = new_r
                new_score = _vectorized_score(current_residues, primes, search_limit)
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
                else:
                    current_residues[idx] = old_r
                    stagnant_count += 1
            continue
            
        # gap_x is the first integer with coverage < 2.
        # We need to change residues to cover gap_x.
        # Primes that CAN cover gap_x are those where current_r_p != gap_x % p.
        # If we set r_p = gap_x % p, gap_x gets +1 coverage.
        
        candidates = []
        for idx, p in enumerate(primes):
            target_r = gap_x % p
            if current_residues[idx] != target_r:
                candidates.append((idx, target_r))
                
        if not candidates:
            # Should not happen if gap_x is truly a gap (coverage < 2)
            # and we have 168 primes.
            break

        # Strategy:
        # 1. Try to fix the gap by changing one residue to target_r.
        # 2. If that fails (breaks other spots), try random moves.
        
        improved = False
        
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        # Try a few candidates
        num_candidates_to_try = 5
        
        for idx, target_r in candidates[:num_candidates_to_try]:
            old_r = current_residues[idx]
            current_residues[idx] = target_r
            new_score = _vectorized_score(current_residues, primes, search_limit)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
                break
            else:
                current_residues[idx] = old_r
        
        if not improved:
            # Try random perturbation
            idx = random.randint(0, n_primes - 1)
            old_r = current_residues[idx]
            new_r = random.randint(0, primes[idx] - 1)
            current_residues[idx] = new_r
            new_score = _vectorized_score(current_residues, primes, search_limit)
            
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
            else:
                # Accept bad move with low probability (Simulated Annealing style)
                if random.random() < 0.05:
                    current_score = new_score
                    stagnant_count += 1
                else:
                    current_residues[idx] = old_r
                    stagnant_count += 1
            
            # Restart from best if stagnant
            if stagnant_count > stagnation_threshold:
                stagnant_count = 0
                current_residues = best_residues[:]
                current_score = best_score

    # Final evaluation with a higher limit to get the true score
    # The evaluator uses exact count capped at 10^5.
    # Our internal search used search_limit=5000.
    # Let's check the best_residues against a larger range to see if it actually goes further.
    # But we can't search that far.
    # However, if the local search pushed it to 5000, it might go higher.
    # Let's just return the best found. The evaluator will score it accurately.
    
    return best_residues
