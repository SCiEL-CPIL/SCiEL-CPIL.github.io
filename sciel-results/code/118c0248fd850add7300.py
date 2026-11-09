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

def _fast_score(residues, primes, limit=5000):
    """
    Vectorized score calculation.
    Returns the length L such that all 1..L have coverage >= 2.
    """
    # Create an array of integers 1 to limit
    # We need to check x % p == r_p for all p
    # Instead of iterating x, we can iterate primes and mark coverage
    
    # Optimization: 
    # cov[x] += 1 if x % p == r_p
    # We can do this by slicing.
    # For a given p and r, the indices in 1..limit that satisfy x % p == r are:
    # r, r+p, r+2p, ...
    # Note: if r == 0, it covers p, 2p, ... (since x starts at 1, x=p is the first if r=0? No, x % p == 0 means x is multiple of p. 
    # If r=0, indices are p, 2p... 
    # If r>0, indices are r, r+p... provided r>=1.
    
    # Let's use a numpy array for coverage counts.
    # limit can be up to 10000. 10000 is small enough for a numpy array.
    cov = np.zeros(limit + 1, dtype=np.int32)
    
    for i, p in enumerate(primes):
        r = residues[i]
        # Find start index
        # We want x in [1, limit] such that x % p == r.
        # If r == 0:
        #   x must be multiple of p. First is p (if p <= limit).
        #   start = p
        # Else:
        #   x must be congruent to r mod p.
        #   If r <= limit: start = r.
        #   Else: no solution in 1..limit? 
        #   Wait, r < p. If p <= limit, r is in 1..limit? 
        #   If r=0, handled above.
        #   If r>0, r is in [1, p-1]. Since p <= 1000 and limit >= 5000, r is always in [1, limit].
        
        if r == 0:
            start = p
        else:
            start = r
            
        if start <= limit:
            # cov[start::p] += 1
            # This is a slice update.
            cov[start::p] += 1
            
    # Find the first index where cov < 2
    # We care about 1..limit
    # cov[0] is unused/irrelevant for score 1..L
    
    # Find indices where cov < 2
    # We want the smallest index i >= 1 such that cov[i] < 2.
    # If no such index exists in 1..limit, score is limit.
    
    # Mask: cov < 2
    # We need to ignore index 0.
    # Let's create a mask of "bad" indices.
    # bad = np.where(cov < 2)[0]
    # Filter bad to be >= 1
    # If bad is empty or min(bad) > limit, score is limit.
    # Else score is min(bad) - 1.
    
    # Optimization: use np.argwhere or similar?
    # np.nonzero(cov < 2)
    bad_indices = np.nonzero(cov < 2)[0]
    
    # Filter out 0
    if len(bad_indices) == 0:
        return limit
        
    # Find first bad index >= 1
    # bad_indices are sorted.
    # If bad_indices[0] == 0, check next.
    # Since cov[0] is 0 (initially), and we never add to cov[0] (start >= 1 or p), cov[0] is always 0.
    # So bad_indices[0] is always 0.
    # We need the first one > 0.
    
    if len(bad_indices) > 1:
        first_bad = bad_indices[1]
    else:
        return limit
        
    if first_bad > limit:
        return limit
    else:
        return first_bad - 1

def construct():
    start_time = time.time()
    deadline = start_time + 115 # 5s safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Initialization
    best_residues = None
    best_score = -1
    
    # Use a larger limit for scoring to ensure we push the boundary
    # 5000 is a good balance. 10000 might be too slow for many iterations if not optimized well, 
    # but vectorized it should be fast. Let's try 5000 first.
    search_limit = 5000
    
    for seed in [42, 123, 999, 2024]:
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        score = _fast_score(residues, primes, search_limit)
        if score > best_score:
            best_score = score
            best_residues = residues
    
    # Local Search
    current_residues = best_residues[:]
    current_score = best_score
    
    stagnant_count = 0
    stagnation_threshold = 1000 # Can be higher with fast score
    
    # We can run many more iterations now
    max_iter = 100000
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Find first gap to guide the search
        # We can reuse the logic from _fast_score but we need the gap location.
        # Actually, _fast_score returns L. Gap is L+1.
        # But we need to know *which* primes cover the gap to fix it.
        # Let's just use random moves or guided moves based on the gap.
        
        gap = current_score + 1
        
        # If we hit the search limit, we might have a very good solution.
        # We should try to extend the limit or just keep searching.
        # If gap > search_limit, we can't easily identify the "next" gap to fix.
        # In that case, random moves might stumble into a better configuration that extends further.
        # Or we can temporarily increase search_limit for the score check.
        
        if gap > search_limit:
            # Try to push further
            # Increase search limit for this evaluation?
            # Or just do random moves.
            # Let's do a random move and check with a higher limit (e.g. 10000)
            # to see if we can extend.
            if random.random() < 0.5:
                # Random move
                idx = random.randint(0, n_primes - 1)
                old_r = current_residues[idx]
                new_r = random.randint(0, primes[idx] - 1)
                current_residues[idx] = new_r
                # Check with higher limit
                new_score = _fast_score(current_residues, primes, 10000)
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = current_residues[:]
                        stagnant_count = 0
                else:
                    current_residues[idx] = old_r
                    stagnant_count += 1
            else:
                # Try to fix the gap at search_limit+1?
                # We don't know the gap exactly if score == search_limit.
                # Let's assume the gap is search_limit+1 and try to cover it.
                target_gap = search_limit + 1
                candidates = []
                for idx, p in enumerate(primes):
                    target_r = target_gap % p
                    if current_residues[idx] != target_r:
                        candidates.append((idx, target_r))
                
                if candidates:
                    random.shuffle(candidates)
                    improved = False
                    for idx, target_r in candidates[:10]:
                        old_r = current_residues[idx]
                        current_residues[idx] = target_r
                        new_score = _fast_score(current_residues, primes, 10000)
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
                        stagnant_count += 1
            continue
            
        # Normal case: gap <= search_limit
        # Identify primes that can cover `gap`
        candidates = []
        for idx, p in enumerate(primes):
            target_r = gap % p
            if current_residues[idx] != target_r:
                candidates.append((idx, target_r))
        
        if not candidates:
            break
            
        # Try to fix the gap
        random.shuffle(candidates)
        improved = False
        
        # Try more candidates if stagnant
        num_to_try = 20 if stagnant_count > 500 else 10
        
        for idx, target_r in candidates[:num_to_try]:
            old_r = current_residues[idx]
            current_residues[idx] = target_r
            new_score = _fast_score(current_residues, primes, search_limit)
            
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
            # Random perturbation
            idx = random.randint(0, n_primes - 1)
            old_r = current_residues[idx]
            new_r = random.randint(0, primes[idx] - 1)
            current_residues[idx] = new_r
            new_score = _fast_score(current_residues, primes, search_limit)
            
            # Accept if better or neutral, or with small prob if worse (SA-like)
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
            else:
                # Simulated annealing style acceptance?
                # Or just revert.
                if random.random() < 0.05:
                    current_score = new_score
                    stagnant_count += 1
                else:
                    current_residues[idx] = old_r
                    stagnant_count += 1
            
            if stagnant_count > stagnation_threshold:
                stagnant_count = 0
                # Restart from best
                current_residues = best_residues[:]
                current_score = best_score

    # Final polish: Check with a very high limit to get the true score
    # The evaluator caps at 10^5, but checking 10^5 might be slow for the final check?
    # _fast_score with 100000:
    # 168 primes. 100000 length array.
    # Slicing 100000 array 168 times.
    # Should be fast (< 1s).
    final_score = _fast_score(best_residues, primes, 100000)
    
    # If final_score is 100000, it might be higher.
    # But we return residues, not score.
    # The evaluator will calculate the score.
    # So we just return the residues that gave the best score in our search.
    # However, if our search limit was 5000, we might have a solution that is actually 10000+ 
    # but we only optimized for 5000.
    # The random moves when hitting the limit should have helped explore further.
    
    return best_residues
