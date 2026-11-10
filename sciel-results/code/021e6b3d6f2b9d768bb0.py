import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate primes up to 1000
    def get_primes(n):
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = False
        return np.array([i for i, is_prime in enumerate(sieve) if is_prime], dtype=np.int64)
    
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Initial residues: simple heuristic
    # The seed program used ((i+2)//2) % p. Let's start with something similar but randomize a bit?
    # Actually, let's stick to the seed's initialization as a baseline, or maybe 0?
    # The seed's initialization gives score 1.0 (L=84).
    # Let's use the seed's initialization to ensure we start at a known valid point.
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    
    # We need to maximize L such that every x in 1..L is covered by at least 2 congruences.
    # "Covered" means x == r_p (mod p).
    
    # Helper to compute the "first failure" L for a given residue vector
    def compute_score(res_vec):
        # We want the largest L such that count(x) >= 2 for all x in 1..L.
        # We can check up to a certain limit, say 2000 or 5000, to be safe.
        # If we reach the limit, we assume it's higher.
        limit = 5000
        # Create a grid of x values
        x_vals = np.arange(1, limit + 1, dtype=np.int64)
        
        # For each prime p_i, compute which x_vals are covered.
        # x % p_i == r_i
        # This is equivalent to (x - r_i) % p_i == 0
        
        # We can do this vectorized over primes, but memory might be an issue for 168 * 5000.
        # 168 * 5000 is 840,000 integers. That's fine.
        # Let's create a 2D boolean array: covered[i, j] is True if x_j is covered by prime i.
        
        # To save memory and time, we can accumulate counts.
        counts = np.zeros(limit, dtype=np.int32)
        
        # Batch process primes to avoid huge temporary arrays? 
        # 168 primes is small enough to loop in Python if the inner loop is vectorized.
        for i in range(n_primes):
            p = primes[i]
            r = res_vec[i]
            # x % p == r
            # Vectorized modulo
            # (x_vals - r) % p == 0
            mask = (x_vals - r) % p == 0
            counts += mask.astype(np.int32)
            
        # Find the first x where counts[x] < 2
        # x_vals are 1-indexed in the array (index 0 is x=1)
        # We need the largest L such that counts[0:L] are all >= 2.
        # So we find the first index where counts < 2.
        # If no such index, L = limit.
        
        # np.argmax returns the first occurrence of the max value.
        # We want the first occurrence of a value < 2.
        # Let's create a boolean array: fail = counts < 2
        fail = counts < 2
        if not np.any(fail):
            return limit
        first_fail_idx = np.argmax(fail)
        # x_vals[first_fail_idx] is the first failing x.
        # The score is x_vals[first_fail_idx] - 1.
        # If first_fail_idx is 0, score is 0.
        return x_vals[first_fail_idx] - 1

    current_score = compute_score(residues)
    
    # Hill Climbing
    # We will try to change residues one prime at a time.
    # For each prime i, we try all possible residues 0..p_i-1.
    # This is O(sum(p_i)) evaluations per full pass.
    # Sum of primes up to 1000 is approx 76127.
    # Each evaluation is O(N * limit) where N is number of primes? No, compute_score is O(N * limit).
    # 168 * 5000 = 840k operations.
    # 76127 * 840k is too slow.
    
    # We need a faster scoring function or a smarter search.
    # Alternative: Update counts incrementally?
    # If we change residue for prime i from r_old to r_new:
    # 1. Subtract coverage of r_old for prime i.
    # 2. Add coverage of r_new for prime i.
    # This is O(limit) per change.
    # Total per full pass: sum(p_i) * limit = 76127 * 5000 = 380 million operations.
    # In Python/NumPy, this might be borderline.
    # Let's optimize compute_score to be incremental.
    
    # Precompute x_vals
    x_vals = np.arange(1, 5001, dtype=np.int64)
    
    # Initialize counts
    counts = np.zeros(5000, dtype=np.int32)
    for i in range(n_primes):
        p = primes[i]
        r = residues[i]
        counts += ((x_vals - r) % p == 0).astype(np.int32)
        
    current_score = compute_score(residues)
    
    # Helper to get score from counts
    def score_from_counts(c):
        fail = c < 2
        if not np.any(fail):
            return 5000
        first_fail_idx = np.argmax(fail)
        return x_vals[first_fail_idx] - 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_primes):
            p = primes[i]
            r_old = residues[i]
            
            # Remove old coverage
            old_mask = ((x_vals - r_old) % p == 0).astype(np.int32)
            counts -= old_mask
            
            best_r = r_old
            best_local_score = current_score
            
            # Try all other residues
            # We can vectorize the evaluation of all residues for this prime?
            # For each candidate r in 0..p-1:
            # new_counts = counts + ((x_vals - r) % p == 0)
            # score = score_from_counts(new_counts)
            
            # This requires storing p * limit integers temporarily.
            # Max p is 997. 997 * 5000 = 5 million integers. 20MB. Acceptable.
            
            candidates = np.arange(p, dtype=np.int64)
            # Exclude r_old? Or just include it, it will give same score.
            
            # Compute all new counts
            # x_vals shape (5000,), candidates shape (p,)
            # We need (x_vals - r) % p for all r.
            # This is (5000, p) array.
            # (x_vals[:, None] - candidates[None, :]) % p
            
            # Memory: 5000 * 997 * 8 bytes (int64) = ~40MB.
            # Then modulo.
            # Then comparison == 0.
            # Then sum along axis 0 to get counts? No, we need to add to base counts.
            
            # Let's do it in chunks if p is large? Or just do it.
            # 40MB is fine.
            
            # Optimization: We only care about the first failure point.
            # But computing full score is easiest.
            
            try:
                # Vectorized generation of masks for all candidates
                # diff = x_vals[:, None] - candidates[None, :]
                # mods = diff % p
                # masks = (mods == 0).astype(np.int32)
                
                # new_counts = counts[None, :] + masks
                # scores = score_from_counts(new_counts) -- this is hard to vectorize score_from_counts directly.
                
                # Let's just loop through candidates for this prime, but use the precomputed masks?
                # No, generating the (5000, p) array is the expensive part.
                # Let's just loop through r in 0..p-1.
                # O(p * limit) per prime.
                # Sum is 380M ops.
                # In NumPy, 380M simple ops might take ~1-2 seconds?
                # Let's try.
                
                for r_new in range(p):
                    if r_new == r_old:
                        continue
                    new_mask = ((x_vals - r_new) % p == 0).astype(np.int32)
                    new_counts = counts + new_mask
                    s = score_from_counts(new_counts)
                    if s > best_local_score:
                        best_local_score = s
                        best_r = r_new
                        
            except Exception:
                pass
            
            # Restore best
            residues[i] = best_r
            new_mask = ((x_vals - best_r) % p == 0).astype(np.int32)
            counts += new_mask
            
            if best_local_score > current_score:
                current_score = best_local_score
                improved = True
            elif best_r != r_old:
                # If we changed residue but score didn't improve, we should revert?
                # No, we set residues[i] = best_r which is r_old if no improvement.
                # So no change if no improvement.
                pass
                
    return [int(r) for r in residues]

# Keep the function name as required
# The task says "Keep the function names and signatures of the seed program."
# Seed program has `def construct():`
# So I just need to output the code block with `construct`.
