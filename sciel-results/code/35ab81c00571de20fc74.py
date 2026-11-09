import numpy as np
import time
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 10) # Slightly larger to be safe for size+1 checks

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = []
    best_size = 0
    
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        mask = np.ones(N + 1, dtype=bool)
        active_indices = np.arange(N + 1)
        
        # Main greedy loop
        while True:
            K = mask.sum()
            if K == 0:
                break
                
            primes_to_check = [p for p in PRIMES if p <= K]
            
            if not primes_to_check:
                break
                
            fixed = False
            for p in primes_to_check:
                active_vals = active_indices[mask]
                
                if len(active_vals) == 0:
                    break
                    
                residues = active_vals % p
                counts = np.bincount(residues, minlength=p)
                
                if counts.min() > 0:
                    min_count = counts.min()
                    candidates = np.where(counts == min_count)[0]
                    chosen = random.choice(candidates)
                    
                    mask[active_indices % p == chosen] = False
                    fixed = True
                    break
            
            if not fixed:
                break
        
        current_size = mask.sum()
        current_set = active_indices[mask].tolist()
        
        # Local Search Polish: Try to add elements back
        # We need to check if adding x keeps the set admissible.
        # Condition: For all p <= |S|+1, S U {x} misses a residue mod p.
        
        # Precompute residue counts for current set S
        # We only care about primes p <= current_size + 1
        # Actually, if we add x, size becomes K+1. We need to check p <= K+1.
        
        # Let's store the set as a Python set for fast lookup? 
        # Or just use the mask.
        
        # To speed up the check, we can precompute for each prime p <= K+1:
        # 1. Does S cover all residues? (No, it's admissible, so no for p<=K)
        # 2. Which residues are missing?
        
        # Let's maintain a structure of missing residues for each prime.
        # missing_residues[p] = set of residues r in [0, p) such that count_S(r) == 0
        
        missing = {}
        max_p_to_check = current_size + 1
        for p in PRIMES:
            if p > max_p_to_check:
                break
            active_vals = np.array(current_set)
            if len(active_vals) == 0:
                missing[p] = set(range(p))
                continue
            residues = active_vals % p
            counts = np.bincount(residues, minlength=p)
            missing[p] = set(np.where(counts == 0)[0])
            
        # Now try to add elements from [0, N] \ S
        # Iterate through candidates. 
        # To be efficient, we can just iterate 0..N and check.
        # N=5000, K~1000-2000? 
        # Check cost: For each x, check all p <= K+1. 
        # K+1 is small (primes are sparse). 
        # Number of primes <= 2000 is ~300.
        # 5000 * 300 = 1.5M operations. This is fast.
        
        added_any = True
        while added_any:
            added_any = False
            for x in range(N + 1):
                if mask[x]:
                    continue
                
                # Check if S U {x} is admissible
                # New size K_new = current_size + 1
                # Check primes p <= K_new
                valid = True
                for p in PRIMES:
                    if p > K_new:
                        break
                    # If p <= current_size, S already missed some residues.
                    # If x fills the ONLY missing residue, it becomes invalid.
                    # If p == K_new, S might have covered all residues (since p > K_old, it wasn't constrained).
                    
                    if p in missing:
                        miss_set = missing[p]
                        if not miss_set:
                            # S covered all residues? 
                            # This can happen for p > K_old.
                            # If S covered all residues, S U {x} definitely covers all.
                            # So invalid.
                            valid = False
                            break
                        else:
                            # S missed miss_set.
                            # S U {x} misses miss_set \ {x % p}
                            # It is admissible if (miss_set \ {x % p}) is not empty.
                            # i.e., len(miss_set) > 1 OR (x % p not in miss_set)
                            if len(miss_set) == 1:
                                if x % p in miss_set:
                                    valid = False
                                    break
                            # else len(miss_set) >= 2, so even if x fills one, one remains.
                    else:
                        # p > K_old but p <= K_new? 
                        # This only happens if p == K_new and K_new is prime.
                        # In this case, we didn't precompute missing[p] if we stopped at K_old?
                        # Wait, my loop for 'missing' went up to K_old + 1.
                        # So if p == K_new, it is in missing.
                        # So this else should not trigger for p <= K_new.
                        pass
                
                if valid:
                    # Add x
                    mask[x] = True
                    current_size += 1
                    added_any = True
                    # Update missing sets for all primes p <= current_size
                    # This is expensive if done naively.
                    # But we can just recompute missing sets for the next iteration?
                    # Or update incrementally.
                    # Incremental:
                    # For each p <= current_size:
                    #   If x % p was in missing[p], remove it.
                    #   If missing[p] becomes empty, and p <= current_size, then we have a violation?
                    #   Wait, if missing[p] becomes empty, it means S U {x} covers all residues mod p.
                    #   This is allowed ONLY IF p > current_size.
                    #   But we are checking p <= current_size.
                    #   So if missing[p] becomes empty, it means S U {x} is NOT admissible for p.
                    #   So we should NOT have added x.
                    #   My check above: "if len(miss_set) == 1: if x % p in miss_set: valid=False".
                    #   This correctly prevents adding x if it fills the last hole for p <= K_old.
                    #   What about p = K_new?
                    #   If p = K_new, and S missed 1 residue, and x fills it, valid=False.
                    #   If S missed >= 2, valid=True.
                    #   So the check is correct.
                    
                    # Now update missing sets for the state S U {x}
                    # We need to update 'missing' for all p <= current_size
                    # Actually, we can just recompute 'missing' from scratch in the next 'while added_any' iteration?
                    # That would be O(P * K). 
                    # P is ~300, K is ~1000. 300k ops. 
                    # If we add 100 elements, 30M ops. Might be too slow for 100 restarts?
                    # 100 restarts * (Greedy + Polish).
                    # Greedy is fast. Polish is the bottleneck.
                    
                    # Better: Update incrementally.
                    for p in PRIMES:
                        if p > current_size:
                            break
                        if p in missing:
                            r = x % p
                            if r in missing[p]:
                                missing[p].discard(r)
                            # If missing[p] is now empty, it's a problem? 
                            # No, we already verified validity. 
                            # If it became empty, it means x filled the last hole.
                            # But we only add x if it's valid.
                            # So if p <= K_old, we ensured len(miss) > 1 or x not in miss.
                            # If p == K_new, we ensured len(miss) > 1 or x not in miss.
                            # So missing[p] should NOT be empty.
                            # Wait, if p < K_old, and S missed 2, and x fills 1, miss is 1. Not empty.
                            # If p < K_old, and S missed 1, and x not in miss, miss is 1. Not empty.
                            # So missing[p] is never empty for p <= current_size after a valid addition.
                    
                    # Also, we might need to add new primes to 'missing' if current_size increased to a new prime?
                    # The loop for p in PRIMES in the check handles up to K_new.
                    # The 'missing' dict was built for K_old+1.
                    # If current_size increases, K_new increases.
                    # If K_new is a prime, we need missing[K_new].
                    # We can compute it on the fly or precompute up to N.
                    # Let's just recompute 'missing' for all p <= current_size at the start of the 'for x' loop?
                    # No, that's O(P*K) per x. Too slow.
                    
                    # Let's stick to incremental update.
                    # If current_size is now a prime P_new, we need missing[P_new].
                    # We can compute it from the current mask.
                    # This happens rarely.
                    if current_size in PRIMES:
                         active_vals = active_indices[mask]
                         residues = active_vals % current_size
                         counts = np.bincount(residues, minlength=current_size)
                         missing[current_size] = set(np.where(counts == 0)[0])
                         
        # End of polish
        if current_size > best_size:
            best_size = current_size
            best_set = active_indices[mask].tolist()

    return best_set
