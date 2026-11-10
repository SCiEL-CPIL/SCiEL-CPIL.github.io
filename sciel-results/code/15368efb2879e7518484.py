import random
import math
import time
import numpy as np

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We will search for a solution that covers up to M
    # Start small, expand dynamically
    M = 200
    max_M = 20000 # Hard cap to prevent infinite memory/time if it runs away
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in primes]
    
    def compute_counts(res, M_val):
        counts = np.zeros(M_val + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = res[i]
            # Determine start index
            if r == 0:
                start = p
            else:
                start = r
            # Add counts using numpy slicing for speed
            counts[start:M_val+1:p] += 1
        return counts

    def get_score(counts, M_val):
        # Find first index where count < 2
        # np.where returns indices
        # We want the first x in 1..M_val such that counts[x] < 2
        # If none, return M_val
        # Note: counts[0] is not used for score 1..M
        bad = np.where(counts[1:M_val+1] < 2)[0]
        if len(bad) == 0:
            return M_val
        return bad[0] # This is 0-indexed relative to slice, so actual index is bad[0] + 1 - 1 = bad[0]?
        # Wait. counts[1:M_val+1] corresponds to x=1..M_val.
        # If bad[0] is 0, it means counts[1] < 2. So score is 0.
        # If bad[0] is k, it means counts[1+k] < 2. So score is k.
        # The score is the largest L such that 1..L are covered.
        # If x=1 is bad, L=0.
        # If x=2 is bad (and 1 is good), L=1.
        # So score = bad[0] if bad[0] is the index in the slice.
        # Let's verify: slice index 0 is x=1. If x=1 is bad, bad[0]=0. Score 0. Correct.
        # If x=2 is bad, bad[0]=1. Score 1. Correct.
        return bad[0]

    # Initial counts and score
    counts = compute_counts(residues, M)
    current_score = get_score(counts, M)
    best_score = current_score
    best_residues = residues[:]
    
    start_time = time.time()
    time_limit = 110.0 # Leave buffer
    
    # Greedy Local Search with Expanding Window
    while (time.time() - start_time) < time_limit:
        # If we cover current M, expand M
        if current_score >= M:
            new_M = min(int(M * 1.1) + 10, max_M)
            if new_M == M:
                break # Reached max_M and fully covered
            M = new_M
            # Extend counts array
            old_counts = counts
            counts = np.zeros(M + 1, dtype=np.int32)
            counts[:len(old_counts)] = old_counts
            # Recompute counts for the new range? 
            # Easier: just recompute full counts for new M. 
            # Or update only the new part? 
            # Recomputing is O(M * sum(1/p)). For M=20000, sum(1/p)~log(log(1000))~2. 
            # 20000 * 2 * 168 is too high for python loop, but numpy is fast.
            # Let's just recompute fully for simplicity and correctness.
            counts = compute_counts(residues, M)
            current_score = get_score(counts, M)
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
            continue
        
        # Find failure point
        x_fail = current_score + 1
        
        if x_fail > M:
            # Should not happen if logic above is correct, but safe check
            break
            
        # Find primes that do NOT cover x_fail
        # We want to change a residue r_p to x_fail % p
        # This will increment count[x_fail] by 1 (if it wasn't covered) or 0 (if it was, but we are fixing <2)
        # Actually, if count[x_fail] < 2, at most 1 prime covers it.
        # So at least n_primes - 1 primes do NOT cover it.
        
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            if x_fail % p != residues[i]:
                candidates.append(i)
        
        if not candidates:
            # This implies all primes cover x_fail, so count >= n_primes.
            # But we said count < 2. Contradiction unless n_primes < 2.
            # n_primes = 168. So this is impossible.
            break
            
        # Pick a candidate. 
        # Strategy: Pick the one that minimizes "damage" elsewhere?
        # Or just random? 
        # To improve performance, we can try a few candidates and pick the best.
        # Let's try 5 random candidates and pick the one that yields the best new score.
        
        best_candidate = -1
        best_new_score = -1
        best_new_counts = None
        
        # Sample up to 10 candidates
        sample_size = min(10, len(candidates))
        sampled_candidates = random.sample(candidates, sample_size)
        
        for i in sampled_candidates:
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            if new_r == old_r:
                continue
                
            # Apply move
            residues[i] = new_r
            
            # Update counts
            # Decrement old
            start_old = old_r if old_r > 0 else p
            counts[start_old:M+1:p] -= 1
            # Increment new
            start_new = new_r if new_r > 0 else p
            counts[start_new:M+1:p] += 1
            
            new_score = get_score(counts, M)
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_candidate = i
                best_new_counts = counts.copy()
                
            # Revert move for next trial
            residues[i] = old_r
            counts[start_old:M+1:p] += 1
            counts[start_new:M+1:p] -= 1
            
        # Apply the best move found
        if best_candidate != -1:
            i = best_candidate
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            residues[i] = new_r
            # Re-apply the update (since we reverted in the loop)
            start_old = old_r if old_r > 0 else p
            counts[start_old:M+1:p] -= 1
            start_new = new_r if new_r > 0 else p
            counts[start_new:M+1:p] += 1
            
            current_score = best_new_score
            
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
                
        else:
            # No move improved the score? 
            # This might be a local optimum. 
            # Try a random move to escape?
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
                
            residues[i] = new_r
            start_old = old_r if old_r > 0 else p
            counts[start_old:M+1:p] -= 1
            start_new = new_r if new_r > 0 else p
            counts[start_new:M+1:p] += 1
            
            new_score = get_score(counts, M)
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
                
    return best_residues
