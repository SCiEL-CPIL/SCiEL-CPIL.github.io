import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=20
    # There are n*(n+1)/2 = 210 pairs
    pairs_i = np.array([i for i in range(n) for j in range(i, n)], dtype=np.int32)
    pairs_j = np.array([j for i in range(n) for j in range(i, n)], dtype=np.int32)
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)

    def is_valid_set(A):
        if len(A) != n:
            return False
        if len(set(A)) != n:
            return False
        if min(A) < 1 or max(A) > max_val:
            return False
        return True

    best_score = float('inf')
    best_set = None
    
    # Strategy: Union of a Geometric Progression (GP) and an Arithmetic Progression (AP)
    # GP: a * r^k for k=0..len_gp-1
    # AP: b + d*j for j=0..len_ap-1
    # We need len_gp + len_ap = 20, distinct integers <= 10000.
    
    # To maximize collisions:
    # GP helps A*A: (a*r^i)*(a*r^j) = a^2 * r^(i+j). Sums of exponents collide.
    # AP helps A+A: (b+d*i)+(b+d*j) = 2b + d(i+j). Sums of indices collide.
    
    # We will iterate over possible splits and parameters.
    # To keep it fast, we will sample a grid of parameters.
    
    # Parameters to search:
    # len_gp: 1 to 19
    # a (GP start): 1 to 20
    # r (GP ratio): integers 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20
    # b (AP start): 1 to 100 (small to avoid overlap with large GP terms, or large to avoid small AP terms)
    # d (AP step): 1 to 100
    
    # Since we have 85 seconds, we can do a fairly coarse grid.
    # Let's estimate the number of evaluations.
    # If we do 20 * 20 * 15 * 100 * 100 = 6,000,000 evals.
    # Each eval is ~50-100 us (vectorized numpy unique on 210 elements).
    # 6M * 100us = 600s. Too slow.
    
    # We need to reduce the search space or speed up evaluation.
    # Optimization: 
    # 1. Pre-filter distinctness and bounds quickly.
    # 2. Use a smaller grid initially, then refine.
    # 3. Or, use random sampling of the parameter space.
    
    # Let's use random sampling with a structured bias towards "good" parameters.
    # Good GP ratios: small integers.
    # Good AP steps: small integers.
    
    # Also, we should try to balance the sizes.
    # Common splits: 10 GP + 10 AP, or 8 GP + 12 AP, etc.
    
    # Let's generate a pool of candidate sets.
    candidates = []
    
    # 1. Pure GP
    for r in [2, 3, 4, 5]:
        for a in [1, 2]:
            seq = []
            curr = a
            for _ in range(n):
                if curr > max_val: break
                seq.append(curr)
                curr *= r
            if len(seq) == n:
                candidates.append(seq)
                
    # 2. Pure AP
    for d in [1, 2, 5, 10]:
        for b in [1, 10, 100]:
            seq = [b + i*d for i in range(n)]
            if max(seq) <= max_val:
                candidates.append(seq)
                
    # 3. Mixed GP + AP
    # We will sample random parameters for mixed sets.
    # To ensure distinctness, we can generate the GP and AP, combine, and check.
    # If duplicates or out of bounds, discard.
    
    import random
    
    # Define search ranges
    gp_ratios = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    gp_starts = list(range(1, 21))
    ap_steps = list(range(1, 51))
    ap_starts = list(range(1, 501))
    
    # Number of random trials
    num_trials = 20000
    
    for _ in range(num_trials):
        if time.time() > deadline - 5:
            break
            
        # Randomly choose split
        len_gp = random.randint(2, 18)
        len_ap = n - len_gp
        
        # Randomly choose GP params
        r = random.choice(gp_ratios)
        a = random.choice(gp_starts)
        
        gp_seq = []
        curr = a
        for _ in range(len_gp):
            if curr > max_val:
                gp_seq = None
                break
            gp_seq.append(curr)
            curr *= r
            
        if gp_seq is None or len(gp_seq) != len_gp:
            continue
            
        # Randomly choose AP params
        d = random.choice(ap_steps)
        b = random.choice(ap_starts)
        
        ap_seq = []
        for j in range(len_ap):
            val = b + j * d
            if val > max_val:
                ap_seq = None
                break
            ap_seq.append(val)
            
        if ap_seq is None or len(ap_seq) != len_ap:
            continue
            
        # Combine
        combined = gp_seq + ap_seq
        
        # Check distinctness
        if len(set(combined)) != n:
            continue
            
        # Check bounds (already checked in generation, but just in case)
        if min(combined) < 1 or max(combined) > max_val:
            continue
            
        candidates.append(combined)
        
        # Early termination if we have enough good candidates?
        # No, let's collect a batch and evaluate.
        # But evaluating 20000 candidates takes time.
        # Let's evaluate in chunks.
        
    # Actually, let's interleave generation and evaluation to save memory and time.
    # Reset bests
    best_score = float('inf')
    best_set = None
    
    # Re-evaluate the simple candidates first
    for cand in candidates[:20]: # Just the pure ones and first few
        if is_valid_set(cand):
            score = evaluate(cand)
            if score < best_score:
                best_score = score
                best_set = cand[:]
                
    # Now do the random search with immediate evaluation
    # We need to be careful with time.
    # Let's estimate time per eval again.
    # numpy unique on 210 ints is very fast. Maybe 20-50us.
    # 80s / 50us = 1,600,000 evals.
    # So we can afford a lot of random trials.
    
    # Let's run a loop that generates and evaluates random mixed sets.
    # We will use a more targeted randomization.
    
    # Bias towards smaller ratios and steps to keep numbers small and collisions high.
    # Also, ensure AP and GP ranges don't overlap too much to maintain distinctness easily,
    # or allow overlap and filter.
    
    # Let's refine the random generation:
    # Choose len_gp
    # Choose r from [2, 3, 4, 5] mostly
    # Choose a from [1, 2, 3]
    # Choose d from [1, 2, 3, 5, 10]
    # Choose b from [1, 10, 50, 100, 500, 1000]
    
    # This reduces the search space significantly while covering the "structured" part of the space.
    
    gp_ratios_biased = [2, 2, 3, 3, 4, 4, 5, 5, 6, 8, 10]
    gp_starts_biased = [1, 1, 2, 2, 3, 5]
    ap_steps_biased = [1, 1, 2, 2, 3, 5, 5, 10, 10, 20]
    ap_starts_biased = [1, 10, 50, 100, 500, 1000, 2000, 5000]
    
    # Let's run this until deadline - 5
    while time.time() < deadline - 5:
        # Batch generation to reduce overhead? No, just one by one.
        
        len_gp = random.randint(2, 18)
        len_ap = n - len_gp
        
        r = random.choice(gp_ratios_biased)
        a = random.choice(gp_starts_biased)
        
        gp_seq = []
        curr = a
        valid_gp = True
        for _ in range(len_gp):
            if curr > max_val:
                valid_gp = False
                break
            gp_seq.append(curr)
            curr *= r
            
        if not valid_gp or len(gp_seq) != len_gp:
            continue
            
        d = random.choice(ap_steps_biased)
        b = random.choice(ap_starts_biased)
        
        ap_seq = []
        valid_ap = True
        for j in range(len_ap):
            val = b + j * d
            if val > max_val:
                valid_ap = False
                break
            ap_seq.append(val)
            
        if not valid_ap or len(ap_seq) != len_ap:
            continue
            
        combined = gp_seq + ap_seq
        
        if len(set(combined)) != n:
            continue
            
        # Evaluate
        score = evaluate(combined)
        
        if score < best_score:
            best_score = score
            best_set = combined[:]
            
    # Final check
    if best_set is None:
        # Fallback to a simple set
        best_set = list(range(1, n+1))
        best_score = evaluate(best_set)
        
    return sorted(best_set)

