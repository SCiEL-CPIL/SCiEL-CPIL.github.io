"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute pair sums for all possible (i, j) with i <= j
    # We need to map these sums to the target indices or mark as out-of-bounds
    # To speed up the objective function, we can precompute which pairs contribute to which targets.
    # However, the number of pairs is ~ (129*130)/2 = 8385.
    # The number of targets is 129.
    # We can create a sparse-like structure or just iterate.
    # Given the 80s limit and DE's slowness, we need the objective to be very fast.
    
    # Let's precompute the contribution matrix M of shape (num_pairs, num_targets)
    # M[k, t] = 1 if pair k sums to target_sums[t], else 0.
    # This allows objective to be: counts = M @ x_pairs
    # where x_pairs is the vector of pair indicators (0 if i,j not both in A, 1 otherwise).
    # But x_pairs is not linear in x (binary vector for A).
    # x_pairs[k] = x[i] * x[j]. This is quadratic.
    # DE works on continuous variables. We will map continuous x in [0,1] to binary A by thresholding 0.5.
    # Then we compute the counts based on the thresholded A.
    
    # To make the objective fast, we can vectorize the count calculation.
    # A is a vector of length 129.
    # Pairs (i, j) with i <= j.
    # Sum s = i + j.
    # We want counts[s] for s in [128, 256].
    
    # Let's precompute all pairs (i, j) with i <= j and i, j in [0, 128].
    # There are 129*130/2 = 8385 pairs.
    # For each pair, compute sum s.
    # If 128 <= s <= 256, it contributes to counts[s-128].
    
    pairs_i = []
    pairs_j = []
    pair_target_idx = [] # index in target_sums, or -1 if out of range
    
    for i in range(129):
        for j in range(i, 129):
            s = i + j
            if min_sum <= s <= max_sum:
                # Find index in target_sums
                # Since target_sums is contiguous range 128..256, index is s - 128
                idx = s - min_sum
                pairs_i.append(i)
                pairs_j.append(j)
                pair_target_idx.append(idx)
            
    pairs_i = np.array(pairs_i)
    pairs_j = np.array(pairs_j)
    pair_target_idx = np.array(pair_target_idx)
    
    num_pairs = len(pairs_i)
    
    def objective(x):
        # x is a vector of length 129, values in [0, 1]
        # Threshold to binary A
        A = (x >= 0.5).astype(int)
        
        # If A is empty, return high penalty
        if np.sum(A) == 0:
            return 1e9
            
        # Compute counts for target sums
        # A[i] * A[j] for all pairs
        # We can do this efficiently?
        # A is 1D.
        # We have pairs_i, pairs_j arrays.
        # active_pairs = A[pairs_i] * A[pairs_j]
        # Then bincount active_pairs into target indices.
        
        active = A[pairs_i] * A[pairs_j]
        
        # If no active pairs, penalty
        if np.sum(active) == 0:
            return 1e9
            
        counts = np.bincount(active * pair_target_idx, weights=active, minlength=num_targets)
        
        # Calculate max multiplicity
        max_mult = np.max(counts)
        
        # Calculate coverage penalty
        # We need every target sum to have count >= 1
        # Penalty = sum of (1 - min(counts[s], 1)) for all s
        # Or simply sum of (1 - counts[s]) for counts[s] == 0?
        # Let's use a large penalty for uncovered sums.
        uncovered = np.sum(counts == 0)
        
        # Objective: minimize max_mult + large_penalty * uncovered
        # We want to prioritize coverage first, then minimize max.
        # If we minimize max_mult directly, DE might find a set with low max but missing coverage.
        # So we must penalize missing coverage heavily.
        
        # Also, we want to discourage very large A? Not strictly necessary, 
        # but larger A increases pairs, potentially increasing max_mult.
        # The minimization of max_mult should handle this naturally if coverage is satisfied.
        
        penalty = 1000 * uncovered
        return max_mult + penalty

    # Run Differential Evolution
    # Bounds: [0, 1] for each of the 129 elements
    bounds = [(0, 1)] * 129
    
    # Parameters for DE
    # maxiter: number of generations
    # popsize: population size (multiplier of number of vars)
    # Since 129 vars is small, we can use larger popsize.
    # But DE is slow. We have 80 seconds.
    # Let's try a moderate configuration.
    
    best_A = None
    best_score = float('inf')
    
    try:
        # Use a callback to track best? DE returns best at the end.
        # To be safe, we might want to run multiple short runs or one long run.
        # Let's run one long run with a time-based stop?
        # scipy DE doesn't have a direct time limit, only maxiter.
        # We can estimate iterations per second.
        # Let's set maxiter to a reasonable number and check time after.
        # Or use a wrapper that stops early?
        # Simpler: run DE with maxiter=50, popsize=15.
        # If time allows, run more.
        
        # Strategy: Run DE with maxiter=100.
        # If it finishes early, good. If it takes too long, we might kill it.
        # But we can't kill a running scipy call easily without threads.
        # We must ensure it finishes within deadline.
        
        # Let's estimate: 129 vars, popsize=10 -> 1290 individuals.
        # Each eval is fast (vectorized numpy).
        # 100 generations * 1290 evals = 129,000 evals.
        # Should be fast enough in Python/Numpy?
        # 129,000 * (small numpy ops) ~ maybe 1-2 seconds?
        # Let's try maxiter=200, popsize=15.
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=0.01, 
            seed=42,
            polish=False # Polish can be slow, and we do thresholding anyway
        )
        
        if result.success or result.fun < best_score:
            best_A_candidate = (result.x >= 0.5).astype(int)
            # Verify and calculate actual score
            # We need to return the set A as a list of indices where A[i]=1
            
            # Let's verify coverage and get peak
            A = best_A_candidate
            if np.sum(A) > 0:
                active = A[pairs_i] * A[pairs_j]
                counts = np.bincount(active * pair_target_idx, weights=active, minlength=num_targets)
                if np.all(counts > 0):
                    peak = np.max(counts)
                    if peak < best_score:
                        best_score = peak
                        best_A = [i for i, v in enumerate(A) if v == 1]
        
    except Exception:
        pass

    # If DE didn't find a valid solution or timed out/failed, try a fallback
    # Fallback: Greedy or a known good set?
    # The current program's fallback was list(range(64, 129)).
    # Let's check if that's valid and what its score is.
    # A = [64, ..., 128]. Size 65.
    # Sums range from 128 to 256.
    # Let's calculate its peak.
    
    if best_A is None:
        # Try a few random restarts of DE with different seeds if time permits
        if time.time() < deadline - 10.0:
            for seed in [1, 2, 3]:
                try:
                    result = differential_evolution(
                        objective, 
                        bounds, 
                        maxiter=50, 
                        popsize=10, 
                        tol=0.1, 
                        seed=seed,
                        polish=False
                    )
                    A_candidate = (result.x >= 0.5).astype(int)
                    if np.sum(A_candidate) > 0:
                        active = A_candidate[pairs_i] * A_candidate[pairs_j]
                        counts = np.bincount(active * pair_target_idx, weights=active, minlength=num_targets)
                        if np.all(counts > 0):
                            peak = np.max(counts)
                            if peak < best_score:
                                best_score = peak
                                best_A = [i for i, v in enumerate(A_candidate) if v == 1]
                except Exception:
                    continue
                if time.time() > deadline - 5.0:
                    break
        
        # Final fallback: Simple construction
        if best_A is None:
            # Use the fallback from the seed program
            # A = list(range(64, 129))
            A = list(range(64, 129))
            # Let's verify this fallback
            A_vec = np.zeros(129, dtype=int)
            for i in A:
                A_vec[i] = 1
            active = A_vec[pairs_i] * A_vec[pairs_j]
            counts = np.bincount(active * pair_target_idx, weights=active, minlength=num_targets)
            if np.all(counts > 0):
                best_A = A
            else:
                # Should not happen for this specific range, but just in case
                # Return a dense set that definitely covers
                best_A = list(range(128)) # 0..127
                # Check 0..127
                # Sums 0..254.
                # Target 128..256.
                # 255 = 127+128 (128 not in A). 127+127=254.
                # So 0..127 doesn't cover 255, 256.
                # list(range(64, 129)) covers 128..256.
                # 64+64=128. 128+128=256.
                # So range(64, 129) is valid.
                best_A = list(range(64, 129))
    
    return best_A
# EVOLVE-BLOCK-END
