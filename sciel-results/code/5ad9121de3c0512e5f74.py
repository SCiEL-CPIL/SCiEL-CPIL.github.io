"""Targeted Stochastic Greedy covering for C(19,7,4)."""
import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Targeted Stochastic Greedy Set Cover.
    1. Precompute all t-subsets.
    2. In each step, pick a random uncovered t-subset S.
    3. Generate all k-subsets containing S.
    4. Select one with probability proportional to its marginal gain.
    5. Update uncovered set.
    """
    start_time = time.time()
    time_limit = 110  # seconds
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map t-subset to index for fast lookup if needed, but we use set of tuples
    # We need to know which t-subsets are in a given block quickly.
    
    best_blocks = None
    best_count = float('inf')
    
    def _run_trial(trial_idx, alpha=1.0):
        """
        Run a single trial.
        alpha controls the 'temperature' of the selection. 
        P(block) proportional to gain^alpha.
        alpha=1 is standard stochastic greedy.
        alpha=0 is uniform random among candidates with gain>0.
        alpha->inf approaches greedy (max gain).
        """
        uncovered_local = set(all_t_subsets)
        blocks = []
        rng = random.Random(trial_idx * 12345 + 67890)
        
        # Precompute the list of points not in S for a given S? 
        # Since S changes, we compute on the fly.
        # v=19, k=7, t=4. k-t=3.
        # Remaining points count: v-t = 15.
        
        while uncovered_local:
            if time.time() - start_time > time_limit - 2:
                return None, False
                
            # 1. Pick a random uncovered t-subset to target
            # Using list(uncovered_local) is O(N) but N is small (<4000)
            target_subset = rng.choice(list(uncovered_local))
            target_set = set(target_subset)
            
            # 2. Generate candidates: all k-subsets containing target_subset
            # Remaining points
            remaining_points = [x for x in range(v) if x not in target_set]
            
            candidates = []
            # We need to choose k-t points from remaining_points
            # k-t = 3
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_subset + combo))
                candidates.append(block)
            
            # 3. Calculate gain for each candidate
            # Gain = number of t-subsets in block that are in uncovered_local
            gains = []
            for block in candidates:
                gain = 0
                # Iterate through all t-subsets of this block
                # Number of t-subsets in a k-block is C(k, t) = C(7,4) = 35
                for sub in itertools.combinations(block, t):
                    if sub in uncovered_local:
                        gain += 1
                gains.append(gain)
            
            # 4. Select block with probability proportional to gain^alpha
            # Filter out zero gains? If all are zero, something is wrong (target_subset should be covered by at least one candidate, i.e., itself, so gain >= 1)
            # Actually, target_subset is in uncovered_local, and every candidate contains target_subset.
            # So every candidate covers target_subset. Thus gain >= 1 for all candidates.
            
            if alpha == 0:
                weights = [1 if g > 0 else 0 for g in gains]
            else:
                weights = [g ** alpha for g in gains]
            
            # Ensure weights are valid for rng.choices
            # If all weights are 0 (shouldn't happen), fallback to uniform
            if sum(weights) == 0:
                weights = [1] * len(candidates)
            
            chosen_idx = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
            chosen_block = candidates[chosen_idx]
            
            # 5. Update uncovered_local
            for sub in itertools.combinations(chosen_block, t):
                uncovered_local.discard(sub)
            
            blocks.append(list(chosen_block))
            
        return blocks, True

    # Run multiple trials with different alphas and seeds
    # We can try a few different alpha values to explore different behaviors
    alphas = [1.0, 0.5, 2.0, 0.0] # 0.0 is uniform random among candidates
    
    num_trials = 50 # Adjust based on speed
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Cycle through alphas
        alpha = alphas[trial % len(alphas)]
        
        blocks, success = _run_trial(trial, alpha)
        
        if success and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    # Fallback if no trial completed in time
    if best_blocks is None:
        # Deterministic Greedy Fallback (Alpha = infinity approx)
        # Use a large alpha or just pick max gain
        uncovered_local = set(all_t_subsets)
        blocks = []
        while uncovered_local:
            target_subset = min(uncovered_local)
            target_set = set(target_subset)
            remaining_points = [x for x in range(v) if x not in target_set]
            
            best_gain = -1
            best_block = None
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_subset + combo))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered_local)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break
                
            for sub in itertools.combinations(best_block, t):
                uncovered_local.discard(sub)
            blocks.append(list(best_block))
        best_blocks = blocks
        
    return best_blocks
