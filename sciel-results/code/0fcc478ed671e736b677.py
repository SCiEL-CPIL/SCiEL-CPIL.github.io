"""Improved B2[2] construction with density-aware greedy and efficient local search."""
import time
import random
import numpy as np

def _fast_count_violations(mask, n=1000):
    """
    Count number of sums with > 2 representations (violation penalty).
    Uses convolution for fast C-speed execution.
    """
    if not np.any(mask):
        return 0
    
    conv = np.convolve(mask, mask, mode='full')
    
    # diag[k] = 1 if k is even and mask[k//2] is True and k//2 < n
    diag = np.zeros(len(conv), dtype=np.int32)
    even_k = np.arange(0, len(conv), 2)
    mask_indices = even_k // 2
    valid_mask_idx = mask_indices < n
    diag_vals = np.zeros(len(even_k), dtype=np.int32)
    diag_vals[valid_mask_idx] = mask[mask_indices[valid_mask_idx]]
    diag[even_k] = diag_vals
    
    total_counts = conv + diag
    violations = np.sum(total_counts > 4)
    
    return int(violations)

def _greedy_construct(n=1000):
    """Greedy construction with density-aware ordering."""
    chosen = []
    counts = {}
    
    # First pass: simple greedy
    for x in range(1, n + 1):
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
        if all(trial[s] <= 2 for s in new_sums):
            chosen.append(x)
            counts = trial
    
    return chosen

def _greedy_density_aware(n=1000, seed=42):
    """
    Density-aware greedy: at each step, pick the candidate that creates
    the fewest new sum collisions, with tie-breaking by randomization.
    """
    rng = random.Random(seed)
    chosen = []
    counts = np.zeros(2 * n + 1, dtype=np.int32)  # counts[s] for s in 2..2n
    
    # Precompute which sums each candidate would create
    for _ in range(n):
        if time.time() > 110:
            break
        
        best_candidate = -1
        best_score = float('inf')
        best_sums = None
        
        # Sample a subset of candidates for efficiency
        candidates = [x for x in range(1, n + 1) if x not in set(chosen)]
        
        if not candidates:
            break
        
        # For large sets, sample candidates
        if len(candidates) > 200:
            sampled = rng.sample(candidates, 200)
        else:
            sampled = candidates
        
        for x in sampled:
            # Compute new sums and check if they would exceed limit
            new_sums = set()
            for y in chosen:
                s = x + y
                if s > 2 * n:
                    continue
                new_sums.add(s)
            new_sums.add(2 * x)
            
            # Check validity
            valid = True
            for s in new_sums:
                if counts[s] >= 2:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Score: number of sums that would become tight (count becomes 2)
            score = 0
            for s in new_sums:
                if counts[s] == 1:
                    score += 1
            
            # Add small random perturbation for tie-breaking
            score += rng.random() * 0.01
            
            if score < best_score:
                best_score = score
                best_candidate = x
                best_sums = new_sums
        
        if best_candidate == -1:
            # No valid candidate found, try all remaining
            for x in candidates:
                new_sums = set()
                for y in chosen:
                    s = x + y
                    if s > 2 * n:
                        continue
                    new_sums.add(s)
                new_sums.add(2 * x)
                
                valid = True
                for s in new_sums:
                    if counts[s] >= 2:
                        valid = False
                        break
                
                if valid:
                    best_candidate = x
                    best_sums = new_sums
                    break
        
        if best_candidate == -1:
            break
        
        # Add the best candidate
        chosen.append(best_candidate)
        for s in best_sums:
            counts[s] += 1
    
    return chosen

def _local_search(mask, n=1000, time_budget=5.0, seed=42):
    """
    Local search to improve set size while maintaining validity.
    Uses add/remove/swap operations with acceptance criteria.
    """
    rng = random.Random(seed)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_size = int(np.sum(mask))
    best_mask = mask.copy()
    best_size = current_size
    
    # Compute current violation count (should be 0 for valid set)
    current_violations = _fast_count_violations(mask, n)
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    attempts = 0
    no_improve_streak = 0
    max_streak = 5000
    
    while time.time() < deadline:
        attempts += 1
        if attempts % 1000 == 0:
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        if no_improve_streak > max_streak:
            # Reset temperature and try harder
            T = max(T, 0.5)
            no_improve_streak = 0
        
        # Randomly pick an operation
        op = rng.random()
        
        if op < 0.35:
            # Try to add a random element
            candidates = np.nonzero(~mask)[0]
            if len(candidates) == 0:
                continue
            idx = candidates[rng.randint(0, len(candidates) - 1)]
            new_mask = mask.copy()
            new_mask[idx] = True
            new_violations = _fast_count_violations(new_mask, n)
            new_size = int(np.sum(new_mask))
            
            if new_violations == 0 and new_size > current_size:
                mask = new_mask
                current_size = new_size
                current_violations = 0
                if current_size > best_size:
                    best_mask = mask.copy()
                    best_size = current_size
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
            elif new_violations > 0:
                delta_v = new_violations - current_violations
                if delta_v > 0:
                    prob = np.exp(-delta_v / max(T, 1e-10))
                    if rng.random() < prob:
                        mask = new_mask
                        current_violations = new_violations
                        current_size = new_size
                        no_improve_streak += 1
                        
        elif op < 0.55:
            # Try to remove a random element
            candidates = np.nonzero(mask)[0]
            if len(candidates) == 0:
                continue
            idx = candidates[rng.randint(0, len(candidates) - 1)]
            new_mask = mask.copy()
            new_mask[idx] = False
            new_violations = _fast_count_violations(new_mask, n)
            new_size = int(np.sum(new_mask))
            
            if new_violations < current_violations:
                mask = new_mask
                current_violations = new_violations
                current_size = new_size
                no_improve_streak += 1
            elif new_violations == current_violations and new_size < current_size:
                if rng.random() < 0.05:
                    mask = new_mask
                    current_violations = new_violations
                    current_size = new_size
                    no_improve_streak += 1
                    
        else:
            # Swap: remove one, try to add another
            candidates_in = np.nonzero(mask)[0]
            candidates_out = np.nonzero(~mask)[0]
            if len(candidates_in) == 0 or len(candidates_out) == 0:
                continue
            idx_in = candidates_in[rng.randint(0, len(candidates_in) - 1)]
            idx_out = candidates_out[rng.randint(0, len(candidates_out) - 1)]
            new_mask = mask.copy()
            new_mask[idx_in] = False
            new_mask[idx_out] = True
            new_violations = _fast_count_violations(new_mask, n)
            new_size = int(np.sum(new_mask))
            
            if new_violations == 0 and new_size > current_size:
                mask = new_mask
                current_size = new_size
                current_violations = 0
                if current_size > best_size:
                    best_mask = mask.copy()
                    best_size = current_size
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
            elif new_violations == 0 and new_size == current_size:
                if rng.random() < 0.15:
                    mask = new_mask
                    current_violations = 0
                    no_improve_streak += 1
            elif new_violations > 0:
                delta_v = new_violations - current_violations
                if delta_v > 0:
                    prob = np.exp(-delta_v / max(T, 1e-10))
                    if rng.random() < prob:
                        mask = new_mask
                        current_violations = new_violations
                        current_size = new_size
                        no_improve_streak += 1
    
    # Return the best valid mask found
    best_violations = _fast_count_violations(best_mask, n)
    if best_violations > 0:
        # Fallback: try to repair by removing elements causing violations
        mask = best_mask.copy()
        violations = _fast_count_violations(mask, n)
        repair_attempts = 0
        while violations > 0 and repair_attempts < 1000:
            repair_attempts += 1
            # Find a random element and try removing it
            candidates = np.nonzero(mask)[0]
            if len(candidates) == 0:
                break
            idx = candidates[rng.randint(0, len(candidates) - 1)]
            new_mask = mask.copy()
            new_mask[idx] = False
            new_violations = _fast_count_violations(new_mask, n)
            if new_violations < violations:
                mask = new_mask
                violations = new_violations
        if _fast_count_violations(mask, n) == 0:
            return [i + 1 for i in np.nonzero(mask)[0]]
        return []
    
    return [i + 1 for i in np.nonzero(best_mask)[0]]

def _verify_validity(result, n=1000):
    """Verify that the result is a valid B2[2] set."""
    counts = {}
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            counts[s] = counts.get(s, 0) + 1
            if counts[s] > 2:
                return False
    return True

def construct():
    start_time = time.time()
    total_budget = 145.0  # Safety margin before 150s kill
    
    # Phase 1: Multiple greedy constructions with different seeds
    best_result = []
    best_size = 0
    
    greedy_seeds = [42, 137, 251, 389, 503, 617, 731, 853]
    for seed in greedy_seeds:
        if time.time() > start_time + total_budget - 30:
            break
        
        result = _greedy_density_aware(1000, seed=seed)
        if len(result) > best_size:
            if _verify_validity(result):
                best_result = result
                best_size = len(result)
    
    # Also try simple greedy
    simple_greedy = _greedy_construct(1000)
    if len(simple_greedy) > best_size:
        if _verify_validity(simple_greedy):
            best_result = simple_greedy
            best_size = len(simple_greedy)
    
    # Phase 2: Local search from best result
    elapsed = time.time() - start_time
    search_deadline = start_time + total_budget - 5.0
    
    if best_result and time.time() < search_deadline:
        # Convert to mask
        mask = np.zeros(1000, dtype=bool)
        for x in best_result:
            mask[x - 1] = True
        
        # Multiple local search restarts
        r = 0
        while time.time() < search_deadline:
            r += 1
            seed = 1000 + r * 7919
            time_left = search_deadline - time.time()
            if time_left < 2.0:
                break
            
            time_per_restart = min(8.0, time_left)
            
            result = _local_search(mask, n=1000, time_budget=time_per_restart, seed=seed)
            
            if result and len(result) > best_size:
                if _verify_validity(result):
                    best_result = result
                    best_size = len(result)
                    # Update mask for next iteration
                    mask = np.zeros(1000, dtype=bool)
                    for x in best_result:
                        mask[x - 1] = True
    
    # Final verification
    if best_result and not _verify_validity(best_result):
        # Fallback to simple greedy
        return _greedy_construct(1000)
    
    if not best_result:
        return _greedy_construct(1000)
    
    return best_result
