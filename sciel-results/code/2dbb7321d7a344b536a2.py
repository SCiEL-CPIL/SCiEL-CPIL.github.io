import random
import time
import numpy as np

def _compute_counts(A):
    """
    Compute representation counts for all targets in [128, 256].
    Returns a numpy array of counts of length 129 (indices 0..128 for targets 128..256).
    """
    A_set = set(A)
    counts = np.zeros(129, dtype=int)
    # Optimization: iterate over unique elements to avoid redundant checks?
    # Since A is a set/list of unique elements, iterating A is fine.
    # For each a in A, we need b such that a+b = t.
    # b = t - a.
    # Condition: a <= b and b in A_set.
    
    # Vectorized approach:
    # For each a, the targets covered are a + A.
    # We can construct a matrix of contributions?
    # |A| is small (~20-30). 129 targets.
    # Looping over A (30) * Targets (129) is ~4000 ops. Very fast.
    
    for a in A:
        # For each target t in [128, 256]:
        # b = t - a
        # if a <= b and b in A_set: count[t] += 1
        
        # Let's iterate targets
        for t in range(128, 257):
            b = t - a
            if a <= b and b in A_set:
                counts[t - 128] += 1
                
    return counts

def _compute_counts_fast(A, A_set):
    """
    Faster count computation using numpy where possible, or optimized loops.
    """
    counts = np.zeros(129, dtype=int)
    for a in A:
        # b must be in A and b >= a
        # t = a + b
        # So for each b in A where b >= a, t = a+b is a target.
        # If 128 <= t <= 256, increment counts[t-128].
        
        # We can filter A for b >= a
        # But A is a list. Converting to array each time is overhead.
        # Simple loop is likely faster for small A.
        for b in A:
            if b < a:
                continue
            t = a + b
            if 128 <= t <= 256:
                counts[t - 128] += 1
    return counts

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Strategy:
    # 1. Initialize A with a dense structure to ensure coverage.
    #    A = range(0, 129) covers everything with high multiplicity.
    #    We want to remove elements to reduce peak multiplicity.
    #    
    #    Alternative: Start with a random valid set and optimize.
    #    Let's try a constructive approach first:
    #    To cover [128, 256], we need pairs.
    #    If we take A = [0, 1, ..., 128], peak multiplicity is high.
    #    If we take A = [64, 65, ..., 128], sums are [128, 256].
    #    Let's check multiplicity for A = [64..128].
    #    t=128: 64+64 (1)
    #    t=129: 64+65 (1)
    #    ...
    #    t=192: 64+128, 65+127, ..., 96+96. (33 reps)
    #    t=256: 128+128 (1)
    #    Peak is around the middle.
    
    # Let's try to find a better set.
    # Randomized Local Search / Simulated Annealing.
    
    # Initial A: Random subset that covers all targets.
    # Greedy construction:
    A = []
    A_set = set()
    counts = np.zeros(129, dtype=int)
    
    # Greedy: Add elements that cover the most uncovered targets.
    uncovered = set(range(128, 257))
    
    for _ in range(129):
        if not uncovered:
            break
        if time.time() > deadline - 30:
            break
            
        best_candidate = None
        best_newly_covered = 0
        best_new_counts = None
        best_peak = float('inf')
        
        # Sample candidates to speed up? Or all?
        # 129 candidates.
        for x in range(129):
            if x in A_set:
                continue
                
            # Calculate delta
            delta = np.zeros(129, dtype=int)
            for y in A:
                t = x + y
                if 128 <= t <= 256:
                    delta[t - 128] += 1
            t_2x = 2 * x
            if 128 <= t_2x <= 256:
                delta[t_2x - 128] += 1
                
            new_counts = counts + delta
            newly_covered = 0
            for i in range(129):
                if counts[i] == 0 and new_counts[i] > 0:
                    newly_covered += 1
                    
            if newly_covered == 0:
                continue
                
            # Score: Maximize newly covered, then minimize peak
            max_new = np.max(new_counts)
            # We prefer candidates that cover more, and have lower peak.
            # If newly_covered > best_newly_covered, it's better.
            # If equal, lower peak is better.
            
            if newly_covered > best_newly_covered or (newly_covered == best_newly_covered and max_new < best_peak):
                best_newly_covered = newly_covered
                best_candidate = x
                best_new_counts = new_counts
                best_peak = max_new
                
        if best_candidate is None:
            break
            
        A.append(best_candidate)
        A_set.add(best_candidate)
        counts = best_new_counts
        
    # Ensure validity
    if not np.all(counts > 0):
        # Fallback: Add random elements until valid
        while not np.all(counts > 0) and len(A) < 129:
            x = random.randint(0, 128)
            if x not in A_set:
                A.append(x)
                A_set.add(x)
                counts = _compute_counts(A)
                
    # If still invalid (shouldn't happen), use full range
    if not np.all(counts > 0):
        A = list(range(129))
        A_set = set(A)
        counts = _compute_counts(A)
        
    current_peak = int(np.max(counts))
    best_peak = current_peak
    best_A = list(A)
    
    # Simulated Annealing
    T = 50.0
    cooling_rate = 0.995
    min_T = 0.5
    
    # Precompute A_set
    A_set = set(A)
    
    while time.time() < deadline - 5:
        if T < min_T:
            break
            
        move_type = random.random()
        
        if len(A) > 20 and move_type < 0.3:
            # Remove
            idx = random.randint(0, len(A) - 1)
            old_elem = A[idx]
            A_set.remove(old_elem)
            A.pop(idx)
            
            new_counts = _compute_counts(A)
            
            if np.all(new_counts > 0):
                new_peak = int(np.max(new_counts))
                # Objective: Minimize peak, then sum of squares
                old_obj = int(np.sum(counts**2))
                new_obj = int(np.sum(new_counts**2))
                
                delta_obj = new_obj - old_obj
                delta_peak = new_peak - current_peak
                
                # Accept if peak decreases, or peak same and obj decreases, or with prob
                if delta_peak < 0:
                    accept = True
                elif delta_peak == 0:
                    accept = delta_obj < 0 or random.random() < np.exp(-delta_obj / T)
                else:
                    # Peak increased, only accept with low prob
                    accept = random.random() < np.exp(-delta_peak * 10 / T)
                    
                if accept:
                    counts = new_counts
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                else:
                    A.append(old_elem)
                    A_set.add(old_elem)
            else:
                A.append(old_elem)
                A_set.add(old_elem)
                
        elif move_type < 0.5 and len(A) < 100:
            # Add
            x = random.randint(0, 128)
            if x in A_set:
                continue
            A.append(x)
            A_set.add(x)
            
            new_counts = _compute_counts(A)
            new_peak = int(np.max(new_counts))
            
            old_obj = int(np.sum(counts**2))
            new_obj = int(np.sum(new_counts**2))
            
            delta_obj = new_obj - old_obj
            delta_peak = new_peak - current_peak
            
            if delta_peak < 0:
                accept = True
            elif delta_peak == 0:
                accept = delta_obj < 0 or random.random() < np.exp(-delta_obj / T)
            else:
                accept = random.random() < np.exp(-delta_peak * 10 / T)
                
            if accept:
                counts = new_counts
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = list(A)
            else:
                A.pop()
                A_set.remove(x)
                
        else:
            # Swap
            idx = random.randint(0, len(A) - 1)
            old_elem = A[idx]
            new_elem = random.randint(0, 128)
            
            if new_elem in A_set and new_elem != old_elem:
                continue
                
            A[idx] = new_elem
            A_set.remove(old_elem)
            A_set.add(new_elem)
            
            new_counts = _compute_counts(A)
            
            if np.all(new_counts > 0):
                new_peak = int(np.max(new_counts))
                
                old_obj = int(np.sum(counts**2))
                new_obj = int(np.sum(new_counts**2))
                
                delta_obj = new_obj - old_obj
                delta_peak = new_peak - current_peak
                
                if delta_peak < 0:
                    accept = True
                elif delta_peak == 0:
                    accept = delta_obj < 0 or random.random() < np.exp(-delta_obj / T)
                else:
                    accept = random.random() < np.exp(-delta_peak * 10 / T)
                    
                if accept:
                    counts = new_counts
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                else:
                    A[idx] = old_elem
                    A_set.remove(new_elem)
                    A_set.add(old_elem)
            else:
                A[idx] = old_elem
                A_set.remove(new_elem)
                A_set.add(old_elem)
                
        T *= cooling_rate
        
    return best_A
