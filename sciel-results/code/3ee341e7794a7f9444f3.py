"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    num_targets = max_sum - min_sum + 1  # 129 targets
    
    def compute_counts(A):
        """Fast computation of unordered pair-sum counts using numpy convolution."""
        A = np.array(A, dtype=np.int64)
        indicator = np.zeros(n + 1, dtype=np.int64)
        indicator[A] = 1
        conv = np.convolve(indicator, indicator, mode='full')
        
        # conv[s] = number of ordered pairs (a,b) with a+b=s where a,b in [0,n] and indicator[a]=indicator[b]=1
        # For unordered pairs: 
        #   If a != b: ordered count = 2 * unordered count
        #   If a == b: ordered count = 1 * unordered count
        # So: unordered = (ordered + diag) / 2 where diag = 1 if s even and s/2 in A, else 0
        
        counts = np.zeros(max_sum + 1, dtype=np.int64)
        for s in range(min_sum, max_sum + 1):
            if s < len(conv):
                ordered = conv[s]
                if s % 2 == 0:
                    half = s // 2
                    if 0 <= half <= n and indicator[half] == 1:
                        diag = 1
                    else:
                        diag = 0
                else:
                    diag = 0
                unordered = (ordered + diag) // 2
                counts[s] = unordered
        return counts
    
    def get_cost(A):
        """Return peak multiplicity, or inf if any target is uncovered."""
        counts = compute_counts(A)
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return float('inf')
        return int(np.max(target_counts))
    
    best_score = float('inf')
    best_set = None
    
    def try_set(A_list):
        nonlocal best_score, best_set
        A_sorted = sorted(A_list)
        if len(A_sorted) == 0:
            return
        c = get_cost(A_sorted)
        if c < best_score:
            best_score = c
            best_set = A_sorted
    
    # Initial solution: contiguous interval
    initial = list(range(64, 129))
    try_set(initial)
    
    # Try symmetric intervals
    for offset in range(0, 40):
        lo = max(0, 64 - offset)
        hi = min(128, 64 + offset)
        candidate = list(range(lo, hi + 1))
        if len(candidate) > 0:
            try_set(candidate)
    
    # Random initializations
    random.seed(42)
    for size in range(25, 85):
        if time.time() > deadline - 15:
            break
        for _ in range(15):
            if time.time() > deadline - 12:
                break
            A = random.sample(range(0, 129), size)
            try_set(A)
    
    # Phase 1: Local search with add/remove/swap moves
    if best_set is not None and time.time() < deadline - 10:
        current = best_set[:]
        current_cost = best_score
        in_set = set(current)
        
        while time.time() < deadline - 5:
            improved = False
            
            # Try removing elements (from end for stability)
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                val = current[i]
                trial = current[:i] + current[i+1:]
                if len(trial) == 0:
                    continue
                tc = get_cost(trial)
                if tc <= current_cost:
                    current = trial
                    current_cost = tc
                    in_set.discard(val)
                    improved = True
                    break
            
            # Try adding elements
            if not improved:
                for new_val in range(129):
                    if time.time() > deadline - 5:
                        break
                    if new_val in in_set:
                        continue
                    trial = current + [new_val]
                    tc = get_cost(trial)
                    if tc < current_cost:
                        current = trial
                        current_cost = tc
                        in_set.add(new_val)
                        improved = True
                        break
            
            # Try swaps
            if not improved:
                for i in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    old_val = current[i]
                    for new_val in range(129):
                        if time.time() > deadline - 5:
                            break
                        if new_val == old_val or new_val in in_set:
                            continue
                        trial = current[:]
                        trial[i] = new_val
                        trial.sort()
                        tc = get_cost(trial)
                        if tc < current_cost:
                            current = trial
                            current_cost = tc
                            in_set.discard(old_val)
                            in_set.add(new_val)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        if current_cost < best_score:
            best_score = current_cost
            best_set = current[:]
    
    # Phase 2: Restart-based search with perturbation
    if time.time() < deadline - 5 and best_score > 3:
        random.seed(123)
        for restart in range(50):
            if time.time() > deadline - 3:
                break
            
            # Create a perturbed version of the best set
            if best_set is None:
                break
            
            base_size = len(best_set)
            size = base_size + random.randint(-5, 5)
            size = max(20, min(90, size))
            
            # Start with some elements from best, fill rest randomly
            A_set = set()
            if best_set and len(best_set) > 0:
                # Keep 60-80% of best elements
                keep_frac = random.uniform(0.6, 0.8)
                keep_count = int(len(best_set) * keep_frac)
                kept = random.sample(best_set, min(keep_count, len(best_set)))
                A_set.update(kept)
            
            # Add random elements to reach target size
            remaining = size - len(A_set)
            if remaining > 0:
                available = [x for x in range(129) if x not in A_set]
                if available:
                    additions = random.sample(available, min(remaining, len(available)))
                    A_set.update(additions)
            
            A = list(A_set)
            try_set(A)
            
            # Quick local improvement from this starting point
            if best_set is not None and time.time() < deadline - 2:
                current = best_set[:]
                current_cost = best_score
                in_set = set(current)
                
                for step in range(20):
                    if time.time() > deadline - 2:
                        break
                    improved = False
                    
                    # Try removing
                    for i in range(len(current) - 1, -1, -1):
                        if time.time() > deadline - 2:
                            break
                        val = current[i]
                        trial = current[:i] + current[i+1:]
                        if len(trial) == 0:
                            continue
                        tc = get_cost(trial)
                        if tc <= current_cost:
                            current = trial
                            current_cost = tc
                            in_set.discard(val)
                            improved = True
                            break
                    
                    # Try adding
                    if not improved:
                        for new_val in range(129):
                            if time.time() > deadline - 2:
                                break
                            if new_val in in_set:
                                continue
                            trial = current + [new_val]
                            tc = get_cost(trial)
                            if tc < current_cost:
                                current = trial
                                current_cost = tc
                                in_set.add(new_val)
                                improved = True
                                break
                    
                    # Try swaps
                    if not improved:
                        for i in range(len(current)):
                            if time.time() > deadline - 2:
                                break
                            old_val = current[i]
                            for new_val in range(129):
                                if time.time() > deadline - 2:
                                    break
                                if new_val == old_val or new_val in in_set:
                                    continue
                                trial = current[:]
                                trial[i] = new_val
                                trial.sort()
                                tc = get_cost(trial)
                                if tc < current_cost:
                                    current = trial
                                    current_cost = tc
                                    in_set.discard(old_val)
                                    in_set.add(new_val)
                                    improved = True
                                    break
                            if improved:
                                break
                    
                    if not improved:
                        break
                
                if current_cost < best_score:
                    best_score = current_cost
                    best_set = current[:]
                
                if best_score <= 3:
                    break
    
    # Phase 3: Final polishing local search
    if best_set is not None and time.time() < deadline - 1:
        current = best_set[:]
        current_cost = best_score
        in_set = set(current)
        
        while time.time() < deadline - 0.5:
            improved = False
            
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 0.5:
                    break
                val = current[i]
                trial = current[:i] + current[i+1:]
                if len(trial) == 0:
                    continue
                tc = get_cost(trial)
                if tc <= current_cost:
                    current = trial
                    current_cost = tc
                    in_set.discard(val)
                    improved = True
                    break
            
            if not improved:
                for new_val in range(129):
                    if time.time() > deadline - 0.5:
                        break
                    if new_val in in_set:
                        continue
                    trial = current + [new_val]
                    tc = get_cost(trial)
                    if tc < current_cost:
                        current = trial
                        current_cost = tc
                        in_set.add(new_val)
                        improved = True
                        break
            
            if not improved:
                for i in range(len(current)):
                    if time.time() > deadline - 0.5:
                        break
                    old_val = current[i]
                    for new_val in range(129):
                        if time.time() > deadline - 0.5:
                            break
                        if new_val == old_val or new_val in in_set:
                            continue
                        trial = current[:]
                        trial[i] = new_val
                        trial.sort()
                        tc = get_cost(trial)
                        if tc < current_cost:
                            current = trial
                            current_cost = tc
                            in_set.discard(old_val)
                            in_set.add(new_val)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        if current_cost < best_score:
            best_score = current_cost
            best_set = current[:]
    
    if best_set is None or len(best_set) == 0:
        return list(range(64, 129))
    
    return list(best_set)

# EVOLVE-BLOCK-END
