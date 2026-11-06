import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    def run_greedy(noise_scale=0.0, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        chosen = []
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        candidates = list(range(1, N + 1))
        
        while candidates:
            if time.time() > deadline:
                break
            
            best_x = None
            best_score = float('inf')
            best_candidates = []
            
            # Vectorized validity check
            valid_mask = np.ones(len(candidates), dtype=bool)
            cost_array = np.zeros(len(candidates), dtype=np.int32)
            
            for i, x in enumerate(candidates):
                # Check 2*x
                if counts[2 * x] >= 2:
                    valid_mask[i] = False
                    continue
                if counts[2 * x] == 1:
                    cost_array[i] += 1
                
                # Check x + y for all y in chosen
                if chosen:
                    sums = x + np.array(chosen)
                    c = counts[sums]
                    if np.any(c >= 2):
                        valid_mask[i] = False
                    else:
                        cost_array[i] += np.sum(c == 1)
            
            if not np.any(valid_mask):
                break
            
            # Get valid candidates and their costs
            valid_indices = np.where(valid_mask)[0]
            valid_costs = cost_array[valid_indices]
            
            if noise_scale > 0:
                noise = rng.random(len(valid_indices)) * noise_scale
                scores = valid_costs + noise
            else:
                scores = valid_costs.astype(float)
            
            min_score = np.min(scores)
            min_indices = valid_indices[scores == min_score]
            
            best_x = candidates[rng.choice(min_indices)]
            
            # Add to set
            chosen.append(best_x)
            counts[2 * best_x] += 1
            if len(chosen) > 1:
                for y in chosen[:-1]:
                    counts[best_x + y] += 1
            
            candidates.remove(best_x)
        
        return chosen
    
    def improve_set(chosen, max_time=5.0):
        """Local search: try to add elements by removing others."""
        chosen = list(chosen)
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        for x in chosen:
            counts[2 * x] += 1
            for y in chosen:
                if y > x:
                    counts[x + y] += 1
        
        improved = True
        start_improve = time.time()
        
        while improved and time.time() < start_improve + max_time and time.time() < deadline:
            improved = False
            chosen_set = set(chosen)
            
            # Try to add each candidate
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                
                # Check if x is valid
                valid = True
                if counts[2 * x] >= 2:
                    continue
                if chosen:
                    sums = x + np.array(chosen)
                    c = counts[sums]
                    if np.any(c >= 2):
                        continue
                
                # x is valid, add it
                counts[2 * x] += 1
                for y in chosen:
                    counts[x + y] += 1
                chosen.append(x)
                chosen_set.add(x)
                improved = True
        
        return chosen
    
    # Phase 1: Deterministic min-cost greedy
    initial_set = run_greedy(noise_scale=0.0)
    best_set = initial_set
    best_size = len(initial_set)
    
    # Phase 2: Try to improve with local search
    if time.time() < deadline - 10:
        improved = improve_set(best_set, max_time=5.0)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Phase 3: Randomized restarts with noise
    seed_counter = 0
    while time.time() < deadline - 5:
        seed_counter += 1
        noise = 0.3 + 0.4 * random.random()
        current_set = run_greedy(noise_scale=noise, seed=seed_counter)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
        
        # Try to improve the best set found
        if time.time() < deadline - 10:
            improved = improve_set(best_set, max_time=3.0)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    return best_set
