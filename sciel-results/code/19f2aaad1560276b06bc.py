import time
import random
import math

def _score_bitset(mask):
    """Count distinct d>0 such that a, a+d, a+2d all in set."""
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _arr_to_mask(arr):
    mask = 0
    for x in arr:
        mask |= (1 << (x - 1))
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 120s limit, keep 2s margin
    
    best_score = 0
    best_arr = []
    
    def get_random_new_elem(current_mask):
        # Try random
        for _ in range(20):
            val = random.randint(1, 512)
            if not (current_mask & (1 << (val - 1))):
                return val
        # Fallback: linear scan
        for val in range(1, 513):
            if not (current_mask & (1 << (val - 1))):
                return val
        return None

    def simulated_annealing(initial_arr, temp_start, temp_end, steps):
        nonlocal best_score, best_arr
        
        current_arr = initial_arr[:]
        current_mask = _arr_to_mask(current_arr)
        current_score = _score_bitset(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_arr = current_arr[:]
        
        current_temp = temp_start
        # Avoid division by zero or negative steps
        if steps <= 0:
            return
        cooling_rate = (temp_end / temp_start) ** (1.0 / steps) if temp_start > 0 else 1.0
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Generate neighbor
            idx = random.randint(0, 31)
            old_val = current_arr[idx]
            
            new_val = get_random_new_elem(current_mask)
            if new_val is None:
                continue
            
            # Apply move
            current_mask &= ~(1 << (old_val - 1))
            current_mask |= (1 << (new_val - 1))
            
            new_score = _score_bitset(current_mask)
            delta = new_score - current_score
            
            if delta > 0 or (current_temp > 1e-9 and random.random() < math.exp(delta / current_temp)):
                current_score = new_score
                current_arr[idx] = new_val
                
                if current_score > best_score:
                    best_score = current_score
                    best_arr = current_arr[:]
            else:
                # Revert
                current_mask &= ~(1 << (new_val - 1))
                current_mask |= (1 << (old_val - 1))
            
            current_temp *= cooling_rate
            if current_temp < 1e-9:
                break

    # Initial Populations
    strategies = []
    
    # Random
    strategies.append(lambda: random.sample(range(1, 513), 32))
    
    # APs
    def make_ap():
        d = random.randint(1, 15)
        start = random.randint(1, 512 - 31 * d)
        if start + 31 * d <= 512:
            return [start + i * d for i in range(32)]
        return random.sample(range(1, 513), 32)
    strategies.append(make_ap)
    
    # Two interleaved APs
    def make_two_aps():
        d1 = random.randint(1, 10)
        d2 = random.randint(1, 10)
        s1 = random.randint(1, 512 - 15 * d1)
        s2 = random.randint(1, 512 - 15 * d2)
        set1 = [s1 + i * d1 for i in range(16)]
        set2 = [s2 + i * d2 for i in range(16)]
        combined = set1 + set2
        seen = set()
        valid = []
        for x in combined:
            if 1 <= x <= 512 and x not in seen:
                valid.append(x)
                seen.add(x)
        if len(valid) < 32:
            avail = [x for x in range(1, 513) if x not in seen]
            valid.extend(random.sample(avail, 32 - len(valid)))
        return valid[:32]
    strategies.append(make_two_aps)
    
    # "Spread" set
    def make_spread():
        d = random.randint(12, 20)
        start = random.randint(1, 512 - 31 * d)
        if start + 31 * d <= 512:
            return [start + i * d for i in range(32)]
        return random.sample(range(1, 513), 32)
    strategies.append(make_spread)

    # Run SA for multiple iterations
    num_restarts = 300
    steps_per_run = 8000
    temp_start = 10.0
    temp_end = 0.01
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        strat = random.choice(strategies)
        init_arr = strat()
        init_arr = sorted(init_arr)
        
        simulated_annealing(init_arr, temp_start, temp_end, steps_per_run)
        
        # Occasionally, perturb the best solution found so far and restart
        if r > 20 and random.random() < 0.3 and best_arr:
            perturbed = best_arr[:]
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 31)
                old_val = perturbed[idx]
                new_val = get_random_new_elem(_arr_to_mask(perturbed))
                if new_val:
                    perturbed[idx] = new_val
            simulated_annealing(sorted(perturbed), 5.0, 0.01, steps_per_run)

    # Final Local Search (Hill Climbing) on the best solution
    if best_arr and time.time() < deadline - 1:
        current_arr = best_arr[:]
        current_mask = _arr_to_mask(current_arr)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(32):
                old_val = current_arr[i]
                for new_val in range(1, 513):
                    if new_val == old_val:
                        continue
                    if current_mask & (1 << (new_val - 1)):
                        continue
                    
                    trial_mask = current_mask & ~(1 << (old_val - 1))
                    trial_mask |= (1 << (new_val - 1))
                    trial_score = _score_bitset(trial_mask)
                    
                    if trial_score > current_score:
                        current_score = trial_score
                        current_arr[i] = new_val
                        current_mask = trial_mask
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_arr = current_arr[:]
                        break # Restart inner loop

    # Ensure result is valid
    if not best_arr or len(best_arr) != 32:
        best_arr = list(range(1, 33))
    
    seen = set()
    final_arr = []
    for x in best_arr:
        if 1 <= x <= 512 and x not in seen:
            final_arr.append(x)
            seen.add(x)
    if len(final_arr) < 32:
        avail = [x for x in range(1, 513) if x not in seen]
        final_arr.extend(avail[:32 - len(final_arr)])
    
    return sorted(final_arr)
