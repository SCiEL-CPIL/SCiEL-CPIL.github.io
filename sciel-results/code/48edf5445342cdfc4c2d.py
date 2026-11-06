import time
import random

def _score_bitset(mask):
    """Count distinct d>0 such that a, a+d, a+2d all in set."""
    count = 0
    # d can range from 1 to 255
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _arr_to_mask(arr):
    mask = 0
    for x in arr:
        mask |= (1 << (x - 1))
    return mask

def _mask_to_arr(mask):
    arr = []
    for x in range(1, 513):
        if mask & (1 << (x - 1)):
            arr.append(x)
    return arr

def _repair(arr):
    """Ensure arr is a list of 32 distinct integers in 1..512."""
    # Remove duplicates and out of range
    seen = set()
    cleaned = []
    for x in arr:
        if 1 <= x <= 512 and x not in seen:
            cleaned.append(x)
            seen.add(x)
    
    if len(cleaned) > 32:
        # Randomly remove excess
        random.shuffle(cleaned)
        cleaned = cleaned[:32]
    elif len(cleaned) < 32:
        # Add random distinct integers
        all_vals = set(range(1, 513))
        available = all_vals - seen
        if len(available) < 32 - len(cleaned):
            # Fallback: just fill with 1..32 if something went wrong
            return list(range(1, 33))
        to_add = random.sample(list(available), 32 - len(cleaned))
        cleaned.extend(to_add)
        
    return sorted(cleaned)

def _crossover(p1, p2):
    k = random.randint(1, 31)
    child = p1[:k] + p2[k:]
    return _repair(child)

def _mutate(arr):
    # Point mutation: change one element
    idx = random.randint(0, 31)
    old = arr[idx]
    new = random.randint(1, 512)
    if new != old:
        arr[idx] = new
        # Check for duplicates
        if arr.count(new) > 1:
            # If duplicate, try to fix by removing the other instance? 
            # Simpler: just repair
            pass
    return _repair(arr)

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    rng = random.Random(42)
    
    pop_size = 200
    population = []
    
    # Initialize
    # Mix of random, dense, and structured
    for i in range(pop_size):
        if i < pop_size // 3:
            # Dense block
            start = rng.randint(1, 512 - 32)
            pop = list(range(start, start + 32))
        elif i < 2 * pop_size // 3:
            # Arithmetic progression with small d
            d = rng.randint(1, 10)
            start = rng.randint(1, 512 - 31*d)
            pop = [start + j*d for j in range(32)]
        else:
            # Random
            pop = rng.sample(range(1, 513), 32)
        
        pop = sorted(pop)
        population.append((pop, _score_bitset(_arr_to_mask(pop))))
        
    # Sort initial population
    population.sort(key=lambda x: x[1], reverse=True)
    best_arr = population[0][0]
    best_score = population[0][1]
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        new_population = []
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        new_population.extend(population[:elite_count])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            t1 = random.sample(population, 3)
            t2 = random.sample(population, 3)
            p1 = max(t1, key=lambda x: x[1])[0]
            p2 = max(t2, key=lambda x: x[1])[0]
            
            # Crossover
            child = _crossover(p1, p2)
            
            # Mutation
            if rng.random() < 0.5:
                child = _mutate(child)
            
            # Evaluate
            score = _score_bitset(_arr_to_mask(child))
            new_population.append((child, score))
            
        population = new_population
        population.sort(key=lambda x: x[1], reverse=True)
        
        if population[0][1] > best_score:
            best_score = population[0][1]
            best_arr = population[0][0]
            
        # Optional: Local search on the best individual every few generations
        if generation % 5 == 0 and time.time() < deadline - 10:
            # Simple hill climb on best_arr
            cur_arr = best_arr[:]
            cur_mask = _arr_to_mask(cur_arr)
            cur_score = best_score
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for idx in range(32):
                    old = cur_arr[idx]
                    # Try a few random replacements
                    for _ in range(10):
                        new = rng.randint(1, 512)
                        if new == old or (cur_mask & (1 << (new - 1))):
                            continue
                        trial_mask = cur_mask & ~(1 << (old - 1)) | (1 << (new - 1))
                        trial_score = _score_bitset(trial_mask)
                        if trial_score > cur_score:
                            cur_arr[idx] = new
                            cur_arr.sort()
                            cur_mask = trial_mask
                            cur_score = trial_score
                            improved = True
                            break
                    if improved:
                        break
                if cur_score > best_score:
                    best_score = cur_score
                    best_arr = cur_arr
                    # Update population with this improved individual
                    population[0] = (best_arr, best_score)
                    population.sort(key=lambda x: x[1], reverse=True)

    # Final validation
    result = sorted(set(best_arr))
    if len(result) != 32:
        result = list(range(1, 33))
    result = [x for x in result if 1 <= x <= 512]
    if len(result) < 32:
        extra = [x for x in range(1, 513) if x not in result]
        result.extend(extra[:32 - len(result)])
        result = sorted(result[:32])
        
    return result
