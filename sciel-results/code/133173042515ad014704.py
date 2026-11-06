import time
import random

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

def _mask_to_arr(mask):
    arr = []
    for x in range(1, 513):
        if mask & (1 << (x - 1)):
            arr.append(x)
    return arr

def _covered_diffs(mask):
    """Return set of d values that are covered."""
    covered = set()
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            covered.add(d)
    return covered

def _greedy_construct(deadline):
    """Greedy set cover: iteratively add elements that maximize new covered differences."""
    mask = 0
    arr = []
    covered = set()
    
    # Start with a seed element to bootstrap
    # Try several starting points and pick the best
    best_initial = None
    best_initial_score = -1
    
    for start in [1, 16, 32, 64, 128, 256, 100, 200, 300, 400]:
        trial_mask = 1 << (start - 1)
        trial_covered = _covered_diffs(trial_mask)
        if len(trial_covered) > best_initial_score:
            best_initial_score = len(trial_covered)
            best_initial = start
    
    if best_initial is not None:
        mask = 1 << (best_initial - 1)
        arr.append(best_initial)
        covered = _covered_diffs(mask)
    
    # Greedily add elements
    while len(arr) < 32 and time.time() < deadline - 5:
        best_gain = -1
        best_elem = None
        best_new_covered = None
        
        # Sample candidate positions rather than checking all 512
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for cand in candidates[:200]:  # Check up to 200 random candidates
            if mask & (1 << (cand - 1)):
                continue
            trial_mask = mask | (1 << (cand - 1))
            trial_covered = _covered_diffs(trial_mask)
            gain = len(trial_covered - covered)
            if gain > best_gain:
                best_gain = gain
                best_elem = cand
                best_new_covered = trial_covered
            elif gain == best_gain and best_elem is not None:
                # Tie-break: prefer element that creates more total 3-APs
                pass
        
        if best_elem is None:
            # No improvement found; pick random
            available = [x for x in range(1, 513) if not (mask & (1 << (x - 1)))]
            if not available:
                break
            best_elem = random.choice(available)
            mask |= (1 << (best_elem - 1))
            arr.append(best_elem)
            covered = _covered_diffs(mask)
        else:
            mask |= (1 << (best_elem - 1))
            arr.append(best_elem)
            covered = best_new_covered
        
        if best_gain == 0 and len(arr) > 10:
            # Diminishing returns; switch to random fill
            available = [x for x in range(1, 513) if not (mask & (1 << (x - 1)))]
            remaining = 32 - len(arr)
            if available and remaining > 0:
                picks = random.sample(available, min(remaining, len(available)))
                for p in picks:
                    mask |= (1 << (p - 1))
                    arr.append(p)
                covered = _covered_diffs(mask)
            break
    
    arr.sort()
    return arr, mask, covered

def _swap_local_search(arr, mask, covered, deadline):
    """Targeted swap local search: try swaps that cover missing differences."""
    cur_arr = arr[:]
    cur_mask = mask
    cur_covered = covered
    cur_score = len(cur_covered)
    
    all_diffs = set(range(1, 256))
    missing = all_diffs - cur_covered
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try each pair of positions for swap
        for i in range(32):
            if time.time() >= deadline - 5:
                break
            old_i = cur_arr[i]
            for j in range(i + 1, 32):
                old_j = cur_arr[j]
                # Try swapping with random external elements
                for _ in range(3):
                    new_i = random.randint(1, 512)
                    if new_i == old_i or new_i == old_j or (cur_mask & (1 << (new_i - 1))):
                        continue
                    
                    trial_mask = cur_mask & ~(1 << (old_i - 1)) | (1 << (new_i - 1))
                    trial_covered = _covered_diffs(trial_mask)
                    trial_score = len(trial_covered)
                    
                    if trial_score > cur_score:
                        cur_arr[i] = new_i
                        cur_mask = trial_mask
                        cur_covered = trial_covered
                        cur_score = trial_score
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if cur_score > _score_bitset(cur_mask):
            pass  # Consistency check
        
    cur_arr.sort()
    return cur_arr, cur_mask, cur_covered, cur_score

def _crossover(p1, p2):
    k = random.randint(1, 31)
    child = p1[:k] + p2[k:]
    seen = set()
    cleaned = []
    for x in child:
        if 1 <= x <= 512 and x not in seen:
            cleaned.append(x)
            seen.add(x)
    if len(cleaned) < 32:
        available = [x for x in range(1, 513) if x not in seen]
        if available:
            to_add = random.sample(available, min(32 - len(cleaned), len(available)))
            cleaned.extend(to_add)
    if len(cleaned) > 32:
        random.shuffle(cleaned)
        cleaned = cleaned[:32]
    return sorted(cleaned)

def _mutate(arr):
    idx = random.randint(0, 31)
    new = random.randint(1, 512)
    if new != arr[idx]:
        arr[idx] = new
    seen = set()
    cleaned = []
    for x in arr:
        if 1 <= x <= 512 and x not in seen:
            cleaned.append(x)
            seen.add(x)
    if len(cleaned) < 32:
        available = [x for x in range(1, 513) if x not in seen]
        if available:
            to_add = random.sample(available, min(32 - len(cleaned), len(available)))
            cleaned.extend(to_add)
    if len(cleaned) > 32:
        random.shuffle(cleaned)
        cleaned = cleaned[:32]
    return sorted(cleaned)

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    rng = random.Random(42)
    
    best_arr = None
    best_score = 0
    best_mask = 0
    best_covered = set()
    
    # Phase 1: Multiple greedy constructions with different seeds
    num_greedy = 8
    greedy_budget = (deadline - start_time) * 0.3
    greedy_deadline = start_time + greedy_budget
    
    for g in range(num_greedy):
        if time.time() >= greedy_deadline:
            break
        # Different random seeds for diversity
        random.seed(42 + g * 1000)
        arr, mask, covered = _greedy_construct(greedy_deadline)
        score = _score_bitset(mask)
        if score > best_score:
            best_score = score
            best_arr = arr
            best_mask = mask
            best_covered = covered
    
    # Phase 2: Local search on best
    if time.time() < deadline - 20:
        best_arr, best_mask, best_covered, best_score = _swap_local_search(
            best_arr, best_mask, best_covered, deadline
        )
    
    # Phase 3: GA to explore further
    if time.time() < deadline - 15:
        pop_size = 80
        population = []
        
        # Initialize population with best + diverse individuals
        population.append((best_arr[:], best_score))
        
        for i in range(pop_size - 1):
            if i < pop_size // 4:
                start = rng.randint(1, 512 - 32)
                pop = list(range(start, start + 32))
            elif i < pop_size // 2:
                d = rng.randint(1, 15)
                start = rng.randint(1, 512 - 31 * d)
                if start + 31 * d <= 512:
                    pop = [start + j * d for j in range(32)]
                else:
                    pop = rng.sample(range(1, 513), 32)
            elif i < 3 * pop_size // 4:
                # Structured: two interleaved APs
                d1 = rng.randint(2, 8)
                d2 = rng.randint(2, 8)
                start1 = rng.randint(1, 512 - 15 * d1)
                start2 = rng.randint(1, 512 - 15 * d2)
                pop = [start1 + j * d1 for j in range(16)] + [start2 + j * d2 for j in range(16)]
                seen = set()
                pop = [x for x in pop if 1 <= x <= 512 and x not in seen and not seen.add(x)]
                if len(pop) < 32:
                    available = [x for x in range(1, 513) if x not in seen]
                    pop.extend(rng.sample(available, min(32 - len(pop), len(available))))
            else:
                pop = rng.sample(range(1, 513), 32)
            
            pop = sorted(pop)
            mask = _arr_to_mask(pop)
            score = _score_bitset(mask)
            population.append((pop, score))
        
        population.sort(key=lambda x: x[1], reverse=True)
        
        if population[0][1] > best_score:
            best_score = population[0][1]
            best_arr = population[0][0]
            best_mask = _arr_to_mask(best_arr)
            best_covered = _covered_diffs(best_mask)
        
        generation = 0
        while time.time() < deadline - 10:
            generation += 1
            new_population = []
            
            elite_count = max(1, pop_size // 10)
            new_population.extend(population[:elite_count])
            
            while len(new_population) < pop_size:
                t1 = random.sample(population, 3)
                t2 = random.sample(population, 3)
                p1 = max(t1, key=lambda x: x[1])[0]
                p2 = max(t2, key=lambda x: x[1])[0]
                
                child = _crossover(p1, p2)
                
                if rng.random() < 0.4:
                    child = _mutate(child)
                
                score = _score_bitset(_arr_to_mask(child))
                new_population.append((child, score))
            
            population = new_population
            population.sort(key=lambda x: x[1], reverse=True)
            
            if population[0][1] > best_score:
                best_score = population[0][1]
                best_arr = population[0][0]
                best_mask = _arr_to_mask(best_arr)
                best_covered = _covered_diffs(best_mask)
                
                # Occasional local search on new best
                if generation % 3 == 0 and time.time() < deadline - 15:
                    best_arr, best_mask, best_covered, best_score = _swap_local_search(
                        best_arr, best_mask, best_covered, deadline
                    )
                    population[0] = (best_arr, best_score)
                    population.sort(key=lambda x: x[1], reverse=True)
    
    # Final local search pass
    if time.time() < deadline - 3:
        best_arr, best_mask, best_covered, best_score = _swap_local_search(
            best_arr, best_mask, best_covered, deadline
        )
    
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
