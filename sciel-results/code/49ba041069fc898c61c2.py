import time
import random
import sys

DEADLINE_SECONDS = 85.0  # Safety margin for 90s limit

def _has_divisibility_conflict(sums):
    """
    Check if any two distinct values in `sums` divide each other.
    Uses a sieve-like approach for efficiency.
    """
    if len(sums) < 2:
        return False
    
    max_val = max(sums)
    # Create a boolean array for quick lookup
    present = [False] * (max_val + 1)
    for s in sums:
        present[s] = True
    
    # For each possible divisor d, check if d and any multiple of d are both present
    for d in range(1, max_val // 2 + 1):
        if not present[d]:
            continue
        # Check multiples 2d, 3d, ...
        m = 2 * d
        while m <= max_val:
            if present[m]:
                return True
            m += d
    
    return False

def _can_add(current_sums_set, x):
    """
    Check if x can be added to the set defined by current_sums_set.
    Returns (new_sums_set) or None.
    """
    if x in current_sums_set:
        return None
        
    # Check if any new sum s+x already exists in current_sums_set
    # This would mean two different subsets have the same sum
    for s in current_sums_set:
        if s + x in current_sums_set:
            return None
    
    new_sums = set(current_sums_set)
    new_sums.add(x)
    for s in current_sums_set:
        new_sums.add(s + x)
    
    if _has_divisibility_conflict(new_sums):
        return None
        
    return new_sums

def _build_sums(elements):
    """
    Rebuild the set of subset sums from scratch for a list of elements.
    Returns the set of sums if valid, None otherwise.
    """
    sums = set()
    for e in elements:
        new_sums = _can_add(sums, e)
        if new_sums is None:
            return None
        sums = new_sums
    return sums

def _greedy_construct(pool, order_key, start_time, deadline):
    """
    Greedy construction.
    """
    elements = []
    sums_set = set()
    
    candidates = list(pool)
    if order_key == 'asc':
        candidates.sort()
    elif order_key == 'desc':
        candidates.sort(reverse=True)
    elif order_key == 'random':
        random.shuffle(candidates)
    elif order_key == 'prime_bias':
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        candidates.sort(key=lambda x: (not is_prime(x), x))
    else:
        random.shuffle(candidates)
        
    for x in candidates:
        if time.time() > deadline:
            break
        res = _can_add(sums_set, x)
        if res is not None:
            sums_set = res
            elements.append(x)
            
    return elements, sums_set

def _simulated_annealing_search(initial_elements, initial_sums, start_time, deadline):
    """
    Perform a random walk search to escape local optima.
    Allows 'downhill' moves (removals) to reshape the set.
    """
    current_elements = list(initial_elements)
    current_sums = set(initial_sums)
    best_elements = list(initial_elements)
    best_sums = set(initial_sums)
    best_size = len(initial_elements)
    
    used = set(current_elements)
    all_candidates = list(range(1, 257))
    
    # Cooling schedule parameters
    temperature = 1.0
    cooling_factor = 0.99
    
    steps = 0
    max_steps_per_temp = 50
    
    while time.time() < deadline - 2:
        if steps >= max_steps_per_temp:
            temperature *= cooling_factor
            steps = 0
            if temperature < 0.01:
                break
        
        steps += 1
        
        # 1. Try to Add (Uphill move)
        if random.random() < 0.7: # Bias towards adding
            unused = [x for x in all_candidates if x not in used]
            if unused:
                x = random.choice(unused)
                new_sums = _can_add(current_sums, x)
                if new_sums is not None:
                    # Accept improvement
                    current_sums = new_sums
                    current_elements.append(x)
                    used.add(x)
                    
                    if len(current_elements) > best_size:
                        best_size = len(current_elements)
                        best_elements = list(current_elements)
                        best_sums = set(current_sums)
                    continue
        
        # 2. Try to Swap (Neutral move)
        # Remove one, try to add another
        if len(current_elements) > 1:
            rem_idx = random.randint(0, len(current_elements) - 1)
            rem_elem = current_elements[rem_idx]
            
            # Rebuild sums without rem_elem
            trial_elements = current_elements[:rem_idx] + current_elements[rem_idx+1:]
            trial_sums = _build_sums(trial_elements)
            
            if trial_sums is not None:
                # Try to add a random unused element
                used_trial = set(trial_elements)
                unused = [x for x in all_candidates if x not in used_trial]
                if unused:
                    # Try a few random candidates
                    for _ in range(5):
                        if not unused: break
                        x = random.choice(unused)
                        new_sums = _can_add(trial_sums, x)
                        if new_sums is not None:
                            # Accept swap (size same)
                            current_elements = trial_elements + [x]
                            current_sums = new_sums
                            used = set(current_elements)
                            break
        
        # 3. Try to Remove (Downhill move)
        # This helps escape local optima where no adds/swaps work
        elif len(current_elements) > 2:
            if random.random() < 0.1: # Low probability
                rem_idx = random.randint(0, len(current_elements) - 1)
                rem_elem = current_elements[rem_idx]
                
                trial_elements = current_elements[:rem_idx] + current_elements[rem_idx+1:]
                trial_sums = _build_sums(trial_elements)
                
                if trial_sums is not None:
                    current_elements = trial_elements
                    current_sums = trial_sums
                    used = set(current_elements)
        
    return best_elements, best_sums

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_elements = []
    best_sums = set()
    best_size = 0
    
    pools = [
        list(range(1, 257)),
        list(range(2, 257)),
        list(range(5, 257)),
        list(range(10, 257)),
        list(range(20, 257)),
        list(range(50, 257)),
        list(range(100, 257)),
    ]
    
    orders = ['asc', 'desc', 'random', 'prime_bias']
    
    # Phase 1: Systematic greedy construction
    for pool in pools:
        for order in orders:
            if time.time() > deadline - 20:
                break
            elems, sums = _greedy_construct(pool, order, start_time, deadline)
            if len(elems) > best_size:
                best_size = len(elems)
                best_elements = elems
                best_sums = sums
    
    # Phase 2: Random restarts with local search (Simulated Annealing)
    max_restarts = 50
    for _ in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        pool = random.choice(pools)
        order = random.choice(orders)
        elems, sums = _greedy_construct(pool, order, start_time, deadline)
        
        if len(elems) > 0:
            improved_elems, improved_sums = _simulated_annealing_search(elems, sums, start_time, deadline - 5)
            if len(improved_elems) > best_size:
                best_size = len(improved_elems)
                best_elements = improved_elems
                best_sums = improved_sums
    
    # Phase 3: Final polish on the best solution
    if best_elements and time.time() < deadline - 2:
        # Try to add any remaining elements to the best set
        used = set(best_elements)
        current_sums = set(best_sums)
        
        remaining = [x for x in range(1, 257) if x not in used]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline - 0.5:
                break
            res = _can_add(current_sums, x)
            if res is not None:
                current_sums = res
                best_elements.append(x)
                best_size += 1

    return [int(x) for x in best_elements]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
