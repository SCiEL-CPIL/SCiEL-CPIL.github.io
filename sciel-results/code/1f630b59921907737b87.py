import random
import itertools
import time

def _is_sidon(subset):
    """Check if a list of distinct ints is a Sidon set (all pair sums including diagonals unique)."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _count_sidon_4subsets(A):
    """Count the number of 4-element subsets of A that are Sidon sets."""
    count = 0
    for subset in itertools.combinations(A, 4):
        if _is_sidon(list(subset)):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s kill limit)
    
    # Start with a random set
    current = sorted(random.sample(range(129), 12))
    current_cost = _count_sidon_4subsets(current)
    best = list(current)
    best_cost = current_cost
    
    if current_cost == 0:
        return current
    
    temperature = 100.0
    cooling = 0.995
    
    while time.time() < deadline:
        if temperature < 0.01:
            break
            
        # Pick two random indices to swap
        i, j = random.sample(range(12), 2)
        
        # Try replacing current[i] with a random value not in the set
        old_val = current[i]
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val in current and new_val != old_val:
            continue
        
        # Create candidate
        candidate = list(current)
        candidate[i] = new_val
        candidate.sort()
        
        candidate_cost = _count_sidon_4subsets(candidate)
        
        # Accept if better or with probability based on temperature
        delta = candidate_cost - current_cost
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(temperature, 1e-10)):
            current = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = list(current)
                best_cost = current_cost
                if best_cost == 0:
                    return best
        
        temperature *= cooling
    
    return best
