import time
import random
import itertools

def _subset_sums(arr):
    """Generate all subset sums of arr."""
    n = len(arr)
    sums = set()
    sums.add(0)
    for x in arr:
        # Add x to all existing sums
        new_sums = {s + x for s in sums}
        sums.update(new_sums)
    return sums

def _is_valid(arr):
    """Check if all subset sums are distinct."""
    # If any element is 0 or negative, invalid (task says positive)
    if any(x <= 0 for x in arr):
        return False
    if len(arr) != 12:
        return False
    # Check distinctness
    sums = _subset_sums(arr)
    return len(sums) == (1 << 12)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Initial solution: Powers of 2
    best_arr = [1 << i for i in range(12)]
    best_val = max(best_arr)
    
    # Local search / Simulated Annealing
    current_arr = best_arr[:]
    current_cost = best_val
    
    T = 100.0  # Temperature
    T_min = 0.1
    alpha = 0.99
    
    while time.time() < deadline:
        # Generate a neighbor
        neighbor = current_arr[:]
        move_type = random.choice(['change', 'swap', 'nudge'])
        
        if move_type == 'change':
            idx = random.randint(0, 11)
            neighbor[idx] = random.randint(1, 2048)
        elif move_type == 'swap':
            i, j = random.sample(range(12), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else: # nudge
            idx = random.randint(0, 11)
            delta = random.choice([-1, 1, -2, 2, -5, 5])
            new_val = neighbor[idx] + delta
            if new_val > 0:
                neighbor[idx] = new_val
        
        # Check validity and cost
        # Optimization: Only check validity if potentially improving or for SA
        # For speed, we can check if max is reduced first?
        # No, we need to know if it's valid to accept.
        
        # Fast check: if max(neighbor) < best_val, it's a strong candidate
        # But we need to ensure validity.
        
        sums = _subset_sums(neighbor)
        is_valid = (len(sums) == 4096)
        
        if is_valid:
            new_val = max(neighbor)
            new_cost = new_val
        else:
            # Penalty for invalidity
            # Number of collisions = 4096 - len(sums)
            collisions = 4096 - len(sums)
            new_cost = max(neighbor) + collisions * 10000
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0:
            # Better
            current_arr = neighbor
            current_cost = new_cost
            if is_valid:
                if new_val < best_val:
                    best_val = new_val
                    best_arr = neighbor[:]
        else:
            # Worse, accept with probability
            if T > T_min:
                prob = 2.71828 ** (-delta_cost / T)
                if random.random() < prob:
                    current_arr = neighbor
                    current_cost = new_cost
        
        T *= alpha
    
    return best_arr
