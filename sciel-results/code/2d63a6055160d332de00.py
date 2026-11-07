import time
import random
import math

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 60.0  # 60s limit, keep 10s buffer
    
    # Helper to check if two masks are comparable (one is subset of other)
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    # Calculate the score of a family (list of masks)
    def calc_score(fam):
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(fam[i], fam[j]):
                    score += 1
        return score

    # Calculate delta in score if we replace fam[idx] with new_mask
    # fam is the list, idx is the index to replace, new_mask is the candidate
    def calc_delta(fam, idx, new_mask):
        delta = 0
        old_mask = fam[idx]
        for i in range(m):
            if i == idx:
                continue
            # Check old contribution
            if is_comparable(old_mask, fam[i]):
                delta -= 1
            # Check new contribution
            if is_comparable(new_mask, fam[i]):
                delta += 1
        return delta

    # Initialize family
    # A good initialization is a random set, but we can bias towards a chain
    # Let's start with a random set of 40 distinct masks
    initial_fam = random.sample(range(total_masks), m)
    current_fam = initial_fam[:]
    current_score = calc_score(current_fam)
    best_fam = current_fam[:]
    best_score = current_score
    
    fam_set = set(current_fam)
    
    # Simulated Annealing parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    iterations = 0
    max_iterations = 1000000  # Will be limited by time
    
    while time.time() < deadline and iterations < max_iterations:
        if T < T_min:
            # If T is too low, do a random restart or just break
            # Let's try a few random restarts if we have time
            if random.random() < 0.1:
                # Random restart
                current_fam = random.sample(range(total_masks), m)
                current_score = calc_score(current_fam)
                fam_set = set(current_fam)
                T = 5.0
            else:
                break
        
        # Pick a random index to replace
        idx = random.randint(0, m - 1)
        
        # Pick a random candidate mask
        # To improve efficiency, we can sample from a neighborhood
        # e.g., flip 1-3 bits of the current mask
        old_mask = current_fam[idx]
        
        # Generate a neighbor
        # 50% chance: flip 1 bit
        # 30% chance: flip 2 bits
        # 20% chance: flip 3 bits
        r = random.random()
        if r < 0.5:
            bits_to_flip = 1
        elif r < 0.8:
            bits_to_flip = 2
        else:
            bits_to_flip = 3
            
        new_mask = old_mask
        for _ in range(bits_to_flip):
            bit = random.randint(0, n - 1)
            new_mask ^= (1 << bit)
            
        # Check if new_mask is already in the family
        if new_mask in fam_set:
            continue
            
        # Calculate delta
        delta = calc_delta(current_fam, idx, new_mask)
        
        # Acceptance criterion
        if delta > 0:
            accept = True
        else:
            # Avoid division by zero or overflow
            if T > 0:
                accept = random.random() < math.exp(delta / T)
            else:
                accept = False
        
        if accept:
            # Update family
            fam_set.remove(old_mask)
            fam_set.add(new_mask)
            current_fam[idx] = new_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_fam = current_fam[:]
        
        # Cool down
        T *= cooling_rate
        iterations += 1
        
    # Convert best_fam to lists of integers
    # The task requires returning a list of lists of distinct integers
    result = []
    for mask in best_fam:
        subset = []
        for b in range(n):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
        
    return result
