import itertools
import time
import random

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(arr):
    n = len(arr)
    for k in range(n, 0, -1):
        for combo in itertools.combinations(arr, k):
            if is_sidon(combo):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Fast energy: count of distinct pair sums.
    def get_energy(s):
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return len(sums)

    # Initialize with a "dense" random set
    random.seed(42)
    
    # Generate a few initial candidates
    init_candidates = []
    # 1. Small range random
    init_candidates.append(list(random.sample(range(30), 12)))
    # 2. Clusters
    init_candidates.append([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23])
    init_candidates.append([0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32])
    # 3. Arithmetic progression
    init_candidates.append(list(range(12)))
    
    # Find the best initial candidate by exact score
    best_set = init_candidates[0]
    best_score = max_sidon_size(sorted(best_set))
    
    for cand in init_candidates[1:]:
        score = max_sidon_size(sorted(cand))
        if score < best_score:
            best_score = score
            best_set = cand
    
    # If we already have a very low score, we might be done.
    # Theoretical minimum?
    # A set of 12 integers.
    # If max Sidon subset is 4, score is 4.
    # If max Sidon subset is 5, score is 5.
    # Let's see if we can get 4.
    
    if best_score <= 4:
        return sorted(best_set)

    current = best_set[:]
    current_energy = get_energy(current)
    
    best_sa_energy = current_energy
    best_sa_set = current[:]
    
    temp = 5.0
    cool_rate = 0.9995
    min_temp = 0.1
    
    # To speed up, we only check exact score when we find a new SA energy minimum.
    # But SA energy min doesn't guarantee exact score min.
    # However, they are correlated.
    
    steps = 0
    while time.time() < deadline:
        # Generate neighbor
        neighbor = current[:]
        
        if random.random() < 0.3:
            # Replace one element
            i = random.randint(0, 11)
            old_val = neighbor[i]
            # Try to pick a value that causes collision
            # Just random for now
            new_val = random.randint(0, 128)
            if new_val in neighbor:
                # Retry or pick another
                while new_val in neighbor:
                    new_val = random.randint(0, 128)
            neighbor[i] = new_val
        else:
            # Swap two elements
            i, j = random.sample(range(12), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        new_energy = get_energy(neighbor)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < pow(2.71828, -delta / temp):
            current = neighbor
            current_energy = new_energy
            
            if new_energy < best_sa_energy:
                best_sa_energy = new_energy
                best_sa_set = current[:]
                
                # Check exact score for this new best
                sorted_set = sorted(best_sa_set)
                exact_score = max_sidon_size(sorted_set)
                if exact_score < best_score:
                    best_score = exact_score
                    if best_score <= 4:
                        return sorted_set
        else:
            # Revert? In SA, we don't revert, we just don't move.
            # But we generated 'neighbor'. If we don't accept, current stays same.
            pass
        
        temp *= cool_rate
        if temp < min_temp:
            temp = min_temp
            # Restart? Or just keep going.
            # Maybe reheat if stuck?
            # Simple reheat: if no improvement in last 1000 steps, temp = 5.0
            # We don't track steps easily here without a counter.
            
        steps += 1
        if steps % 1000 == 0:
             # Occasional reheat
             if temp < 1.0:
                 temp = 2.0

    return sorted(best_set)
