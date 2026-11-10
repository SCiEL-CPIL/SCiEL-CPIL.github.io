import itertools
import time
import random
import math

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
    
    def get_energy(s):
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return len(sums)
    
    def get_collision_energy(s):
        """Count of sum values that appear more than once. Higher = more collisions = better."""
        from collections import Counter
        sums = []
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.append(s[i] + s[j])
        counts = Counter(sums)
        # Number of "wasted" sums: total pairs - distinct sums
        return len(sums) - len(counts)
    
    random.seed(42)
    
    # Generate diverse initial candidates
    init_candidates = []
    # Dense clusters
    init_candidates.append([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23])
    init_candidates.append([0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32])
    init_candidates.append([0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20])
    init_candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    # More spread but with local clusters
    init_candidates.append([0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17])
    init_candidates.append([0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21])
    # Random dense
    for _ in range(10):
        base = random.randint(0, 20)
        cluster_size = random.randint(3, 6)
        vals = set()
        while len(vals) < 12:
            if random.random() < 0.7:
                vals.add(random.randint(base, base + 30))
            else:
                vals.add(random.randint(0, 128))
        init_candidates.append(list(vals))
    
    # Find best initial candidate
    best_set = init_candidates[0]
    best_score = max_sidon_size(sorted(best_set))
    best_energy = get_collision_energy(best_set)
    
    for cand in init_candidates[1:]:
        score = max_sidon_size(sorted(cand))
        energy = get_collision_energy(cand)
        if score < best_score or (score == best_score and energy > best_energy):
            best_score = score
            best_energy = energy
            best_set = cand
    
    if best_score <= 4:
        return sorted(best_set)
    
    current = best_set[:]
    current_energy = get_collision_energy(current)
    
    best_sa_energy = current_energy
    best_sa_set = current[:]
    
    temp = 10.0
    cool_rate = 0.9998
    min_temp = 0.5
    
    steps = 0
    no_improve = 0
    
    while time.time() < deadline:
        # Generate neighbor
        neighbor = current[:]
        
        r = random.random()
        if r < 0.4:
            # Replace one element with a value that creates collisions
            i = random.randint(0, 11)
            # Try to find a value that maximizes collisions with existing elements
            # Sample a few candidates and pick the best
            best_new_val = neighbor[i]
            best_new_energy = current_energy
            for _ in range(20):
                new_val = random.randint(0, 128)
                if new_val in neighbor:
                    continue
                old_val = neighbor[i]
                neighbor[i] = new_val
                e = get_collision_energy(neighbor)
                if e > best_new_energy:
                    best_new_energy = e
                    best_new_val = new_val
                neighbor[i] = old_val
            
            if best_new_val != neighbor[i]:
                neighbor[i] = best_new_val
            else:
                # Just random replacement
                new_val = random.randint(0, 128)
                while new_val in neighbor:
                    new_val = random.randint(0, 128)
                neighbor[i] = new_val
        elif r < 0.7:
            # Swap two elements
            i, j = random.sample(range(12), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Move one element to a random position
            i = random.randint(0, 11)
            old_val = neighbor[i]
            new_val = random.randint(0, 128)
            while new_val in neighbor:
                new_val = random.randint(0, 128)
            neighbor[i] = new_val
        
        new_energy = get_collision_energy(neighbor)
        delta = new_energy - current_energy
        
        # We want to maximize collision energy, so accept if delta > 0 or with prob exp(delta/temp)
        if delta > 0 or random.random() < math.exp(delta / temp):
            current = neighbor
            current_energy = new_energy
            
            if new_energy > best_sa_energy:
                best_sa_energy = new_energy
                best_sa_set = current[:]
                no_improve = 0
                
                # Check exact score periodically
                if steps % 100 == 0 or new_energy > best_energy:
                    sorted_set = sorted(best_sa_set)
                    exact_score = max_sidon_size(sorted_set)
                    if exact_score < best_score:
                        best_score = exact_score
                        best_set = best_sa_set[:]
                        if best_score <= 4:
                            return sorted_set
        else:
            no_improve += 1
        
        temp *= cool_rate
        if temp < min_temp:
            temp = min_temp
        
        # Reheat if stuck
        if no_improve > 500:
            temp = 5.0
            no_improve = 0
        
        steps += 1
    
    return sorted(best_set)
