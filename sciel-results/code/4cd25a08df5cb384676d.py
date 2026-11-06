import random
import math
import time

def construct():
    # Generate 5-smooth numbers (Hamming numbers) up to 10000
    smooth_numbers = []
    i2 = i3 = i5 = 0
    next_2, next_3, next_5 = 2, 3, 5
    smooth_numbers.append(1)
    while True:
        min_val = min(next_2, next_3, next_5)
        if min_val > 10000:
            break
        smooth_numbers.append(min_val)
        if min_val == next_2:
            i2 += 1
            next_2 = smooth_numbers[i2] * 2
        if min_val == next_3:
            i3 += 1
            next_3 = smooth_numbers[i3] * 3
        if min_val == next_5:
            i5 += 1
            next_5 = smooth_numbers[i5] * 5
    
    # Candidate pool for replacements
    # Mix of small integers and smooth numbers
    candidate_pool = list(set(list(range(1, 150)) + smooth_numbers[:50]))
    random.shuffle(candidate_pool)
    
    def evaluate(A):
        A_set = set(A)
        sum_set = set()
        prod_set = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                sum_set.add(A[i] + A[j])
                prod_set.add(A[i] * A[j])
        return max(len(sum_set), len(prod_set))

    def hill_climb(initial_set, time_limit=10.0):
        current_A = sorted(initial_set)
        current_score = evaluate(current_A)
        best_A = current_A[:]
        best_score = current_score
        
        start_time = time.time()
        iterations = 0
        stuck_counter = 0
        
        while time.time() - start_time < time_limit:
            # Pick a random index to replace
            idx = random.randint(0, 19)
            old_val = current_A[idx]
            
            # Pick a new value from candidate pool
            # Bias towards values close to old_val or small values?
            # For now, random choice from pool
            new_val = random.choice(candidate_pool)
            
            if new_val in current_A and new_val != old_val:
                continue
            
            # Create new set
            new_A = current_A[:]
            new_A[idx] = new_val
            new_A.sort()
            
            new_score = evaluate(new_A)
            
            if new_score < current_score:
                current_A = new_A
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A[:]
                stuck_counter = 0
            else:
                stuck_counter += 1
                # If stuck too long, try a more aggressive move or reset
                if stuck_counter > 200:
                    # Try a 2-swap: pick two indices i, j and swap values with two new values?
                    # Or just restart this hill climb from best_A with a different random seed?
                    # Let's try swapping two elements in current_A with two new values from pool
                    if random.random() < 0.5:
                        i1 = random.randint(0, 19)
                        i2 = random.randint(0, 19)
                        if i1 != i2:
                            v1 = random.choice(candidate_pool)
                            v2 = random.choice(candidate_pool)
                            # Ensure uniqueness
                            if v1 not in current_A or v1 == current_A[i1]:
                                if v2 not in current_A or v2 == current_A[i2]:
                                    # Check if v1 == v2 and i1 != i2, then invalid
                                    if v1 != v2 or i1 == i2: # This logic is messy, let's keep it simple
                                        pass
                            # Simpler: just do single swaps but accept worse moves occasionally?
                            # No, let's just reset to best_A and continue
                            current_A = best_A[:]
                            current_score = best_score
                            stuck_counter = 0
            iterations += 1
            
        return best_A, best_score

    # Generate diverse initial sets
    init_sets = []
    
    # 1. Smallest smooth numbers
    init_sets.append(smooth_numbers[:20])
    
    # 2. Small consecutive integers
    init_sets.append(list(range(1, 21)))
    
    # 3. Geometric progression (base 1.2)
    gp = []
    val = 1
    while len(gp) < 20 and val <= 10000:
        gp.append(int(val))
        val *= 1.2
    # Ensure distinct
    gp = list(dict.fromkeys(gp))
    if len(gp) < 20:
        # Fill with small random numbers
        while len(gp) < 20:
            gp.append(random.randint(1, 50))
    init_sets.append(gp[:20])
    
    # 4. Random small numbers
    init_sets.append(random.sample(range(1, 100), 20))
    
    # 5. Powers of 2 and 3 mixed
    p23 = []
    for i in range(10):
        p23.append(2**i)
    for i in range(10):
        p23.append(3**i)
    p23 = list(set(p23))
    p23.sort()
    if len(p23) >= 20:
        init_sets.append(p23[:20])
    else:
        while len(p23) < 20:
            p23.append(random.randint(1, 100))
        init_sets.append(p23)

    best_overall_A = None
    best_overall_score = float('inf')
    
    total_time_limit = 50.0
    start_total = time.time()
    
    for i, init in enumerate(init_sets):
        remaining_time = total_time_limit - (time.time() - start_total)
        if remaining_time <= 0:
            break
        # Allocate time per restart
        time_per_restart = remaining_time / (len(init_sets) - i)
        if time_per_restart < 1.0:
            time_per_restart = 1.0
            
        A, score = hill_climb(init, time_limit=time_per_restart)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_A = A

    return best_overall_A
