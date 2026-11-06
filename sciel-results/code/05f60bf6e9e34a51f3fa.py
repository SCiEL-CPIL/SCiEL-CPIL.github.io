import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    def get_score(S):
        S_set = S
        S_sorted = sorted(S)
        d_set = set()
        n = len(S_sorted)
        contains = S_set.__contains__
        for i in range(n):
            a = S_sorted[i]
            for j in range(i + 1, n):
                b = S_sorted[j]
                d = b - a
                c = b + d
                if contains(c):
                    d_set.add(d)
        return len(d_set)

    def generate_initial_sets():
        sets = []
        # 1. Dense start
        sets.append(set(range(1, 33)))
        
        # 2. Two dense blocks
        s2 = set(range(1, 17)) | set(range(200, 216))
        sets.append(s2)
        
        # 3. Grid-like (i*16 + j)
        s3 = set()
        for i in range(1, 5):
            for j in range(1, 9):
                val = i * 16 + j
                if val <= 512:
                    s3.add(val)
        if len(s3) >= 32:
            sets.append(set(random.sample(sorted(s3), 32)))
            
        # 4. Random
        sets.append(set(random.sample(range(1, 513), 32)))
        
        # 5. Random
        sets.append(set(random.sample(range(1, 513), 32)))
        
        return sets

    best_set = None
    best_score = -1
    
    init_sets = generate_initial_sets()
    
    # Evaluate initial sets to pick a good starting point
    # Also run short SA on each? No, just pick the best initial one for the first restart.
    # Actually, let's just run the main loop with multiple restarts.
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() >= deadline - 2.0:
            break
            
        if r == 0:
            # Pick the best initial set
            current_set = max(init_sets, key=get_score)
        else:
            # Random restart, maybe perturbed from best?
            # Let's just do random for diversity, or slightly perturb best
            if best_set and random.random() < 0.5:
                current_set = set(best_set)
                # Perturb 2 elements
                for _ in range(2):
                    elem = random.choice(list(current_set))
                    new_val = random.randint(1, 512)
                    if new_val not in current_set:
                        current_set.remove(elem)
                        current_set.add(new_val)
            else:
                current_set = set(random.sample(range(1, 513), 32))
                
        current_score = get_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        # SA Parameters
        start_temp = 10.0
        end_temp = 0.01
        cooling_rate = 0.995
        
        restart_deadline = start_time + (r + 1) * time_per_restart
        temp = start_temp
        
        current_list = list(current_set)
        
        while time.time() < restart_deadline:
            # Generate neighbor
            move_type = random.random()
            
            if move_type < 0.5:
                # Random replacement
                elem_to_replace = random.choice(current_list)
                new_val = random.randint(1, 512)
                if new_val in current_set:
                    continue
                new_set = set(current_set)
                new_set.remove(elem_to_replace)
                new_set.add(new_val)
            elif move_type < 0.8:
                # Local nudge
                elem_to_replace = random.choice(current_list)
                delta = random.randint(-5, 5)
                if delta == 0:
                    continue
                new_val = elem_to_replace + delta
                if new_val < 1 or new_val > 512:
                    continue
                if new_val in current_set:
                    continue
                new_set = set(current_set)
                new_set.remove(elem_to_replace)
                new_set.add(new_val)
            else:
                # AP Completion Move
                # Pick two elements a, b from current_set
                # Try to complete a, a+d, a+2d or a-2d, a-d, a
                # d = b - a
                # Candidate 1: c = b + d = a + 2d
                # Candidate 2: c = a - d = b - 2d
                current_list_copy = list(current_set)
                if len(current_list_copy) < 2:
                    continue
                a, b = random.sample(current_list_copy, 2)
                if a > b:
                    a, b = b, a
                d = b - a
                candidates = []
                c1 = b + d
                if 1 <= c1 <= 512 and c1 not in current_set:
                    candidates.append(c1)
                c2 = a - d
                if 1 <= c2 <= 512 and c2 not in current_set:
                    candidates.append(c2)
                
                if not candidates:
                    continue
                
                new_val = random.choice(candidates)
                # Replace a random element with new_val
                elem_to_replace = random.choice(current_list)
                if elem_to_replace == new_val: # Should not happen as new_val not in set
                    continue
                new_set = set(current_set)
                new_set.remove(elem_to_replace)
                new_set.add(new_val)
            
            new_score = get_score(new_set)
            delta_score = new_score - current_score
            
            if delta_score >= 0:
                current_set = new_set
                current_score = new_score
                current_list = list(current_set)
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                if temp > 0:
                    prob = math.exp(delta_score / temp)
                    if random.random() < prob:
                        current_set = new_set
                        current_score = new_score
                        current_list = list(current_set)
            
            temp *= cooling_rate
            if temp < end_temp:
                temp = end_temp

    return sorted(best_set)
