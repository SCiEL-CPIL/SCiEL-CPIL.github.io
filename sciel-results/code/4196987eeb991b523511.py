import random
import time

def construct():
    start_time = time.time()
    # 150s limit, keep 5s margin
    deadline = start_time + 145.0
    
    fixed = {1, 2, 5, 11}
    max_val = 284
    
    def compute_diffs(a_set):
        diffs = [0] * (max_val)
        a_list = sorted(a_set)
        for i in range(len(a_list)):
            ai = a_list[i]
            for j in range(i + 1, len(a_list)):
                d = a_list[j] - ai
                diffs[d] += 1
        return diffs

    def calc_energy(diffs):
        e = 0
        for d in diffs:
            if d > 1:
                e += (d - 1)
        return e

    def get_score(a_set, diffs):
        size = len(a_set)
        energy = calc_energy(diffs)
        # Prioritize size, then validity
        return size * 1000 - energy

    def is_valid(a_set):
        diffs = compute_diffs(a_set)
        return calc_energy(diffs) == 0

    def greedy_init():
        a_set = set(fixed)
        diffs = compute_diffs(a_set)
        
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in a_set:
                continue
            # Check conflicts with existing elements
            ok = True
            for y in a_set:
                d = abs(x - y)
                if diffs[d] >= 1:
                    ok = False
                    break
            if ok:
                a_set.add(x)
                for y in a_set:
                    if y != x:
                        d = abs(x - y)
                        diffs[d] += 1
                if len(a_set) >= 20:
                    break
        return a_set

    def simulated_annealing(initial_set, time_limit):
        current_set = set(initial_set)
        best_set = set(current_set)
        best_score = -1
        
        current_diffs = compute_diffs(current_set)
        current_score = get_score(current_set, current_diffs)
        best_score = current_score
        
        T = 50.0
        cooling = 0.99995
        min_T = 1e-5
        
        while time.time() < time_limit and T > min_T:
            # Move
            new_set = set(current_set)
            r = random.random()
            
            if r < 0.4:
                # Add
                x = random.randint(1, max_val)
                if x not in new_set:
                    new_set.add(x)
            elif r < 0.7:
                # Remove
                removable = [x for x in new_set if x not in fixed]
                if removable:
                    x = random.choice(removable)
                    new_set.remove(x)
            else:
                # Swap
                removable = [x for x in new_set if x not in fixed]
                if removable:
                    x = random.choice(removable)
                    y = random.randint(1, max_val)
                    if y not in new_set:
                        new_set.remove(x)
                        new_set.add(y)
            
            new_diffs = compute_diffs(new_set)
            new_score = get_score(new_set, new_diffs)
            
            delta = new_score - current_score
            if delta > 0 or (T > 0 and random.random() < pow(2.71828, delta / T)):
                current_set = new_set
                current_diffs = new_diffs
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
                    # Early exit if we found a perfect 20
                    if len(best_set) == 20 and calc_energy(best_diffs) == 0:
                         return best_set
            
            T *= cooling
            
        return best_set

    best_overall = set(fixed)
    
    # Run initial greedy to get a baseline
    init_set = greedy_init()
    if len(init_set) > len(best_overall):
        best_overall = init_set
    if is_valid(best_overall) and len(best_overall) == 20:
        return best_overall

    # Run SA for the remaining time
    # We can run one long SA or multiple short ones.
    # Multiple short ones with different inits might be better for exploration.
    
    while time.time() < deadline:
        # New greedy init
        init_set = greedy_init()
        
        # Check if this init is already good
        if is_valid(init_set) and len(init_set) > len(best_overall):
            best_overall = init_set
            if len(best_overall) == 20:
                return best_overall

        # Run SA for a chunk of time
        chunk_time = min(5.0, deadline - time.time() - 1.0)
        if chunk_time <= 0:
            break
            
        result_set = simulated_annealing(init_set, time.time() + chunk_time)
        
        if len(result_set) > len(best_overall):
            best_overall = result_set
            if is_valid(best_overall) and len(best_overall) == 20:
                return best_overall

    return best_overall
