"""Improved B2[2] construction using randomized local search."""
import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    # First, get a baseline greedy solution
    def greedy_greedy():
        chosen, counts = [], {}
        for x in range(1, 1001):
            new = [x + y for y in chosen] + [2 * x]
            trial = dict(counts)
            for s in new:
                trial[s] = trial.get(s, 0) + 1
            if all(trial[s] <= 2 for s in new):
                chosen.append(x)
                counts = trial
        return chosen
    
    baseline = greedy_greedy()
    best_set = set(baseline)
    best_size = len(best_set)
    
    # Build sum count map for the baseline
    def build_sum_counts(s):
        counts = {}
        lst = sorted(s)
        n = len(lst)
        for i in range(n):
            for j in range(i, n):
                sm = lst[i] + lst[j]
                counts[sm] = counts.get(sm, 0) + 1
        return counts
    
    best_counts = build_sum_counts(best_set)
    
    # Try randomized restarts with different orderings
    def try_randomized_greedy(order):
        chosen, counts = [], {}
        for x in order:
            if x in counts and False:  # placeholder
                pass
            new = [x + y for y in chosen] + [2 * x]
            trial = dict(counts)
            for s in new:
                trial[s] = trial.get(s, 0) + 1
            if all(trial[s] <= 2 for s in new):
                chosen.append(x)
                counts = trial
        return chosen
    
    # Try several random orderings
    for restart in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Create a random ordering with some bias
        order = list(range(1, 1001))
        random.shuffle(order)
        
        result = try_randomized_greedy(order)
        if len(result) > best_size:
            best_set = set(result)
            best_size = len(result)
            best_counts = build_sum_counts(best_set)
    
    # Local search: try to add elements or swap
    if time.time() - start_time < DEADLINE:
        # Try adding elements that fit
        added = True
        while added and time.time() - start_time < DEADLINE:
            added = False
            for x in range(1, 1001):
                if x in best_set:
                    continue
                # Check if x can be added
                lst = sorted(best_set)
                new_sums = [x + y for y in lst] + [2 * x]
                trial = dict(best_counts)
                ok = True
                for s in new_sums:
                    trial[s] = trial.get(s, 0) + 1
                    if trial[s] > 2:
                        ok = False
                        break
                if ok:
                    best_set.add(x)
                    best_counts = trial
                    best_size += 1
                    added = True
    
    # Try local swaps: remove one element, try to add two others
    if time.time() - start_time < DEADLINE:
        for _ in range(500):
            if time.time() - start_time > DEADLINE:
                break
            
            # Pick a random element to remove
            if not best_set:
                break
            remove_elem = random.choice(list(best_set))
            
            # Try adding two random elements
            candidates = [x for x in range(1, 1001) if x not in best_set]
            if len(candidates) < 2:
                continue
            
            a, b = random.sample(candidates, 2)
            
            # Check if we can swap remove_elem with a and b
            temp_set = best_set - {remove_elem}
            temp_set.add(a)
            temp_set.add(b)
            
            # Verify validity
            lst = sorted(temp_set)
            n = len(lst)
            counts = {}
            valid = True
            for i in range(n):
                for j in range(i, n):
                    sm = lst[i] + lst[j]
                    counts[sm] = counts.get(sm, 0) + 1
                    if counts[sm] > 2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_set = temp_set
                best_size = len(best_set)
                best_counts = build_sum_counts(best_set)
    
    return sorted(best_set)
