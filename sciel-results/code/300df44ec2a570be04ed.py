import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    N = 1000
    max_sum = 2 * N  # 2000
    
    def greedy_forward(counts, chosen):
        """Continue greedy from current state. Returns (chosen, counts)."""
        for x in range(chosen[-1] + 1, N + 1) if chosen else range(1, N + 1):
            # Check if adding x keeps all sums <= 2
            ok = True
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if counts[s] >= 2:
                    ok = False
            if ok:
                # Update counts
                for y in chosen:
                    s = x + y
                    counts[s] += 1
                s = 2 * x
                counts[s] += 1
                chosen.append(x)
        return chosen, counts
    
    def greedy_with_skip(counts, chosen, skip_prob=0.1):
        """Greedy with occasional skips to explore different paths."""
        for x in range(chosen[-1] + 1, N + 1) if chosen else range(1, N + 1):
            # Check if adding x keeps all sums <= 2
            ok = True
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if counts[s] >= 2:
                    ok = False
            if ok and random.random() > skip_prob:
                for y in chosen:
                    s = x + y
                    counts[s] += 1
                s = 2 * x
                counts[s] += 1
                chosen.append(x)
        return chosen, counts
    
    def greedy_with_cost(counts, chosen, seed=0):
        """Greedy that picks the candidate with minimum 'cost' (sums that would become tight)."""
        random.seed(seed)
        candidates = list(range(chosen[-1] + 1, N + 1)) if chosen else list(range(1, N + 1))
        
        while candidates:
            best_x = None
            min_cost_found = float('inf')
            best_candidates = []
            
            for x in candidates:
                if counts[2 * x] >= 2:
                    continue
                valid = True
                cost = 0
                if counts[2 * x] == 1:
                    cost += 1
                for y in chosen:
                    s = x + y
                    if counts[s] >= 2:
                        valid = False
                        break
                    if counts[s] == 1:
                        cost += 1
                if not valid:
                    continue
                
                if cost < min_cost_found:
                    min_cost_found = cost
                    best_candidates = [x]
                elif cost == min_cost_found:
                    best_candidates.append(x)
            
            if not best_candidates:
                break
            
            best_x = random.choice(best_candidates)
            chosen.append(best_x)
            counts[2 * best_x] += 1
            for y in chosen[:-1]:
                s = best_x + y
                counts[s] += 1
            candidates.remove(best_x)
            
        return chosen, counts
    
    def local_search(chosen, counts, max_steps=500):
        """Try to improve by swapping elements."""
        chosen_set = set(chosen)
        best_size = len(chosen)
        
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            
            # Pick a random element to remove
            i = random.randint(0, len(chosen) - 1)
            a = chosen[i]
            
            # Remove a
            for y in chosen:
                if y == a:
                    counts[2 * a] -= 1
                else:
                    s = a + y
                    counts[s] -= 1
            
            chosen.pop(i)
            chosen_set.discard(a)
            
            # Try to find a replacement b that is valid and allows adding a new element c
            valid_replacements = []
            for b in range(1, N + 1):
                if b in chosen_set:
                    continue
                if counts[2 * b] >= 2:
                    continue
                valid = True
                for y in chosen:
                    s = b + y
                    if counts[s] >= 2:
                        valid = False
                        break
                if valid:
                    valid_replacements.append(b)
            
            if not valid_replacements:
                # Put a back
                chosen.insert(i, a)
                chosen_set.add(a)
                for y in chosen:
                    if y == a:
                        counts[2 * a] += 1
                    else:
                        s = a + y
                        counts[s] += 1
                continue
            
            b = random.choice(valid_replacements)
            
            # Add b
            chosen.append(b)
            chosen_set.add(b)
            counts[2 * b] += 1
            for y in chosen[:-1]:
                s = b + y
                counts[s] += 1
            
            # Now, try to add a new element c
            added_new = False
            for c in range(1, N + 1):
                if c in chosen_set:
                    continue
                if counts[2 * c] >= 2:
                    continue
                valid = True
                for y in chosen:
                    s = c + y
                    if counts[s] >= 2:
                        valid = False
                        break
                if valid:
                    # Add c
                    chosen.append(c)
                    chosen_set.add(c)
                    counts[2 * c] += 1
                    for y in chosen[:-1]:
                        s = c + y
                        counts[s] += 1
                    added_new = True
                    break
            
            if added_new:
                best_size = len(chosen)
                improved = True
        
        return chosen, counts, best_size
    
    # Phase 1: Basic greedy
    chosen, counts = greedy_forward([0] * (max_sum + 1), [])
    best_chosen = list(chosen)
    best_size = len(chosen)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 2: Multiple greedy restarts with different skip probabilities
    for trial in range(50):
        if time.time() > deadline:
            break
        random.seed(trial * 7 + 13)
        chosen2, counts2 = greedy_with_skip([0] * (max_sum + 1), [], skip_prob=0.05 + 0.01 * trial)
        if len(chosen2) > best_size:
            best_size = len(chosen2)
            best_chosen = list(chosen2)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 3: Greedy with cost-based selection
    for trial in range(30):
        if time.time() > deadline:
            break
        random.seed(trial * 11 + 17)
        chosen3, counts3 = greedy_with_cost([0] * (max_sum + 1), [], seed=trial)
        if len(chosen3) > best_size:
            best_size = len(chosen3)
            best_chosen = list(chosen3)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 4: Local search on best solution
    chosen_ls = list(best_chosen)
    counts_ls = [0] * (max_sum + 1)
    for i, a in enumerate(chosen_ls):
        for j in range(i, len(chosen_ls)):
            s = chosen_ls[i] + chosen_ls[j]
            counts_ls[s] += 1
    
    chosen_ls, counts_ls, best_size_ls = local_search(chosen_ls, counts_ls, max_steps=1000)
    if best_size_ls > best_size:
        best_size = best_size_ls
        best_chosen = list(chosen_ls)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 5: More greedy restarts with varying parameters
    for trial in range(30):
        if time.time() > deadline:
            break
        random.seed(1000 + trial * 3 + 5)
        skip_prob = 0.02 + 0.02 * (trial % 10)
        chosen5, counts5 = greedy_with_skip([0] * (max_sum + 1), [], skip_prob=skip_prob)
        if len(chosen5) > best_size:
            best_size = len(chosen5)
            best_chosen = list(chosen5)
    
    # Final local search if there's time
    if time.time() < deadline - 10:
        chosen_final = list(best_chosen)
        counts_final = [0] * (max_sum + 1)
        for i, a in enumerate(chosen_final):
            for j in range(i, len(chosen_final)):
                s = chosen_final[i] + chosen_final[j]
                counts_final[s] += 1
        
        chosen_final, counts_final, best_size_final = local_search(chosen_final, counts_final, max_steps=2000)
        if best_size_final > best_size:
            best_size = best_size_final
            best_chosen = list(chosen_final)
    
    return best_chosen
