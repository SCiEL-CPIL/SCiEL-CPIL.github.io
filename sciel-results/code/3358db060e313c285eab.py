<<<<<<< SEARCH
def local_search(start_set, deadline, max_iter=2000):
    current = sorted(start_set)
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    # Define moves
    def try_move(new_set):
        nonlocal current, current_score, best_set, best_score
        new_set = sorted(set(new_set))
        if 0 not in new_set:
            return False
        if len(new_set) < 2:
            return False
        new_score = score_set(new_set)
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
            return True
        return False

    iterations = 0
    while iterations < max_iter and time.time() < deadline - 5:
        iterations += 1
        moved = False
        
        # 1. Move one element
        if len(current) > 1:
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            # Try small shifts
            for delta in random.sample([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10], 3):
                new_val = old_val + delta
                if new_val > 0:
                    new_set = current[:]
                    new_set[i] = new_val
                    if try_move(new_set):
                        moved = True
                        break
        if moved: continue
        
        # 2. Add an element
        if len(current) < 20:
            max_val = current[-1]
            # Try adding near existing elements or extending
            candidates = []
            for x in current:
                candidates.extend([x+1, x-1, x*2, x+random.randint(1,5)])
            candidates.append(max_val + random.randint(1, 10))
            candidates.append(random.randint(1, max_val))
            
            for new_val in random.sample(candidates, min(5, len(candidates))):
                if new_val > 0 and new_val not in current:
                    new_set = current + [new_val]
                    if try_move(new_set):
                        moved = True
                        break
        if moved: continue
        
        # 3. Remove an element (if set is too large)
        if len(current) > 4:
            i = random.randint(1, len(current) - 1)
            new_set = [x for j, x in enumerate(current) if j != i]
            if try_move(new_set):
                moved = True
                continue

        # 4. Simulated Annealing step (allow small worsening)
        if random.random() < 0.1:
             i = random.randint(1, len(current) - 1)
             new_val = current[i] + random.randint(-3, 3)
             if new_val > 0:
                 new_set = current[:]
                 new_set[i] = new_val
                 new_set = sorted(set(new_set))
                 if 0 in new_set and len(new_set) > 1:
                     new_score = score_set(new_set)
                     # Simple acceptance
                     if new_score > current_score - 0.01: 
                         if try_move(new_set):
                             moved = True

    return best_set, best_score
=======
def local_search(start_set, deadline, max_iter=2000):
    current = sorted(start_set)
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    # Adaptive temperature for simulated annealing
    temperature = 0.05
    cooling_rate = 0.995
    
    # Define moves
    def try_move(new_set, allow_worse=False):
        nonlocal current, current_score, best_set, best_score, temperature
        new_set = sorted(set(new_set))
        if 0 not in new_set:
            return False
        if len(new_set) < 2:
            return False
        new_score = score_set(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
            return True
        elif allow_worse and new_score > current_score - temperature:
            # Accept worse moves with probability based on temperature
            if random.random() < temperature:
                current = new_set
                current_score = new_score
                return True
        return False

    iterations = 0
    no_improvement_count = 0
    max_no_improvement = 200
    
    while iterations < max_iter and time.time() < deadline - 5:
        iterations += 1
        moved = False
        improved = False
        
        # 1. Move one element with adaptive step size
        if len(current) > 1:
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            # Adaptive delta range based on set size and position
            max_delta = max(1, int(0.1 * current[-1]))
            delta_range = list(range(-max_delta, max_delta + 1))
            random.shuffle(delta_range)
            
            for delta in delta_range[:5]:
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val > 0:
                    new_set = current[:]
                    new_set[i] = new_val
                    if try_move(new_set, allow_worse=(no_improvement_count > 100)):
                        moved = True
                        if new_score > current_score:
                            improved = True
                        break
        if moved:
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            continue
        
        # 2. Add an element
        if len(current) < 25:
            max_val = current[-1]
            # Try adding near existing elements or extending
            candidates = []
            for x in current:
                candidates.extend([x+1, x-1, x*2, x+random.randint(1,5)])
            candidates.append(max_val + random.randint(1, 10))
            candidates.append(random.randint(1, max_val))
            # Add some structured candidates
            if max_val > 1:
                candidates.append(int(max_val * 1.5))
                candidates.append(int(max_val * 0.7))
            
            unique_candidates = list(set(c for c in candidates if c > 0 and c not in current))
            if unique_candidates:
                for new_val in random.sample(unique_candidates, min(5, len(unique_candidates))):
                    new_set = current + [new_val]
                    if try_move(new_set, allow_worse=(no_improvement_count > 100)):
                        moved = True
                        improved = True
                        break
        if moved:
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            continue
        
        # 3. Remove an element (if set is too large)
        if len(current) > 4:
            i = random.randint(1, len(current) - 1)
            new_set = [x for j, x in enumerate(current) if j != i]
            if try_move(new_set):
                moved = True
                no_improvement_count = 0
                continue
            else:
                no_improvement_count += 1

        # 4. Simulated Annealing step (allow small worsening)
        if random.random() < 0.15:
            i = random.randint(1, len(current) - 1)
            step = random.randint(1, max(1, int(0.05 * current[-1])))
            new_val = current[i] + random.choice([-1, 1]) * step
            if new_val > 0:
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                if 0 in new_set and len(new_set) > 1:
                    if try_move(new_set, allow_worse=True):
                        moved = True
                        no_improvement_count += 1
                        temperature *= cooling_rate
        
        if not moved:
            no_improvement_count += 1
            # Reset temperature if stuck
            if no_improvement_count > max_no_improvement:
                temperature = 0.05
                no_improvement_count = 0

    # Final polish: try very fine adjustments
    if time.time() < deadline - 3:
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            i = random.randint(1, len(best_set) - 1)
            for delta in [-1, 1]:
                new_val = best_set[i] + delta
                if new_val > 0:
                    new_set = best_set[:]
                    new_set[i] = new_val
                    new_set = sorted(set(new_set))
                    if 0 in new_set and len(new_set) > 1:
                        new_score = score_set(new_set)
                        if new_score > best_score:
                            best_set = new_set
                            best_score = new_score
                            break

    return best_set, best_score
>>>>>>> REPLACE

