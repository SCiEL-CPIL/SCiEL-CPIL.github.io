<<<<<<< SEARCH
    # Simulated annealing with various initializations
    random.seed(42)
    
    def simulate_anneal(initial, max_iters=5000, t0=1.0, cooling=0.999):
        """Simulated annealing to optimize the set."""
        current = list(initial)
        current_score = get_score(current)
        best_local = list(current)
        best_local_score = current_score
        
        t = t0
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Choose mutation
            mutation_type = random.random()
            
            if mutation_type < 0.3 and len(current) < 20:
                # Add an element
                max_cur = max(current)
                # Try adding near existing or slightly beyond
                candidates = set()
                for el in current:
                    for delta in range(-3, 4):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates.add(new_el)
                for new_el in range(max_cur + 1, max_cur + 8):
                    candidates.add(new_el)
                # Also try some random larger values
                for _ in range(5):
                    new_el = random.randint(1, max_cur + 20)
                    if new_el not in current:
                        candidates.add(new_el)
                
                if candidates:
                    new_el = random.choice(sorted(candidates))
                    new_set = sorted(current + [new_el])
            elif mutation_type < 0.5 and len(current) > 3:
                # Remove an element (not 0)
                removable = [i for i in range(len(current)) if current[i] != 0]
                if removable:
                    idx = random.choice(removable)
                    new_set = current[:idx] + current[idx+1:]
                    new_set = sorted(set(new_set))
                else:
                    continue
            elif mutation_type < 0.7:
                # Replace an element
                if len(current) < 3:
                    continue
                replaceable = [i for i in range(len(current)) if current[i] != 0]
                if not replaceable:
                    continue
                idx = random.choice(replaceable)
                old_el = current[idx]
                max_cur = max(current)
                # Try nearby or random replacement
                if random.random() < 0.5:
                    new_el = random.randint(max(1, old_el - 5), min(max_cur + 10, old_el + 6))
                else:
                    new_el = random.randint(1, max_cur + 15)
                
                if new_el == old_el or new_el in current:
                    continue
                
                new_set = list(current)
                new_set[idx] = new_el
                new_set = sorted(set(new_set))
                if 0 not in new_set or len(new_set) != len(current):
                    continue
            else:
                # Swap two elements (neither is 0)
                if len(current) < 4:
                    continue
                non_zero = [i for i in range(len(current)) if current[i] != 0]
                if len(non_zero) < 2:
                    continue
                i, j = random.sample(non_zero, 2)
                # Generate new values
                max_cur = max(current)
                new_i = random.randint(1, max_cur + 10)
                new_j = random.randint(1, max_cur + 10)
                
                if new_i in current or new_j in current or new_i == new_j:
                    continue
                
                new_set = list(current)
                new_set[i] = new_i
                new_set[j] = new_j
                new_set = sorted(set(new_set))
                if 0 not in new_set or len(new_set) != len(current):
                    continue
            
            new_score = get_score(new_set)
            
            # Accept or reject
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if new_score > best_local_score:
                    best_local = list(new_set)
                    best_local_score = new_score
            else:
                # Accept with probability based on temperature
                delta = current_score - new_score
                if delta < 0:
                    accept_prob = 1.0
                else:
                    if t > 1e-10:
                        accept_prob = math.exp(-delta / t)
                    else:
                        accept_prob = 0.0
                
                if random.random() < accept_prob:
                    current = new_set
                    current_score = new_score
            
            t *= cooling
        
        return best_local, best_local_score
=======
    # Simulated annealing with various initializations
    random.seed(42)
    
    def simulate_anneal(initial, max_iters=5000, t0=1.0, cooling=0.999):
        """Simulated annealing to optimize the set."""
        current = list(initial)
        current_score = get_score(current)
        best_local = list(current)
        best_local_score = current_score
        
        t = t0
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Choose mutation
            mutation_type = random.random()
            
            if mutation_type < 0.35 and len(current) < 30:
                # Add an element - try both local and global additions
                max_cur = max(current)
                candidates = set()
                # Local additions near existing elements
                for el in current:
                    for delta in range(-3, 4):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates.add(new_el)
                # Additions just beyond max
                for new_el in range(max_cur + 1, max_cur + 10):
                    candidates.add(new_el)
                # Larger jumps to explore sparser configurations
                for _ in range(8):
                    jump = random.randint(max_cur // 3, max_cur + 30)
                    new_el = random.randint(1, max_cur + jump)
                    if new_el not in current:
                        candidates.add(new_el)
                # Try doubling-like growth
                for el in current[-3:]:
                    new_el = el * 2 + random.randint(-2, 3)
                    if new_el > 0 and new_el not in current:
                        candidates.add(new_el)
                
                if candidates:
                    new_el = random.choice(sorted(candidates))
                    new_set = sorted(current + [new_el])
            elif mutation_type < 0.55 and len(current) > 3:
                # Remove an element (not 0)
                removable = [i for i in range(len(current)) if current[i] != 0]
                if removable:
                    idx = random.choice(removable)
                    new_set = current[:idx] + current[idx+1:]
                    new_set = sorted(set(new_set))
                else:
                    continue
            elif mutation_type < 0.75:
                # Replace an element
                if len(current) < 3:
                    continue
                replaceable = [i for i in range(len(current)) if current[i] != 0]
                if not replaceable:
                    continue
                idx = random.choice(replaceable)
                old_el = current[idx]
                max_cur = max(current)
                # Try nearby or random replacement
                r = random.random()
                if r < 0.3:
                    new_el = random.randint(max(1, old_el - 5), min(max_cur + 10, old_el + 6))
                elif r < 0.6:
                    new_el = random.randint(1, max_cur + 30)
                else:
                    # Try to maintain Golomb-like property: pick value that doesn't create
                    # duplicate differences with existing elements
                    max_cur2 = max(current)
                    new_el = random.randint(1, max_cur2 + 40)
                
                if new_el == old_el or new_el in current:
                    continue
                
                new_set = list(current)
                new_set[idx] = new_el
                new_set = sorted(set(new_set))
                if 0 not in new_set or len(new_set) != len(current):
                    continue
            else:
                # Perturb multiple elements at once
                if len(current) < 4:
                    continue
                non_zero = [i for i in range(len(current)) if current[i] != 0]
                if len(non_zero) < 2:
                    continue
                k = min(2, len(non_zero))
                indices = random.sample(non_zero, k)
                max_cur = max(current)
                
                new_set = list(current)
                valid = True
                for idx in indices:
                    new_el = random.randint(1, max_cur + 25)
                    if new_el in current and current.index(new_el) not in indices:
                        valid = False
                        break
                    new_set[idx] = new_el
                
                if not valid:
                    continue
                
                new_set = sorted(set(new_set))
                if 0 not in new_set or len(new_set) != len(current):
                    continue
            
            new_score = get_score(new_set)
            
            # Accept or reject
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if new_score > best_local_score:
                    best_local = list(new_set)
                    best_local_score = new_score
            else:
                # Accept with probability based on temperature
                delta = current_score - new_score
                if delta < 0:
                    accept_prob = 1.0
                else:
                    if t > 1e-10:
                        accept_prob = math.exp(-delta / t)
                    else:
                        accept_prob = 0.0
                
                if random.random() < accept_prob:
                    current = new_set
                    current_score = new_score
            
            t *= cooling
        
        return best_local, best_local_score
>>>>>>> REPLACE

