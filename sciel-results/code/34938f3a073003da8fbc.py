<<<<<<< SEARCH
    # Main Optimization Loop
    temperature = 10.0
    cooling = 0.95
    min_temp = 0.01
    
    # Initial reduction
    current_blocks, cover_counts = reduce_redundant(current_blocks, cover_counts)
    if len(current_blocks) < best_size:
        best_blocks = list(current_blocks)
        best_size = len(best_blocks)
        
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # 1. Reduce redundant blocks
        current_blocks, cover_counts = reduce_redundant(current_blocks, cover_counts)
        
        # Update best
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(best_blocks)
        
        # 2. Perturb - choose strategy
        r = random.random()
        
        if r < 0.35 and len(current_blocks) > 1:
            # Strategy A: Remove 1-2 random blocks
            num_removals = random.randint(1, 2)
            if len(current_blocks) <= num_removals:
                break
            removals = random.sample(current_blocks, num_removals)
            for idx in removals:
                cover_counts -= Cover[idx]
            current_blocks = [b for b in current_blocks if b not in removals]
            
        elif r < 0.70 and len(current_blocks) > 1:
            # Strategy B: Try block replacement
            current_blocks, cover_counts, success = try_replace_one(
                current_blocks, cover_counts, max_candidates=100
            )
            if not success:
                # Fallback: remove one random block
                if len(current_blocks) > 1:
                    idx = random.choice(current_blocks)
                    cover_counts -= Cover[idx]
                    current_blocks = [b for b in current_blocks if b != idx]
            
        else:
            # Strategy C: Remove 2-4 random blocks (aggressive)
            num_removals = random.randint(2, min(4, max(1, len(current_blocks) - 1)))
            if len(current_blocks) <= num_removals:
                break
            removals = random.sample(current_blocks, num_removals)
            for idx in removals:
                cover_counts -= Cover[idx]
            current_blocks = [b for b in current_blocks if b not in removals]
        
        # 3. Repair
        current_blocks = repair(current_blocks, cover_counts)
        
        # 4. Acceptance
        new_size = len(current_blocks)
        delta = new_size - best_size
        
        if delta < 0:
            # Improvement
            best_blocks = list(current_blocks)
            best_size = new_size
        elif temperature > min_temp:
            # Simulated Annealing acceptance
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                pass
            else:
                # Reject: revert to best state
                current_blocks = list(best_blocks)
                cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                for idx in current_blocks:
                    cover_counts += Cover[idx]
                
        temperature *= cooling
=======
    # Main Optimization Loop
    temperature = 15.0
    cooling = 0.93
    min_temp = 0.05
    
    # Initial reduction
    current_blocks, cover_counts = reduce_redundant(current_blocks, cover_counts)
    if len(current_blocks) < best_size:
        best_blocks = list(current_blocks)
        best_size = len(best_blocks)
        
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # 1. Reduce redundant blocks
        current_blocks, cover_counts = reduce_redundant(current_blocks, cover_counts)
        
        # Update best
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(best_blocks)
        
        # 2. Perturb - choose strategy
        r = random.random()
        
        if r < 0.40 and len(current_blocks) > 1:
            # Strategy A: Remove 1-2 random blocks
            num_removals = random.randint(1, 2)
            if len(current_blocks) <= num_removals:
                break
            removals = random.sample(current_blocks, num_removals)
            for idx in removals:
                cover_counts -= Cover[idx]
            current_blocks = [b for b in current_blocks if b not in removals]
            
        elif r < 0.75 and len(current_blocks) > 1:
            # Strategy B: Try block replacement
            current_blocks, cover_counts, success = try_replace_one(
                current_blocks, cover_counts, max_candidates=150
            )
            if not success:
                # Fallback: remove one random block
                if len(current_blocks) > 1:
                    idx = random.choice(current_blocks)
                    cover_counts -= Cover[idx]
                    current_blocks = [b for b in current_blocks if b != idx]
            
        else:
            # Strategy C: Remove 2-3 random blocks (aggressive)
            num_removals = random.randint(2, min(3, max(1, len(current_blocks) - 1)))
            if len(current_blocks) <= num_removals:
                break
            removals = random.sample(current_blocks, num_removals)
            for idx in removals:
                cover_counts -= Cover[idx]
            current_blocks = [b for b in current_blocks if b not in removals]
        
        # 3. Repair
        current_blocks = repair(current_blocks, cover_counts)
        
        # 4. Acceptance
        new_size = len(current_blocks)
        delta = new_size - best_size
        
        if delta < 0:
            # Improvement
            best_blocks = list(current_blocks)
            best_size = new_size
        elif temperature > min_temp:
            # Simulated Annealing acceptance
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                pass
            else:
                # Reject: revert to best state
                current_blocks = list(best_blocks)
                cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                for idx in current_blocks:
                    cover_counts += Cover[idx]
                
        temperature *= cooling
>>>>>>> REPLACE

