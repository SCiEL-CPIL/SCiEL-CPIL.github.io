<<<<<<< SEARCH
    # Verify that this is a valid Steiner system S(5,6,12)
    used_5subs = set()
    valid = True
    for b in blocks:
        for s in combinations(b, 5):
            if s in used_5subs:
                valid = False
                break
            used_5subs.add(s)
        if not valid:
            break
    
    if valid and len(blocks) == 132:
        return blocks
    else:
        # Fall back to the ILP solver
        start_time = time.time()
        deadline = start_time + 110.0
        
        points = range(12)
        all_blocks = list(combinations(points, 6))
        n_blocks = len(all_blocks)
        
        # Precompute 5-subsets for each block
        block_5subs = [list(combinations(b, 5)) for b in all_blocks]
        
        # Map 5-subset to a unique ID
        all_5subs = set()
        for subs in block_5subs:
            for s in subs:
                all_5subs.add(s)
        all_5subs = list(all_5subs)
        sub_to_id = {s: i for i, s in enumerate(all_5subs)}
        n_5subs = len(all_5subs)
        
        # Build the incidence matrix A
        A = np.zeros((n_5subs, n_blocks), dtype=np.float64)
        
        for i, b in enumerate(all_blocks):
            for s in block_5subs[i]:
                j = sub_to_id[s]
                A[j, i] = 1.0
                
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c = -np.ones(n_blocks)
        
        # Define bounds
        bounds = Bounds(lb=np.zeros(n_blocks), ub=np.ones(n_blocks))
        
        # Define linear constraint
        lincons = LinearConstraint(A, lb=np.zeros(n_5subs), ub=np.ones(n_5subs))
        
        # Integrality: all variables are integer
        integrality = np.ones(n_blocks)
        
        try:
            res = milp(
                c=c,
                constraints=[lincons],
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": 100.0}
            )
            
            if res.status == 0:
                x = res.x
                selected_indices = [i for i in range(n_blocks) if x[i] > 0.5]
                selected_blocks = [list(all_blocks[i]) for i in selected_indices]
                
                # Verify validity
                used_5subs = set()
                valid = True
                for b in selected_blocks:
                    for s in combinations(b, 5):
                        if s in used_5subs:
                            valid = False
                            break
                        used_5subs.add(s)
                    if not valid:
                        break
                
                if valid:
                    return selected_blocks
        except Exception:
            pass
        
        # Fallback heuristic
        import random
        rng = random.Random(42)
        best_blocks = []
        best_count = 0
        
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            shuffled_indices = list(range(n_blocks))
            rng.shuffle(shuffled_indices)
            
            used_5subs = set()
            current_blocks = []
            
            for idx in shuffled_indices:
                b = all_blocks[idx]
                conflict = False
                for s in block_5subs[idx]:
                    if s in used_5subs:
                        conflict = True
                        break
                        
                if not conflict:
                    current_blocks.append(b)
                    for s in block_5subs[idx]:
                        used_5subs.add(s)
                        
                    if len(current_blocks) == 132:
                        return [list(b) for b in current_blocks]
            
            if len(current_blocks) > best_count:
                best_count = len(current_blocks)
                best_blocks = [list(b) for b in current_blocks]
                
        return best_blocks
=======
    return blocks
>>>>>>> REPLACE
