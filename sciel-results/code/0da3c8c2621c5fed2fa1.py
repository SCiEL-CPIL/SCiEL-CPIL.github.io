<<<<<<< SEARCH
def hill_climb(initial_set, deadline, max_iter=5000):
    """Local search via element replacement"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    best_score = compute_score(U)
    best_set = U[:]
    
    max_val = max(U)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(U)
        
        # Try replacing each element with a random value
        for i in range(n):
            if time.time() > deadline:
                break
            # Try several random replacements
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_val = random.randint(0, max_val * 2)
                if new_val in U and new_val != U[i]:
                    continue
                new_U = U[:i] + [new_val] + U[i+1:]
                new_U = sorted(set(new_U))
                if 0 not in new_U:
                    new_U = sorted([0] + [x for x in new_U if x > 0])
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(U)
                    improved = True
                    break
        
        if not improved:
            # Try adding a new element
            if len(U) < 50:
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    new_val = random.randint(1, max_val * 3)
                    new_U = sorted(U + [new_val])
                    s = compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        max_val = max(U)
                        improved = True
                        break
            
            if not improved:
                # Try removing an element
                if len(U) > 3:
                    for i in range(len(U)):
                        if time.time() > deadline:
                            break
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = compute_score(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = new_U
                            U = new_U
                            max_val = max(U)
                            improved = True
                            break
            
            if not improved:
                break
    
    return best_score, best_set
=======
def hill_climb(initial_set, deadline, max_iter=5000):
    """Local search via element replacement with smarter moves"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    best_score = compute_score(U)
    best_set = U[:]
    
    max_val = max(U)
    stagnant = 0
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(U)
        
        # Try replacing each element with smart candidates
        for i in range(n):
            if time.time() > deadline:
                break
            # Generate diverse replacement candidates
            candidates = set()
            # Random in range
            for _ in range(5):
                candidates.add(random.randint(0, max_val * 2))
            # Nearby values
            candidates.add(max(0, U[i] - 1))
            candidates.add(U[i] + 1)
            candidates.add(max(0, U[i] // 2))
            candidates.add(U[i] * 2)
            # Values that might improve structure
            if i > 0 and i < n - 1:
                candidates.add((U[i-1] + U[i+1]) // 2)
            
            for new_val in candidates:
                if time.time() > deadline:
                    break
                if new_val in U and new_val != U[i]:
                    continue
                new_U = sorted(set(U[:i] + [new_val] + U[i+1:]))
                if 0 not in new_U:
                    new_U = sorted([0] + [x for x in new_U if x > 0])
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(U)
                    improved = True
                    stagnant = 0
                    break
        
        if not improved:
            stagnant += 1
            
            # Try adding a new element
            if len(U) < 60 and stagnant < 10:
                for _ in range(8):
                    if time.time() > deadline:
                        break
                    # Smart addition: try values that extend the pattern
                    if random.random() < 0.3:
                        new_val = max_val + random.randint(1, max(1, max_val // 5))
                    elif random.random() < 0.6:
                        new_val = random.randint(1, max_val * 3)
                    else:
                        new_val = random.randint(1, max_val)
                    if new_val in U:
                        continue
                    new_U = sorted(U + [new_val])
                    s = compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        max_val = max(U)
                        improved = True
                        stagnant = 0
                        break
            
            if not improved:
                # Try removing an element
                if len(U) > 3 and stagnant < 10:
                    # Remove random non-zero element
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        idx = random.choice([i for i in range(len(U)) if U[i] != 0])
                        new_U = U[:idx] + U[idx+1:]
                        s = compute_score(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = new_U
                            U = new_U
                            max_val = max(U)
                            improved = True
                            stagnant = 0
                            break
            
            if not improved:
                # Try a bigger jump: replace 2 elements
                if len(U) > 4 and stagnant >= 5:
                    for _ in range(3):
                        if time.time() > deadline:
                            break
                        i, j = random.sample(range(1, len(U)), 2)
                        new_i = random.randint(0, max_val * 2)
                        new_j = random.randint(0, max_val * 2)
                        new_U = sorted(set(U[:i] + [new_i] + U[i+1:j] + [new_j] + U[j+1:]))
                        if 0 not in new_U:
                            new_U = sorted([0] + [x for x in new_U if x > 0])
                        if len(new_U) < 2:
                            continue
                        s = compute_score(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = new_U
                            U = new_U
                            max_val = max(U)
                            improved = True
                            stagnant = 0
                            break
            
            if not improved and stagnant > 30:
                break
    
    return best_score, best_set
>>>>>>> REPLACE

