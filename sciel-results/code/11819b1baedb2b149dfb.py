<<<<<<< SEARCH
    start = time.time()
    deadline = start + 135  # Safety margin before 150s kill
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Systematic exploration of superincreasing-like sequences
    # Powers of 2
    for n in range(2, 25):
        if time.time() > deadline:
            return best_set
        U = [0] + [2**i for i in range(n-1)]
        if max(U) > 100000:
            break
        s = fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Other bases
    for base in range(2, 10):
        if time.time() > deadline:
            return best_set
        for n in range(2, 20):
            U = [0] + [base**i for i in range(n-1)]
            if max(U) > 200000:
                break
            s = fast_score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Phase 2: Random search with guided generation
    # Strategy: generate sets that are "spread out" to maximize differences
    for n in range(3, 18):
        for M in [10, 20, 50, 100, 200, 500, 1000, 2000]:
            if time.time() > deadline:
                return best_set
            
            for trial in range(50):
                if time.time() > deadline:
                    return best_set
                
                # Generate a random set with 0
                U = {0}
                attempts = 0
                while len(U) < n and attempts < 5000:
                    # Bias toward larger values to maximize spread
                    if random.random() < 0.3:
                        U.add(random.randint(1, M))
                    else:
                        # Geometric-like distribution
                        U.add(int(random.random() ** 0.5 * M))
                    attempts += 1
                if len(U) < n:
                    continue
                U = sorted(U)
                
                s = fast_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
=======
    start = time.time()
    deadline = start + 140  # Safety margin before 150s kill
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Systematic exploration of geometric progressions
    # Powers of 2
    for n in range(2, 25):
        if time.time() > deadline:
            return best_set
        U = [0] + [2**i for i in range(n-1)]
        if max(U) > 100000:
            break
        s = fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Other integer bases
    for base in range(2, 15):
        if time.time() > deadline:
            return best_set
        for n in range(2, 20):
            U = [0] + [base**i for i in range(n-1)]
            if max(U) > 200000:
                break
            s = fast_score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Phase 1b: Fractional/rational bases for geometric progressions
    for b_num in range(2, 8):
        for b_den in range(1, 4):
            if b_den >= b_num:
                continue
            b = b_num / b_den
            for m in range(2, 30):
                if time.time() > deadline:
                    return best_set
                U = [0] + [int(b**k) for k in range(1, m+1)]
                U = sorted(set(U))
                if max(U) > 200000 or len(U) < 2:
                    break
                s = fast_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Phase 1c: Geometric with jitter
    for base in [1.5, 2, 2.5, 3, 4]:
        for m in range(3, 25):
            if time.time() > deadline:
                return best_set
            for jitter in [0, 1, 2, 5]:
                U = structured_geometric(base, m, jitter)
                if U is None or max(U) > 200000:
                    continue
                s = fast_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Phase 2: Random search with guided generation
    # Strategy: generate sets that are "spread out" to maximize differences
    for n in range(3, 18):
        for M in [10, 20, 50, 100, 200, 500, 1000, 2000]:
            if time.time() > deadline:
                return best_set
            
            for trial in range(50):
                if time.time() > deadline:
                    return best_set
                
                # Generate a random set with 0
                U = {0}
                attempts = 0
                while len(U) < n and attempts < 5000:
                    # Bias toward larger values to maximize spread
                    if random.random() < 0.3:
                        U.add(random.randint(1, M))
                    else:
                        # Geometric-like distribution
                        U.add(int(random.random() ** 0.5 * M))
                    attempts += 1
                if len(U) < n:
                    continue
                U = sorted(U)
                
                s = fast_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
>>>>>>> REPLACE

