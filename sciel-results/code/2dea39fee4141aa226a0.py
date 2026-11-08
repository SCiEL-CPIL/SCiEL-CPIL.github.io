<<<<<<< SEARCH
    # Score columns by "removability" - columns that share more support are more redundant
    col_nonzeros_U = np.count_nonzero(U46, axis=0)
    col_nonzeros_V = np.count_nonzero(V46, axis=0)
    col_nonzeros_W = np.count_nonzero(W46, axis=0)
    
    # Create support sets for each column
    support_U = [set(np.nonzero(U46[:, c])[0]) for c in range(46)]
    support_V = [set(np.nonzero(V46[:, c])[0]) for c in range(46)]
    support_W = [set(np.nonzero(W46[:, c])[0]) for c in range(46)]
    
    # Score: lower = more likely to be removable (smaller support, more overlap with others)
    col_scores = []
    for c in range(46):
        support_size = len(support_U[c]) + len(support_V[c]) + len(support_W[c])
        # Count how many other columns share significant support
        overlap_count = 0
        for c2 in range(46):
            if c2 == c:
                continue
            overlap = len(support_U[c] & support_U[c2]) + len(support_V[c] & support_V[c2]) + len(support_W[c] & support_W[c2])
            if overlap > 3:
                overlap_count += 1
        score = support_size + overlap_count * 0.5
        col_scores.append(score)
    
    col_order = np.argsort(col_scores)
    
    # Strategy 1: Try removing single columns (rank 45)
    max_attempts = min(20, len(col_order))
    for attempt in range(max_attempts):
        if time.time() > deadline - 30:
            break
            
        col_to_zero = col_order[attempt]
        mask = [i for i in range(46) if i != col_to_zero]
        U_test = U46[:, mask].copy()
        V_test = V46[:, mask].copy()
        W_test = W46[:, mask].copy()
        
        # Try multiple noise scales and seeds
        found = False
        for seed in [42, 123, 999]:
            if time.time() > deadline - 25:
                break
            np.random.seed(seed + attempt * 100)
            for noise_scale in [0.005, 0.02, 0.08]:
                if time.time() > deadline - 22:
                    break
                U_try = U_test.copy()
                V_try = V_test.copy()
                W_try = W_test.copy()
                U_try += np.random.randn(*U_try.shape) * noise_scale
                V_try += np.random.randn(*V_try.shape) * noise_scale
                W_try += np.random.randn(*W_try.shape) * noise_scale
                
                U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 45, max_iter=80, lr=0.04)
                
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Strategy 2: If rank 45 failed, try removing pairs of columns (rank 44)
    if best_result is None and time.time() < deadline - 35:
        # Try pairs of columns with high overlap
        pair_candidates = []
        for i in range(len(col_order)):
            for j in range(i+1, len(col_order)):
                c1 = col_order[i]
                c2 = col_order[j]
                overlap = len(support_U[c1] & support_U[c2]) + len(support_V[c1] & support_V[c2]) + len(support_W[c1] & support_W[c2])
                pair_candidates.append((overlap, c1, c2))
        pair_candidates.sort(reverse=True)
        
        for idx in range(min(15, len(pair_candidates))):
            if time.time() > deadline - 25:
                break
            _, c1, c2 = pair_candidates[idx]
            mask = [i for i in range(46) if i != c1 and i != c2]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            for seed in [42, 123]:
                if time.time() > deadline - 22:
                    break
                np.random.seed(seed + idx * 10)
                for noise_scale in [0.01, 0.05]:
                    if time.time() > deadline - 20:
                        break
                    U_try = U_test.copy()
                    V_try = V_test.copy()
                    W_try = W_test.copy()
                    U_try += np.random.randn(*U_try.shape) * noise_scale
                    V_try += np.random.randn(*V_try.shape) * noise_scale
                    W_try += np.random.randn(*W_try.shape) * noise_scale
                    
                    U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 44, max_iter=70, lr=0.035)
                    
                    if U_opt is not None:
                        best_result = (U_opt, V_opt, W_opt)
                        break
                if best_result is not None:
                    break
            if best_result is not None:
                break
    
    # Strategy 3: Try rank 43 with structured perturbation
    if best_result is None and time.time() < deadline - 30:
        # Remove 3 columns with highest mutual overlap
        triple_candidates = []
        for i in range(len(col_order)):
            for j in range(i+1, len(col_order)):
                for k in range(j+1, len(col_order)):
                    c1, c2, c3 = col_order[i], col_order[j], col_order[k]
                    overlap = (len(support_U[c1] & support_U[c2]) + len(support_V[c1] & support_V[c2]) + len(support_W[c1] & support_W[c2]) +
                              len(support_U[c1] & support_U[c3]) + len(support_V[c1] & support_V[c3]) + len(support_W[c1] & support_W[c3]) +
                              len(support_U[c2] & support_U[c3]) + len(support_V[c2] & support_V[c3]) + len(support_W[c2] & support_W[c3]))
                    triple_candidates.append((overlap, c1, c2, c3))
        triple_candidates.sort(reverse=True)
        
        for idx in range(min(8, len(triple_candidates))):
            if time.time() > deadline - 20:
                break
            _, c1, c2, c3 = triple_candidates[idx]
            mask = [i for i in range(46) if i not in (c1, c2, c3)]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            for seed in [42, 123]:
                if time.time() > deadline - 18:
                    break
                np.random.seed(seed + idx * 20)
                U_try = U_test.copy()
                V_try = V_test.copy()
                W_try = W_test.copy()
                U_try += np.random.randn(*U_try.shape) * 0.05
                V_try += np.random.randn(*V_try.shape) * 0.05
                W_try += np.random.randn(*W_try.shape) * 0.05
                
                U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 43, max_iter=60, lr=0.03)
                
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    break
            if best_result is not None:
                break
=======
    # Score columns by "removability" - columns that share more support are more redundant
    col_nonzeros_U = np.count_nonzero(U46, axis=0)
    col_nonzeros_V = np.count_nonzero(V46, axis=0)
    col_nonzeros_W = np.count_nonzero(W46, axis=0)
    
    # Create support sets for each column
    support_U = [set(np.nonzero(U46[:, c])[0]) for c in range(46)]
    support_V = [set(np.nonzero(V46[:, c])[0]) for c in range(46)]
    support_W = [set(np.nonzero(W46[:, c])[0]) for c in range(46)]
    
    # Score: lower = more likely to be removable (smaller support, more overlap with others)
    col_scores = []
    for c in range(46):
        support_size = len(support_U[c]) + len(support_V[c]) + len(support_W[c])
        # Count how many other columns share significant support
        overlap_count = 0
        for c2 in range(46):
            if c2 == c:
                continue
            overlap = len(support_U[c] & support_U[c2]) + len(support_V[c] & support_V[c2]) + len(support_W[c] & support_W[c2])
            if overlap > 3:
                overlap_count += 1
        score = support_size + overlap_count * 0.5
        col_scores.append(score)
    
    col_order = np.argsort(col_scores)
    
    # Strategy 1: Try removing single columns (rank 45) with more aggressive search
    max_attempts = min(25, len(col_order))
    for attempt in range(max_attempts):
        if time.time() > deadline - 35:
            break
            
        col_to_zero = col_order[attempt]
        mask = [i for i in range(46) if i != col_to_zero]
        U_test = U46[:, mask].copy()
        V_test = V46[:, mask].copy()
        W_test = W46[:, mask].copy()
        
        # Try multiple noise scales and seeds
        found = False
        for seed in [42, 123, 999, 777, 555]:
            if time.time() > deadline - 30:
                break
            np.random.seed(seed + attempt * 100)
            for noise_scale in [0.003, 0.01, 0.03, 0.08, 0.15]:
                if time.time() > deadline - 28:
                    break
                U_try = U_test.copy()
                V_try = V_test.copy()
                W_try = W_test.copy()
                U_try += np.random.randn(*U_try.shape) * noise_scale
                V_try += np.random.randn(*V_try.shape) * noise_scale
                W_try += np.random.randn(*W_try.shape) * noise_scale
                
                U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 45, max_iter=100, lr=0.05)
                
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Strategy 2: If rank 45 failed, try removing pairs of columns (rank 44)
    if best_result is None and time.time() < deadline - 40:
        # Try pairs of columns with high overlap
        pair_candidates = []
        for i in range(len(col_order)):
            for j in range(i+1, len(col_order)):
                c1 = col_order[i]
                c2 = col_order[j]
                overlap = len(support_U[c1] & support_U[c2]) + len(support_V[c1] & support_V[c2]) + len(support_W[c1] & support_W[c2])
                pair_candidates.append((overlap, c1, c2))
        pair_candidates.sort(reverse=True)
        
        for idx in range(min(20, len(pair_candidates))):
            if time.time() > deadline - 30:
                break
            _, c1, c2 = pair_candidates[idx]
            mask = [i for i in range(46) if i != c1 and i != c2]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            for seed in [42, 123, 999]:
                if time.time() > deadline - 28:
                    break
                np.random.seed(seed + idx * 10)
                for noise_scale in [0.01, 0.03, 0.08]:
                    if time.time() > deadline - 26:
                        break
                    U_try = U_test.copy()
                    V_try = V_test.copy()
                    W_try = W_test.copy()
                    U_try += np.random.randn(*U_try.shape) * noise_scale
                    V_try += np.random.randn(*V_try.shape) * noise_scale
                    W_try += np.random.randn(*W_try.shape) * noise_scale
                    
                    U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 44, max_iter=80, lr=0.04)
                    
                    if U_opt is not None:
                        best_result = (U_opt, V_opt, W_opt)
                        break
                if best_result is not None:
                    break
            if best_result is not None:
                break
    
    # Strategy 3: Try rank 43 with structured perturbation
    if best_result is None and time.time() < deadline - 35:
        # Remove 3 columns with highest mutual overlap
        triple_candidates = []
        for i in range(len(col_order)):
            for j in range(i+1, len(col_order)):
                for k in range(j+1, len(col_order)):
                    c1, c2, c3 = col_order[i], col_order[j], col_order[k]
                    overlap = (len(support_U[c1] & support_U[c2]) + len(support_V[c1] & support_V[c2]) + len(support_W[c1] & support_W[c2]) +
                              len(support_U[c1] & support_U[c3]) + len(support_V[c1] & support_V[c3]) + len(support_W[c1] & support_W[c3]) +
                              len(support_U[c2] & support_U[c3]) + len(support_V[c2] & support_V[c3]) + len(support_W[c2] & support_W[c3]))
                    triple_candidates.append((overlap, c1, c2, c3))
        triple_candidates.sort(reverse=True)
        
        for idx in range(min(12, len(triple_candidates))):
            if time.time() > deadline - 25:
                break
            _, c1, c2, c3 = triple_candidates[idx]
            mask = [i for i in range(46) if i not in (c1, c2, c3)]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            for seed in [42, 123, 999]:
                if time.time() > deadline - 23:
                    break
                np.random.seed(seed + idx * 20)
                for noise_scale in [0.03, 0.08]:
                    if time.time() > deadline - 22:
                        break
                    U_try = U_test.copy()
                    V_try = V_test.copy()
                    W_try = W_test.copy()
                    U_try += np.random.randn(*U_try.shape) * noise_scale
                    V_try += np.random.randn(*V_try.shape) * noise_scale
                    W_try += np.random.randn(*W_try.shape) * noise_scale
                    
                    U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 43, max_iter=70, lr=0.04)
                    
                    if U_opt is not None:
                        best_result = (U_opt, V_opt, W_opt)
                        break
                if best_result is not None:
                    break
            if best_result is not None:
                break
>>>>>>> REPLACE

