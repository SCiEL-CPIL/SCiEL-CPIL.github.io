    # Greedy Init
    S = set()
    remaining = list(range(N))
    random.shuffle(remaining)
    for v in remaining:
        if not neighbors[v].intersection(S):
            S.add(v)
            
    best_set = S.copy()
    best_size = len(S)
    
    # SA
    T = 1.0
    cooling = 0.999
    # Steps: 200,000 should be safe
    for step in range(200000):
        v = random.randint(0, N - 1)
        if v in S:
            # Remove
            if random.random() < np.exp(-1.0 / T):
                S.remove(v)
                best_size = len(S) # Update current size
                # We don't update best here as size decreased
        else:
            N_conf = neighbors[v].intersection(S)
            if len(N_conf) == 0:
                S.add(v)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
            else:
                delta = 1 - len(N_conf)
                if delta > 0:
                    S.difference_update(N_conf)
                    S.add(v)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S.copy()
                else:
                    if random.random() < np.exp(delta / T):
                        S.difference_update(N_conf)
                        S.add(v)
        T *= cooling
        
    # Construct result
    result_vectors = [vectors[i] for i in best_set]
    return np.array(result_vectors, dtype=np.int64)
