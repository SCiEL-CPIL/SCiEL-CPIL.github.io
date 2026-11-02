def run_greedy_list(candidates, limit_dot, max_iter=20):
    n_cands = len(candidates)
    if n_cands == 0:
        return np.array([], dtype=np.int8)
        
    best_set = []
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        indices = np.random.permutation(n_cands)
        current_set = []
        
        for idx in indices:
            v = candidates[idx]
            compatible = True
            for u in current_set:
                if np.dot(v, u) > limit_dot:
                    compatible = False
                    break
            if compatible:
                current_set.append(v)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return np.array(best_set, dtype=np.int8)
