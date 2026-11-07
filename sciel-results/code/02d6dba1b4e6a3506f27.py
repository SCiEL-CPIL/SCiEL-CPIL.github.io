<<<<<<< SEARCH
def _local_exchange(h, max_rounds=500, rng=None):
    """
    Local search: repeatedly swap small amounts between pairs (i,j) to
    reduce the max correlation. Use a greedy heuristic: for the lag with
    max correlation, adjust elements to reduce it.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    best_h = h.copy()
    best_val = _value(best_h)
    
    for round_num in range(max_rounds):
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        max_val = corr[max_idx]
        
        # The lag is max_idx - (N-1)
        lag = max_idx - (N - 1)
        
        # Try to reduce correlation at this lag by adjusting pairs
        # corr[lag] = sum_i h[i] * (1-h[i+lag]) for valid i
        # = sum_i h[i] - sum_i h[i]*h[i+lag]
        # To reduce this, we want to increase sum h[i]*h[i+lag] or decrease sum h[i]
        # But sum is fixed, so we want to make h[i] and h[i+lag] more correlated
        # i.e., both high or both low
        
        improved = False
        # Try random pairs within the lag
        if lag > 0:
            valid_i = list(range(0, N - lag))
        else:
            valid_i = list(range(-lag, N))
        
        # Shuffle and try a few
        rng.shuffle(valid_i)
        for i in valid_i[:20]:
            j = i + lag if lag >= 0 else i - lag
            if j < 0 or j >= N:
                continue
            # Try to make h[i] and h[j] more aligned
            if h[i] > h[j]:
                # Move some from i to j
                delta = min(0.05, (h[i] - h[j]) / 2.0, h[i])
                if delta < 1e-6:
                    continue
                new_h = best_h.copy()
                new_h[i] -= delta
                new_h[j] += delta
                new_h = _project(new_h)
                v = _value(new_h)
                if v < best_val - 1e-12:
                    best_h = new_h
                    best_val = v
                    improved = True
                    break
            else:
                delta = min(0.05, (h[j] - h[i]) / 2.0, h[j])
                if delta < 1e-6:
                    continue
                new_h = best_h.copy()
                new_h[j] -= delta
                new_h[i] += delta
                new_h = _project(new_h)
                v = _value(new_h)
                if v < best_val - 1e-12:
                    best_h = new_h
                    best_val = v
                    improved = True
                    break
        
        if not improved:
            # Try random small perturbations
            for _ in range(10):
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i == j:
                    continue
                delta = rng.uniform(0.01, 0.1) * (1 if rng.random() < 0.5 else -1)
                new_h = best_h.copy()
                new_h[i] += delta
                new_h[j] -= delta
                new_h = _project(new_h)
                v = _value(new_h)
                if v < best_val - 1e-12:
                    best_h = new_h
                    best_val = v
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_h, best_val
=======
def _local_exchange(h, max_rounds=500, rng=None):
    """
    Improved local search: target the worst lag and make larger strategic moves.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    best_h = h.copy()
    best_val = _value(best_h)
    
    for round_num in range(max_rounds):
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        max_val = corr[max_idx]
        
        lag = max_idx - (N - 1)
        
        improved = False
        
        # For the worst lag, try to align pairs
        if lag > 0:
            valid_i = list(range(0, N - lag))
        elif lag < 0:
            valid_i = list(range(-lag, N))
        else:
            valid_i = list(range(N))
        
        # Sort by misalignment magnitude (largest first)
        if lag > 0:
            misalignments = [(abs(best_h[i] - best_h[i+lag]), i) for i in valid_i]
        elif lag < 0:
            misalignments = [(abs(best_h[i] - best_h[i-lag]), i) for i in valid_i]
        else:
            misalignments = [(0.0, i) for i in valid_i]
        
        misalignments.sort(key=lambda x: -x[0])
        
        # Try top 30 most misaligned pairs
        for _, i in misalignments[:30]:
            if lag > 0:
                j = i + lag
            elif lag < 0:
                j = i - lag
            else:
                continue
            
            if j < 0 or j >= N:
                continue
            
            # Try multiple delta sizes
            for delta_frac in [0.5, 0.3, 0.1]:
                if best_h[i] > best_h[j]:
                    delta = min(delta_frac, (best_h[i] - best_h[j]) / 2.0, best_h[i])
                    if delta < 1e-6:
                        continue
                    new_h = best_h.copy()
                    new_h[i] -= delta
                    new_h[j] += delta
                    new_h = _project(new_h)
                    v = _value(new_h)
                    if v < best_val - 1e-12:
                        best_h = new_h
                        best_val = v
                        improved = True
                        break
                else:
                    delta = min(delta_frac, (best_h[j] - best_h[i]) / 2.0, best_h[j])
                    if delta < 1e-6:
                        continue
                    new_h = best_h.copy()
                    new_h[j] -= delta
                    new_h[i] += delta
                    new_h = _project(new_h)
                    v = _value(new_h)
                    if v < best_val - 1e-12:
                        best_h = new_h
                        best_val = v
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            # Try random perturbations with larger steps
            for _ in range(20):
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i == j:
                    continue
                delta = rng.uniform(0.05, 0.3) * (1 if rng.random() < 0.5 else -1)
                new_h = best_h.copy()
                new_h[i] += delta
                new_h[j] -= delta
                new_h = _project(new_h)
                v = _value(new_h)
                if v < best_val - 1e-12:
                    best_h = new_h
                    best_val = v
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_h, best_val
>>>>>>> REPLACE

