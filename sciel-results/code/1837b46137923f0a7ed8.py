import numpy as np
import time

def _max_weight_matching(weights, n):
    """
    Compute maximum weight matching in a complete graph using a simple greedy
    + local improvement approach (not exact, but fast and effective for heuristics).
    Returns a list of (i, j) pairs.
    """
    # Start with greedy: sort all edges by weight descending, pick if both endpoints free
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if weights[i, j] > 1e-9:
                edges.append((weights[i, j], i, j))
    
    edges.sort(reverse=True)
    
    matched = {}
    result = []
    for w, i, j in edges:
        if i not in matched and j not in matched:
            matched[i] = j
            matched[j] = i
            result.append((i, j))
    
    # Local improvement: try swapping pairs to improve total weight
    improved = True
    max_swaps = 50  # limit swaps for speed
    swaps_done = 0
    while improved and swaps_done < max_swaps:
        improved = False
        swaps_done += 1
        # For each pair (a,b) and (c,d), check if swapping to (a,c),(b,d) or (a,d),(b,c) improves
        for idx1 in range(len(result)):
            for idx2 in range(idx1 + 1, len(result)):
                a, b = result[idx1]
                c, d = result[idx2]
                current_w = weights[a, b] + weights[c, d]
                
                # Option 1: (a,c), (b,d)
                new_w1 = weights[a, c] + weights[b, d]
                if new_w1 > current_w + 1e-12:
                    result[idx1] = (a, c)
                    result[idx2] = (b, d)
                    matched[a] = c
                    matched[c] = a
                    matched[b] = d
                    matched[d] = b
                    improved = True
                    break
                
                # Option 2: (a,d), (b,c)
                new_w2 = weights[a, d] + weights[b, c]
                if new_w2 > current_w + 1e-12:
                    result[idx1] = (a, d)
                    result[idx2] = (b, c)
                    matched[a] = d
                    matched[d] = a
                    matched[b] = c
                    matched[c] = b
                    improved = True
                    break
            if improved:
                break
    
    return result

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Compute pair compatibility based on tightness (inverse waste).
    2. Find maximum weight matching to identify optimal disjoint pairs.
    3. Construct heuristic with high values for matched pairs, moderate for other valid pairs.
    4. Use spectral method as fallback if matching produces poor coverage.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Compute pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Valid mask: pair sum <= capacity
    valid = pair_sums <= capacity
    
    # Initialize heuristic matrix with small positive value for invalid entries
    eta = np.full((n, n), 1e-9, dtype=float)
    
    if not np.any(valid):
        return eta
    
    # For valid entries, compute waste and tightness-based weights
    waste = capacity - pair_sums
    
    # Tightness-based weight: higher when waste is small (tight fit)
    # Using a smooth function that's well-behaved
    waste_valid = waste[valid]
    # Weight = 1/(1 + waste/5)^2 - gives strong preference for tight fits
    direct_values = 1.0 / (1.0 + waste_valid / 5.0) ** 2
    
    # Build weight matrix for matching
    W = np.zeros((n, n), dtype=float)
    W[valid] = direct_values
    
    # Make symmetric (it already is, but ensure)
    W = (W + W.T) / 2.0
    
    # Find maximum weight matching
    try:
        matched_pairs = _max_weight_matching(W, n)
    except Exception:
        matched_pairs = []
    
    # Construct heuristic from matching
    # Matched pairs get high weight, other valid pairs get moderate weight
    eta_match = np.full((n, n), 1e-9, dtype=float)
    
    if len(matched_pairs) > 0:
        # High value for matched pairs (both directions)
        match_weight = 1.0
        for i, j in matched_pairs:
            eta_match[i, j] = match_weight
            eta_match[j, i] = match_weight
        
        # Moderate value for other valid pairs, scaled by their direct weight
        # but lower than matched pairs
        other_valid = valid.copy()
        for i, j in matched_pairs:
            other_valid[i, j] = False
            other_valid[j, i] = False
        
        if np.any(other_valid):
            # Scale other pairs to be 0.3x of their direct weight
            eta_match[other_valid] = 0.3 * direct_values[other_valid]
    
    # Check if matching produced reasonable coverage
    # If fewer than 30% of items are matched, the matching is poor
    matched_items = set()
    for i, j in matched_pairs:
        matched_items.add(i)
        matched_items.add(j)
    
    if len(matched_items) < 0.3 * n:
        # Fallback: use spectral method
        degree = np.sum(W, axis=1)
        L = np.diag(degree) - W
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            eta[valid] = direct_values
            eta[~np.isfinite(eta)] = 1e-9
            eta[eta <= 0] = 1e-9
            return eta
        
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        tol = 1e-8
        pos_indices = np.where(eigenvalues > tol)[0]
        
        if len(pos_indices) == 0:
            eta[valid] = direct_values
            eta[~np.isfinite(eta)] = 1e-9
            eta[eta <= 0] = 1e-9
            return eta
            
        fiedler_idx = pos_indices[0]
        v = eigenvectors[:, fiedler_idx]
        H = np.outer(v, v)
        
        max_h = np.max(H)
        if max_h > 0:
            H = H / max_h
        else:
            eta[valid] = direct_values
            eta[~np.isfinite(eta)] = 1e-9
            eta[eta <= 0] = 1e-9
            return eta
            
        H[H <= 0] = 1e-9
        H[~np.isfinite(H)] = 1e-9
        
        return H
    else:
        # Use matching-based heuristic
        eta[~np.isfinite(eta_match)] = 1e-9
        eta_match[eta_match <= 0] = 1e-9
        return eta_match
