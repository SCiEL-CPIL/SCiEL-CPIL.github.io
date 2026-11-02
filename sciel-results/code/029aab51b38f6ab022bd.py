import numpy as np
import time
import random

N = 50
DELTA = 1.0 / N  # step size

def _autoconv(h):
    """Full autoconvolution of h (length 2N-1)."""
    return np.correlate(h, h, mode='full')

def _ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    h: array of nonnegative heights.
    Returns the ratio value (higher is better).
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h**2)
    
    c = _autoconv(h)
    sup_c = np.max(c)
    
    if sup_c < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_c)

def _is_sidon_set(indices):
    """
    Check if a set of indices forms a Sidon set (B_2 set):
    all pairwise differences are distinct.
    For a Sidon set, the autoconvolution peak is at most 1 (for unit heights),
    because no two pairs have the same difference.
    """
    if len(indices) < 2:
        return True
    
    diffs = set()
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            d = indices[j] - indices[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _sidon_greedy(indices, max_size=None):
    """
    Greedy construction of a Sidon set from candidate indices.
    Try to add indices one by one, keeping only those that maintain the Sidon property.
    """
    result = []
    diffs = set()
    
    for idx in indices:
        # Check if adding idx maintains Sidon property
        valid = True
        for existing in result:
            d = abs(idx - existing)
            if d in diffs:
                valid = False
                break
            diffs.add(d)
        
        if valid:
            result.append(idx)
        
        if max_size and len(result) >= max_size:
            break
    
    return result

def _sidon_random_search(n_trials=500, deadline=None):
    """
    Random search for large Sidon sets.
    Try random orderings of [0, N) and greedily build Sidon sets.
    """
    best_set = []
    best_score = 0.0
    
    for trial in range(n_trials):
        if deadline and time.time() > deadline:
            break
        
        # Random permutation of indices
        indices = list(range(N))
        random.shuffle(indices)
        
        # Greedy Sidon construction
        sidon_set = _sidon_greedy(indices)
        
        if len(sidon_set) > len(best_set):
            best_set = sidon_set
            # Create unit-height vector on this Sidon set
            h = np.zeros(N)
            for idx in sidon_set:
                h[idx] = 1.0
            best_score = _ratio(h)
    
    return best_set, best_score

def _sidon_local_search(sidon_set, n_iterations=100, deadline=None):
    """
    Local search to improve a Sidon set by swapping elements.
    Try to replace one element with another to increase set size or maintain size
    while improving the ratio.
    """
    current_set = list(sidon_set)
    current_score = _ratio(np.array([1.0 if i in current_set else 0.0 for i in range(N)]))
    
    for iteration in range(n_iterations):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # Try swapping one element with a non-member
        for i, member in enumerate(current_set):
            for candidate in range(N):
                if candidate in current_set:
                    continue
                
                # Try the swap
                new_set = current_set[:i] + [candidate] + current_set[i+1:]
                new_set = sorted(new_set)
                
                # Check if it's still a Sidon set
                if not _is_sidon_set(new_set):
                    continue
                
                # Evaluate
                h_new = np.zeros(N)
                for idx in new_set:
                    h_new[idx] = 1.0
                score_new = _ratio(h_new)
                
                if score_new > current_score + 1e-10:
                    current_set = new_set
                    current_score = score_new
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_set, current_score

def _try_non_uniform_amplitudes(sidon_set, deadline=None):
    """
    For a given Sidon set, try non-uniform amplitudes to improve the ratio.
    Even though the Sidon property caps the peak, varying amplitudes might help.
    """
    h = np.zeros(N)
    for idx in sidon_set:
        h[idx] = 1.0
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    # Try scaling individual amplitudes
    for _ in range(50):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        for idx in sidon_set:
            current_val = h[idx]
            if current_val < 1e-15:
                continue
            
            # Try different scales
            for scale in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]:
                h_trial = h.copy()
                h_trial[idx] = current_val * scale
                s = _ratio(h_trial)
                if s > best_score + 1e-12:
                    best_score = s
                    best_h = h_trial.copy()
                    h = h_trial
                    improved = True
                    break
            # Reset to best if no improvement found for this coordinate
            if not improved:
                h = best_h.copy()
        
        if not improved:
            break
    
    return best_h, best_score

def construct():
    """
    Construct a height vector using Sidon set structure.
    
    Strategy:
    1. Find a large Sidon set (B_2 set) where all pairwise differences are distinct.
    2. For a Sidon set with unit heights, the autoconvolution peak is at most 1.
    3. The ratio becomes ||h||_2^2 / (||h||_1 * 1) = n / n = 1 for n unit spikes.
    4. But we can do better by using non-uniform amplitudes or finding the optimal structure.
    
    Actually, for unit heights on a Sidon set of size k:
    - L2^2 = k (sum of squares of k ones)
    - L1 = k
    - Sup autoconv = 1 (since no two pairs have same difference)
    - Ratio = k / (k * 1) = 1
    
    To improve, we need to either:
    - Use non-uniform amplitudes
    - Find a structure where the peak is less than the L2/L1 ratio
    """
    deadline = time.time() + 120  # Safety margin
    
    # Start with a known good Sidon set construction
    # Using quadratic residues or greedy random search
    
    random.seed(42)
    np.random.seed(42)
    
    # Try multiple random Sidon constructions
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Method 1: Random greedy Sidon search
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        indices = list(range(N))
        random.shuffle(indices)
        sidon_set = _sidon_greedy(indices)
        
        if len(sidon_set) < 2:
            continue
        
        # Create unit height vector
        h = np.zeros(N)
        for idx in sidon_set:
            h[idx] = 1.0
        
        score = _ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Method 2: Local search improvement on best Sidon set
    if best_score > 0:
        # Extract the support from best_h
        support = [i for i in range(N) if best_h[i] > 1e-10]
        
        # Try local search
        improved_set, improved_score = _sidon_local_search(support, n_iterations=50, deadline=deadline - 5)
        
        if improved_score > best_score:
            best_score = improved_score
            best_h = np.zeros(N)
            for idx in improved_set:
                best_h[idx] = 1.0
    
    # Method 3: Try non-uniform amplitudes on the best Sidon set
    if best_score > 0:
        support = [i for i in range(N) if best_h[i] > 1e-10]
        h_opt, score_opt = _try_non_uniform_amplitudes(support, deadline=deadline - 2)
        
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Fallback: if nothing worked, use a single spike
    if best_score < 1e-10:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return list(best_h)
