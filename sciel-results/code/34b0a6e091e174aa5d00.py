import numpy as np
import time
import random
import itertools

# Global state
_init_done = False
_best_weights = None
_DEADLINE = 315.0  # Safety margin before 330s kill

def fast_greedy_size(n, weights):
    """
    Simulates the greedy cap set construction using a linear priority score.
    Returns the size of the resulting cap set.
    """
    if n > 12:
        return 0

    total_vectors = 3 ** n
    if total_vectors > 1000000:
        return 0

    # Generate all vectors
    idx = np.arange(total_vectors)
    vectors = np.zeros((total_vectors, n), dtype=np.int8)
    for i in range(n):
        vectors[:, i] = (idx // (3**i)) % 3

    # Calculate priorities
    priorities = vectors.dot(weights)
    
    # Get sorted indices
    sorted_indices = np.argsort(-priorities)
    
    # Greedy construction
    S = set()
    size = 0
    
    for idx_i in sorted_indices:
        v = vectors[idx_i]
        v_tuple = tuple(v)
        
        can_add = True
        for u_tuple in S:
            w_candidate = tuple((2*vi - ui) % 3 for vi, ui in zip(v_tuple, u_tuple))
            if w_candidate in S:
                can_add = False
                break
        
        if can_add:
            S.add(v_tuple)
            size += 1
            
    return size

def compute_fourier_priority(n):
    """
    Computes a priority vector based on the Fourier spectrum of the cap set constraint.
    The idea is to penalize vectors that are "aligned" with the low-frequency components
    that tend to form 3-term arithmetic progressions (3-APs).
    
    In {0,1,2}^n, a 3-AP is x, x+d, x+2d (mod 3).
    The "danger" of a vector v is related to how easily it can be part of such a structure.
    A spectral approach suggests looking at the correlation of the vector with characters of the group.
    
    We approximate a "spectral penalty" by computing the sum of squares of the coordinates' deviations from the mean,
    weighted by a frequency-dependent factor. Vectors with high variance in their coordinates (in a spectral sense)
    are often less likely to form APs with other high-priority vectors if we prioritize "flat" or "balanced" vectors?
    Actually, the literature suggests that cap sets are large when they avoid arithmetic structure.
    
    Let's try a different angle: 
    The "all-ones" vector (1,1,...,1) is often part of many APs.
    Vectors close to the mean (1,1,...,1) in the L2 sense might be "high frequency" in the sense of being central.
    Instead, let's prioritize vectors that are "far" from the center in a way that breaks symmetry, or conversely,
    prioritize vectors that are "balanced" to avoid clumping.
    
    A known heuristic for cap sets is to prefer vectors with a specific distribution of 0s, 1s, and 2s.
    For n=9, the maximum cap set size is 36.
    
    Let's use a simple spectral proxy: 
    Priority = - ||v - 1||^2 ? No, that prioritizes the center.
    Priority = ||v - 1||^2 ? That prioritizes the corners.
    
    Let's try to optimize weights that are derived from a random projection of the "Fourier" space.
    Since we can't compute the full Fourier transform of the greedy set efficiently, we use a static heuristic
    inspired by spectral properties: 
    We want to avoid vectors that are "collinear" in the affine geometry over F3.
    
    Strategy: 
    1. Generate a random weight vector w.
    2. Compute the "spectral energy" of each vector v: E(v) = sum_i (v_i - 1)^2 * w_i.
    3. This is just a weighted L2 distance from the center.
    
    Let's try to optimize weights that maximize the cap set size using this specific form:
    priority(v) = sum_i w_i * (v_i - 1)^2
    This is equivalent to a linear priority on the vector of (v_i-1)^2.
    Since v_i in {0,1,2}, (v_i-1)^2 is 1 for 0 and 2, and 0 for 1.
    So priority(v) = sum_{i: v_i != 1} w_i.
    This means we prioritize vectors that have many coordinates not equal to 1, weighted by w_i.
    
    This is a distinct structural change from the current linear dot product. It focuses on the "non-center" coordinates.
    """
    
    # Optimize weights w for the priority function: sum_{i: v_i != 1} w_i
    # We can simulate this by creating a binary matrix B where B[j, i] = 1 if v_j[i] != 1 else 0.
    # Then priority = B @ w.
    
    if n > 10:
        return np.ones(n)
        
    total_vectors = 3 ** n
    idx = np.arange(total_vectors)
    vectors = np.zeros((total_vectors, n), dtype=np.int8)
    for i in range(n):
        vectors[:, i] = (idx // (3**i)) % 3
        
    # Binary matrix for "non-1" coordinates
    B = (vectors != 1).astype(np.float64)
    
    # Simulate greedy with priority = B @ w
    def simulate(w):
        priorities = B.dot(w)
        sorted_indices = np.argsort(-priorities)
        
        S = set()
        size = 0
        for idx_i in sorted_indices:
            v = vectors[idx_i]
            v_tuple = tuple(v)
            can_add = True
            for u_tuple in S:
                w_candidate = tuple((2*vi - ui) % 3 for vi, ui in zip(v_tuple, u_tuple))
                if w_candidate in S:
                    can_add = False
                    break
            if can_add:
                S.add(v_tuple)
                size += 1
        return size

    best_w = np.random.rand(n)
    best_score = simulate(best_w)
    
    start_time = time.time()
    deadline = _DEADLINE - 10.0
    
    current_w = best_w.copy()
    
    while time.time() < deadline:
        # Perturb weights
        # Weights can be negative or positive. 
        # Positive weight means "prefer non-1". Negative means "prefer 1".
        noise = np.random.normal(0, 0.5, n)
        new_w = current_w + noise
        
        new_score = simulate(new_w)
        
        if new_score > best_score:
            best_score = new_score
            best_w = new_w
            current_w = new_w.copy()
        else:
            current_w = best_w.copy()
            
        if time.time() > deadline:
            break
            
    return best_w

def priority(vector, n):
    global _init_done, _best_weights
    
    if not _init_done:
        start_time = time.time()
        
        if n <= 10:
            _best_weights = compute_fourier_priority(n)
        else:
            _best_weights = np.ones(n)
            
        _init_done = True
        
    # Calculate priority: sum_{i: vector[i] != 1} _best_weights[i]
    score = 0.0
    for i in range(n):
        if vector[i] != 1:
            score += _best_weights[i]
            
    return score
