import numpy as np
import time
import random

_PRIORITY_CACHE = {}
_DEADLINE = time.time() + 325.0  # Safety margin

def _generate_data(n):
    """
    Generate all 3^n vectors and their integer indices.
    Returns:
        V: Numpy array of shape (3^n, n) with int8 digits.
        total: 3^n
        weights: Numpy array of powers of 3 for index calculation.
    """
    total = 3 ** n
    V = np.zeros((total, n), dtype=np.int8)
    
    # Generate digits
    # V[i, j] is the j-th digit (0-indexed from right) of i in base 3
    # i = d_{n-1} ... d_0
    # V[i, 0] is d_0 (LSB), V[i, n-1] is d_{n-1} (MSB)
    # weights should match: weights[j] = 3^j
    weights = 3 ** np.arange(n, dtype=np.int64)
    
    for j in range(n):
        V[:, j] = (np.arange(total) // (3 ** j)) % 3
        
    return V, total, weights

def _greedy_cap_set_numpy(V, total, weights, n, time_limit, seed):
    """
    Greedy cap set construction using Numpy vectorized conflict detection.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    
    # Forbidden is a boolean array of size 3^n
    Forbidden = np.zeros(total, dtype=bool)
    
    S_indices = []
    
    start_time = time.time()
    
    # Pre-allocate a buffer for conflicts to avoid repeated allocation?
    # Max size of conflicts is 3 * len(S_indices)
    
    for idx in perm:
        if time.time() - start_time > time_limit:
            break
        
        if Forbidden[idx]:
            continue
            
        v = V[idx]
        
        if S_indices:
            # Get current set vectors
            S_vals = V[S_indices]
            
            # Calculate conflicts
            # c1: 2v - a
            # c2: 2a - v
            # c3: 2(v + a)
            
            c1 = (2 * v - S_vals) % 3
            c2 = (2 * S_vals - v) % 3
            c3 = (2 * (v + S_vals)) % 3
            
            # Combine
            # To avoid vstack overhead, we can do:
            # conflicts = np.concatenate((c1, c2, c3))
            # But np.concatenate creates a new array.
            
            conflict_indices = np.empty((len(S_vals) * 3,), dtype=np.int64)
            
            # Calculate indices for c1
            conflict_indices[0:len(S_vals)] = np.sum(c1 * weights, axis=1)
            conflict_indices[len(S_vals):2*len(S_vals)] = np.sum(c2 * weights, axis=1)
            conflict_indices[2*len(S_vals):3*len(S_vals)] = np.sum(c3 * weights, axis=1)
            
            Forbidden[conflict_indices] = True
            
        # Add v to set
        Forbidden[idx] = True
        S_indices.append(idx)
        
    return S_indices

def priority(vector, n):
    if n not in _PRIORITY_CACHE:
        V, total, weights = _generate_data(n)
        
        best_set = set()
        best_size = 0
        start_search = time.time()
        
        # Run multiple restarts
        seed = 42
        while time.time() - start_search < 300.0:
            # Estimate time for one run
            # If n is small, runs are fast. If n is large, runs are slow.
            # Use a dynamic time limit per run.
            # Let's try to fit as many runs as possible.
            # For n=10, one run might take ~10-20s.
            # We have 300s. So ~15-30 runs.
            
            # Simple heuristic: limit each run to 20s, or until deadline.
            time_limit_for_run = min(20.0, 300.0 - (time.time() - start_search))
            
            if time_limit_for_run <= 0:
                break
                
            S_indices = _greedy_cap_set_numpy(V, total, weights, n, time_limit_for_run, seed)
            
            if len(S_indices) > best_size:
                best_size = len(S_indices)
                best_set = set(S_indices)
                
            seed += 1
            
        _PRIORITY_CACHE[n] = best_set

    # Convert input vector to index
    # vector is a tuple of length n
    # index = sum(vector[i] * 3^i for i in range(n))
    # Note: V[i, j] corresponds to 3^j.
    # So index = sum(vector[j] * 3^j)
    
    idx = 0
    for j in range(n):
        idx += vector[j] * (3 ** j)
        
    return 1.0 if idx in _PRIORITY_CACHE[n] else 0.0
