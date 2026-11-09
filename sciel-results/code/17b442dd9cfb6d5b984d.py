import numpy as np
import time
import random

# Global state for the optimized strategy
_init_done = False
_best_weights = None
_DEADLINE = 320.0  # Safety margin before 330s kill

def fast_greedy_size(n, weights):
    """
    Simulates the greedy cap set construction using a linear priority score.
    Returns the size of the resulting cap set.
    Uses NumPy for vectorized operations where possible, but the core loop
    is iterative due to the dependency of the set S.
    """
    # Generate all vectors in {0, 1, 2}^n
    # For n > 10, 3^n is huge. The task likely uses small n (e.g., n=6, 7, 8, 9, 10).
    # If n is too large for full enumeration in time, we might need a different approach,
    # but typically these tasks allow n up to ~10 or 12 for a 300s budget if optimized.
    # Let's assume n is small enough for 3^n to be manageable (e.g., n <= 10 -> 59049 vectors).
    
    if n > 12:
        # Fallback for very large n: just return a heuristic estimate or random priority
        # But the contract says vectors are added greedily. If we can't enumerate, we can't simulate.
        # However, the evaluator provides the vectors or expects us to handle them.
        # The interface is priority(vector, n). The evaluator generates the set.
        # For our *simulation* (to find good weights), we need to know the set.
        # If n is large, we can't simulate the full greedy process easily.
        # Let's assume the test cases for n are small enough to allow simulation in < 1s.
        # If n is large, we will rely on a fixed good heuristic.
        pass

    # Generate all vectors
    # Using np.array to store them for fast access
    # We need to be careful with memory for large n, but 3^10 is small.
    try:
        # Create meshgrid of 0,1,2 for n dimensions
        # This is memory intensive for large n.
        # Alternative: generate indices and decode.
        # For n <= 10, list(product) is fine but slow in Python.
        # Let's use a base-3 decoding approach for speed.
        
        total_vectors = 3 ** n
        # If total_vectors is too big, we can't simulate.
        if total_vectors > 1000000:
            return 0 # Signal that simulation is not possible/needed
        
        # Decode indices to base-3 digits
        # This is faster than itertools.product for large n in some contexts, 
        # but let's stick to a standard efficient generation.
        
        # Actually, for the simulation, we just need to sort by priority.
        # Priority = dot product of vector and weights.
        
        # Generate all vectors
        # Using np.unravel_index might be complex.
        # Let's use a simple loop or np.indices?
        # For n=8, 3^8 = 6561. Very fast.
        # For n=10, 3^10 = 59049. Fast.
        # For n=12, 3^12 = 531441. Still okay in Python if optimized.
        
        # We will generate vectors as a numpy array of shape (3**n, n)
        # Method: 
        #   idx = np.arange(3**n)
        #   vectors = np.zeros((3**n, n), dtype=np.int8)
        #   for i in range(n):
        #       vectors[:, i] = (idx // (3**i)) % 3
        
        idx = np.arange(total_vectors)
        vectors = np.zeros((total_vectors, n), dtype=np.int8)
        for i in range(n):
            vectors[:, i] = (idx // (3**i)) % 3
            
        # Calculate priorities
        # weights shape (n,), vectors shape (N, n)
        # priorities = vectors @ weights
        priorities = vectors.dot(weights)
        
        # Get sorted indices
        sorted_indices = np.argsort(-priorities)
        
        # Greedy construction
        S = set()
        size = 0
        
        # Pre-compute "2*v" for all vectors? No, v changes.
        # For each candidate v in sorted order:
        #   Check if for any u in S, w = 2v - u is in S.
        #   If not, add v to S.
        
        # To speed this up, we can use a dictionary or set of tuples.
        # Converting numpy arrays to tuples is slow.
        # Let's keep S as a set of integers representing the base-3 index?
        # If we map vector to index, we can check membership faster.
        # index = sum(v_i * 3^i)
        
        # Let's create a mapping from vector tuple to index?
        # Or just use the index directly in the set?
        # We need to calculate w_candidate = 2v - u.
        # If we have v and u as numpy arrays, 2v-u is easy.
        # But checking if (2v-u) is in S requires converting (2v-u) to an index.
        
        # Let's stick to tuples for correctness and simplicity, as n is likely small.
        # Optimization: Convert sorted_indices to a list of tuples once?
        # No, we iterate through sorted_indices.
        
        for idx_i in sorted_indices:
            v = vectors[idx_i]
            v_tuple = tuple(v)
            
            can_add = True
            for u_tuple in S:
                # u_tuple is a tuple of ints
                # w_candidate = tuple((2*vi - ui) % 3 for vi, ui in zip(v_tuple, u_tuple))
                # This inner loop is the bottleneck.
                
                # Vectorized check for a single u?
                # w = (2*v - np.array(u_tuple)) % 3
                # w_tuple = tuple(w)
                # if w_tuple in S:
                #    can_add = False
                #    break
                
                # Even better: Check all u in S at once?
                # S is a set of tuples. We can't vectorize set membership easily.
                
                # However, for small n (<= 8), the set size will be small (cap set size grows as 2^n approx).
                # So the inner loop is short.
                
                w_candidate = tuple((2*vi - ui) % 3 for vi, ui in zip(v_tuple, u_tuple))
                if w_candidate in S:
                    can_add = False
                    break
            
            if can_add:
                S.add(v_tuple)
                size += 1
                
    except Exception:
        return 0
        
    return size

def optimize_weights(n):
    """
    Uses hill climbing to find weights that maximize the cap set size.
    """
    global _best_weights
    
    if n > 12:
        # For large n, simulation is too slow. Use a default heuristic.
        # Vectors with more 1s and 2s (higher sum) often work well?
        # Or balanced?
        # Let's return a uniform weight vector, which corresponds to sum(v).
        return np.ones(n)
    
    # Start with random weights
    best_weights = np.random.rand(n)
    best_score = fast_greedy_size(n, best_weights)
    
    start_time = time.time()
    deadline = _DEADLINE - 10.0  # Leave 10s for actual priority calls and overhead
    
    current_weights = best_weights.copy()
    
    while time.time() < deadline:
        # Perturb weights
        # Add small noise
        noise = np.random.normal(0, 0.1, n)
        new_weights = current_weights + noise
        
        # Optional: Normalize? Not strictly necessary for linear priority.
        
        new_score = fast_greedy_size(n, new_weights)
        
        if new_score > best_score:
            best_score = new_score
            best_weights = new_weights
            current_weights = new_weights.copy()
        else:
            # Revert or try different perturbation
            # Simple hill climbing: revert
            current_weights = best_weights.copy()
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_weights

def priority(vector, n):
    global _init_done, _best_weights
    
    if not _init_done:
        # Determine n for optimization
        # If n is small, optimize for this n.
        # If n is large, we might use a generic weight or optimize for a representative n.
        # The contract implies n is fixed for a run.
        
        # We should check if we have time.
        start_time = time.time()
        
        if n <= 10:
            _best_weights = optimize_weights(n)
        else:
            # For large n, use a heuristic: prefer vectors with sum close to n/2?
            # Or just sum?
            # Let's use a random weight vector as a baseline, or ones.
            # Actually, for large n, the structure is more complex.
            # Let's try to optimize for n=8 and use those weights? 
            # No, weights are dimension-specific.
            # If n > 10, we can't simulate. We must use a fixed heuristic.
            # A common good heuristic is to prioritize vectors with higher "entropy" or specific patterns.
            # Let's use a random projection as a fallback, but deterministic.
            # Or simply sum(v).
            _best_weights = np.ones(n)
            
        _init_done = True
        
    # Calculate priority
    # vector is a tuple or list
    # Convert to numpy array for dot product
    # To avoid conversion cost in the hot loop, we could cache?
    # But the evaluator calls priority(vector, n) for each vector in the sorted list?
    # No, the evaluator sorts the vectors by priority.
    # So priority is called 3^n times.
    # Converting tuple to np.array 3^n times is expensive.
    
    # Optimization: Calculate sum(v * weights) using Python loop?
    # For n=10, 10 multiplications per call. 59049 calls. ~600k ops. Fast.
    # Using np.dot involves overhead of creating the array.
    
    # Let's use a pure Python dot product for speed in the hot loop.
    score = 0.0
    for i in range(n):
        score += vector[i] * _best_weights[i]
        
    return score
