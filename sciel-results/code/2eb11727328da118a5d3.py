import numpy as np
import itertools
import time

D = 11

def generate_candidates(w):
    """Generate all vectors of weight w in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), w):
        for signs in itertools.product([1, -1], repeat=w):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def solve_for_weight(w, deadline):
    """
    Find max independent set for weight w vectors with dot product <= floor(w/2).
    """
    candidates = generate_candidates(w)
    N = len(candidates)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64), 0

    # Determine threshold
    # Condition: dist^2 >= w  =>  2w - 2 dot >= w  =>  dot <= w/2
    # Since dot is integer, dot <= floor(w/2)
    threshold = w // 2
    
    # Precompute conflict matrix if N is small enough
    # N^2 memory check
    # For w=5, N=14784, N^2 ~ 218M. Bool matrix ~ 218MB. OK.
    # For w=6, N=29568, N^2 ~ 874M. Bool matrix ~ 874MB. Might be tight.
    
    use_matrix = N < 20000 # Heuristic threshold
    
    if use_matrix:
        # Compute all pairwise dots
        # candidates shape (N, D)
        # dots shape (N, N)
        dots = candidates @ candidates.T
        conflicts = dots > threshold
        
        # Compute degrees
        degrees = np.sum(conflicts, axis=1)
        
        # Sort indices by degree
        sorted_indices = np.argsort(degrees)
        
        selected_indices = []
        selected_vectors = []
        
        # We need to check if a candidate conflicts with ANY already selected.
        # We can maintain a boolean mask 'blocked' of size N.
        # When we select idx, we mark all j where conflicts[idx, j] is True as blocked?
        # No, 'blocked' should mean "cannot be added because it conflicts with a selected vector".
        # Initially all False.
        # If we select idx, then for all j such that conflicts[idx, j] is True, j becomes blocked.
        # This is O(N) per selection. Total O(k*N).
        
        blocked = np.zeros(N, dtype=bool)
        
        for idx in sorted_indices:
            if time.time() > deadline:
                break
            if blocked[idx]:
                continue
            
            # Select idx
            selected_indices.append(idx)
            selected_vectors.append(candidates[idx])
            
            # Mark conflicts as blocked
            # conflicts[idx] is a boolean array of size N
            blocked |= conflicts[idx]
            
        return np.array(selected_vectors, dtype=np.int64), len(selected_indices)
    else:
        # Fallback for large N: Random Greedy with vectorized checks
        # We don't have the full conflict matrix.
        # We'll just do a few random restarts.
        best_k = 0
        best_set = []
        
        num_restarts = 10 # Adjust based on time
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            perm = np.random.permutation(N)
            selected_indices = []
            selected_vectors = []
            
            # To speed up, we can keep the selected vectors in a list/array
            # and check against them.
            
            for idx in perm:
                if time.time() > deadline:
                    break
                    
                v = candidates[idx]
                
                if not selected_vectors:
                    selected_indices.append(idx)
                    selected_vectors.append(v)
                    continue
                
                # Check against all selected
                # Convert selected_vectors to array for dot product?
                # This is inefficient if we do it every time.
                # Better to maintain a numpy array of selected vectors.
                
                # Let's restructure: maintain `sel_arr`
                # But appending to numpy array is O(N).
                # List of lists is better for appending, then convert?
                # Or just use the list `selected_vectors` and convert to array for dot?
                # That's O(k) conversion + O(k) dot.
                
                # Alternative: Just iterate?
                # Python loop over k items is slow.
                
                # Let's stick to the matrix approach for w<=5 and skip w=6 or use a simpler heuristic.
                # Given w=5 is likely best, let's ensure w=5 works well.
                
                # For now, if N >= 20000, we just return empty or a small greedy set.
                # But w=6 is likely not optimal.
                pass
        
        return np.array(best_set, dtype=np.int64), best_k

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_vectors = None
    best_k = 0
    
    # Try weights 2, 3, 4, 5
    for w in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
            
        vectors, k = solve_for_weight(w, deadline)
        
        if k > best_k:
            best_k = k
            best_vectors = vectors
            
    if best_vectors is None:
        return np.empty((0, D), dtype=np.int64)
        
    return best_vectors
