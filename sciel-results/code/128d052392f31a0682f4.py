import numpy as np
import time
import random
from itertools import product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    def generate_vectors(n):
        """Generate all integer vectors in Z^11 with squared norm n."""
        if n == 0:
            return [np.zeros(d, dtype=np.int64)]
        
        vectors = []
        
        # Recursive helper to generate combinations of squared values
        # We need sums of squares equal to n.
        # Since n is small, we can iterate over possible non-zero entries.
        
        # Max value for any entry is sqrt(n).
        max_val = int(np.sqrt(n))
        
        # We can use backtracking to assign values to positions 0..d-1
        # To avoid duplicates, we can enforce an order? No, signs and positions matter.
        # Better: iterate over all possible multisets of non-zero squares?
        # Or just brute force small n?
        # For n=6, max_val=2.
        # Number of vectors is not huge.
        
        # Let's use a simple backtracking generator
        def backtrack(pos, remaining, current_vec):
            if pos == d:
                if remaining == 0:
                    vectors.append(current_vec.copy())
                return
            
            # Try 0
            current_vec[pos] = 0
            backtrack(pos + 1, remaining, current_vec)
            
            # Try non-zero values
            # To avoid generating permutations of the same multiset of non-zeros multiple times?
            # No, we want all distinct vectors.
            # We can just try all possible values v such that v^2 <= remaining.
            for val in range(1, max_val + 1):
                if val * val <= remaining:
                    current_vec[pos] = val
                    backtrack(pos + 1, remaining - val * val, current_vec)
                    current_vec[pos] = -val
                    backtrack(pos + 1, remaining - val * val, current_vec)
            current_vec[pos] = 0 # Reset for next iteration? No, loop handles it.
        
        current_vec = np.zeros(d, dtype=np.int64)
        backtrack(0, n, current_vec)
        
        # Remove zero vector if n>0 (shouldn't be there)
        if n > 0:
            vectors = [v for v in vectors if np.any(v != 0)]
            
        return vectors

    best_points = None
    best_size = 0

    # Try different norms
    for n in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
            
        candidates = generate_vectors(n)
        if not candidates:
            continue
            
        cand_array = np.array(candidates, dtype=np.int64)
        k_cand = len(cand_array)
        
        # Greedy search with random restarts
        # We want to select a subset S such that ||u-v||^2 >= n for all u!=v in S
        
        # Precompute squared distances? k_cand^2 might be large for n=4 (5300^2 ~ 28M).
        # 28M floats is ~200MB. Might be okay, but let's check memory.
        # Or compute on the fly.
        
        # Let's try to precompute if k_cand < 2000, else compute on fly?
        # 5300 is not that big. 5300*5300*11*8 bytes ~ 2.4 GB. Too big.
        # So compute on the fly or use matrix multiplication in chunks?
        # Actually, we can just compute ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 2n - 2 u.v.
        # Since ||u||^2 = ||v||^2 = n.
        # So condition ||u-v||^2 >= n  <=>  2n - 2 u.v >= n  <=>  n >= 2 u.v  <=>  u.v <= n/2.
        
        # This is much faster! Just check dot product.
        
        max_dot = n / 2.0
        
        def is_compatible(u, v, s):
            # u, v are int64 arrays
            # Check if v is compatible with all in s
            # Dot products
            dots = s @ v
            return np.all(dots <= max_dot)
            
        # We can vectorize the check against the whole set S if S is stored as a matrix.
        # S: (k_s, d). v: (d,).
        # S @ v -> (k_s,).
        
        for restart in range(200): # Adjust based on time
            if time.time() > deadline - 5:
                break
                
            random.shuffle(candidates)
            
            S = []
            S_matrix = None
            
            for v in candidates:
                # Check compatibility
                if S_matrix is None:
                    S.append(v)
                    S_matrix = v.reshape(1, -1)
                else:
                    dots = S_matrix @ v
                    if np.all(dots <= max_dot):
                        S.append(v)
                        S_matrix = np.vstack([S_matrix, v.reshape(1, -1)])
            
            if len(S) > best_size:
                best_size = len(S)
                best_points = np.array(S, dtype=np.int64)
                
                # Early exit if we find a very large set?
                # Theoretical max for kissing number 11 is 552 (real).
                # Integer points might be less.
                # If we find > 300, it's good.
                
    # Fallback to the original 220 points if nothing found or worse
    if best_points is None or best_size < 220:
         points = []
         for i in range(d):
             for j in range(i + 1, d):
                 v = [0] * d
                 v[i] = 1
                 v[j] = 1
                 points.append(v)
                 v = [0] * d
                 v[i] = 1
                 v[j] = -1
                 points.append(v)
                 v = [0] * d
                 v[i] = -1
                 v[j] = 1
                 points.append(v)
                 v = [0] * d
                 v[i] = -1
                 v[j] = -1
                 points.append(v)
         best_points = np.array(points, dtype=np.int64)
         best_size = 220

    # Final verification
    if best_size < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    norms = np.sum(best_points**2, axis=1)
    max_norm = np.max(norms)
    
    min_dist_sq = float('inf')
    k = len(best_points)
    for i in range(k):
        for j in range(i + 1, k):
            diff = best_points[i] - best_points[j]
            dist_sq = int(np.dot(diff, diff))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    if min_dist_sq < max_norm:
        # Fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    return best_points
