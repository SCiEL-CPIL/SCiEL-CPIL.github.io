"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools

def construct():
    deadline = time.time() + 140  # 140s safety margin before 150s kill
    
    # Focus on Type C vectors (squared norm 4, 4 coordinates +-1)
    # This avoids the mixing constraint issues where R^2=4 forces strict
    # compatibility between lower-norm vectors.
    
    vectors = []
    indices = []
    
    # Type C: four coordinates ±1
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                for sl in (1, -1):
                                    v = [0] * D
                                    v[i] = si
                                    v[j] = sj
                                    v[k] = sk
                                    v[l] = sl
                                    vectors.append(v)
                                    indices.append((i, j, k, l, si, sj, sk, sl))
    
    n = len(vectors)
    mat = np.array(vectors, dtype=np.int64)
    
    # Compute dot product matrix
    dots = mat @ mat.T
    
    # For pure Type C set, max norm squared is 4.
    # Constraint: ||u - v||^2 >= 4 => 8 - 2*u.v >= 4 => u.v <= 2.
    # Conflict if dot > 2.
    conflict = dots > 2
    np.fill_diagonal(conflict, False)
    
    best_set = set()
    best_size = 0
    
    rng = random.Random(42)
    
    # Use a faster greedy with random restarts
    # n is 5280, which is manageable for O(n^2) dot products precomputed.
    # Greedy step is O(k) where k is current set size.
    
    max_restarts = 1000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering
        order = list(range(n))
        rng.shuffle(order)
        
        selected = []
        selected_set = set()
        
        # Optimization: maintain a bitmask or boolean array of conflicts with selected?
        # Since n=5280, a boolean array of size n is small.
        # We can maintain `available` mask. Initially all True.
        # When we pick idx, we AND `available` with `~conflict[idx]`.
        
        available = np.ones(n, dtype=bool)
        
        for idx in order:
            if not available[idx]:
                continue
            
            # Pick idx
            selected.append(idx)
            selected_set.add(idx)
            
            # Update available: remove idx and all its conflicts
            available[idx] = False
            available[conflict[idx]] = False
            
            if time.time() > deadline:
                break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
            # Try to add more if possible (simple local search)
            if time.time() < deadline - 10:
                improved = True
                current_set = set(selected)
                # Re-compute available based on current_set
                # Actually, we can just iterate through all n and check
                # But checking all n against current_set is O(n*k).
                # Better to use the `available` array logic if we tracked it.
                # Let's just do a simple pass:
                # Find any vector not in current_set that is compatible with all in current_set.
                
                # For efficiency, let's just try to add from the remaining `available` 
                # from the greedy step, plus maybe some others?
                # The greedy `available` is correct for the final set.
                # So any idx where available[idx] is True is compatible.
                
                # However, `available` was updated during greedy.
                # If we didn't pick a vector, it might still be available.
                # So we can just scan `available`.
                
                for idx in range(n):
                    if available[idx]:
                        current_set.add(idx)
                        # Update available for the new addition
                        available[idx] = False
                        available[conflict[idx]] = False
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                            
                # Check if we improved best
                if len(current_set) > len(best_set):
                    best_set = current_set
                    best_size = len(best_set)
    
    if best_set:
        result = mat[list(best_set)]
    else:
        # Fallback: return a small valid set, e.g., just one vector
        result = mat[:1]
        
    return result
# EVOLVE-BLOCK-END
