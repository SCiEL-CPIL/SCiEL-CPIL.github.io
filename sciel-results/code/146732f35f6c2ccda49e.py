"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    deadline = time.time() + 140  # 140s safety margin before 150s kill
    
    # Generate all nonzero integer vectors with squared norm 4 in R^11
    # Type A: one coordinate ±2
    # Type B: two coordinates ±1
    # Type C: four coordinates ±1
    
    vectors = []
    
    # Type A: one coordinate ±2
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            vectors.append(v)
    
    # Type B: two coordinates ±1
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
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
    
    n = len(vectors)
    mat = np.array(vectors, dtype=np.int64)
    
    # Compute dot product matrix
    # Two vectors are compatible if dot product <= 2
    # Incompatible if dot product > 2
    dots = mat @ mat.T
    
    # Conflict: dot > 2
    conflict = dots > 2
    
    # Diagonal is always conflict (self), but we won't select same vector twice
    np.fill_diagonal(conflict, False)
    
    # Greedy independent set with random restarts
    best_set = set()
    best_size = 0
    
    rng = random.Random(42)
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering
        order = list(range(n))
        rng.shuffle(order)
        
        # Greedy: pick vectors in order if they don't conflict with already picked
        selected = []
        selected_set = set()
        
        for idx in order:
            if time.time() > deadline:
                break
            # Check if idx conflicts with any selected
            compatible = True
            for s in selected:
                if conflict[idx, s]:
                    compatible = False
                    break
            if compatible:
                selected.append(idx)
                selected_set.add(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
        
        # If we found a large set, try local search to improve
        if len(selected) > best_size - 5 and time.time() < deadline - 10:
            # Try to add more vectors
            improved = True
            current_set = set(selected)
            while improved and time.time() < deadline - 5:
                improved = False
                # Try adding each unselected vector
                for idx in range(n):
                    if idx in current_set:
                        continue
                    # Check compatibility with all in current_set
                    compatible = True
                    for s in current_set:
                        if conflict[idx, s]:
                            compatible = False
                            break
                    if compatible:
                        current_set.add(idx)
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
            
            # Try swapping: remove one, add two
            if time.time() < deadline - 5:
                for idx_remove in list(current_set):
                    if time.time() > deadline:
                        break
                    remaining = current_set - {idx_remove}
                    # Find two vectors that can be added
                    candidates = []
                    for idx_add in range(n):
                        if idx_add in remaining:
                            continue
                        compatible = True
                        for s in remaining:
                            if conflict[idx_add, s]:
                                compatible = False
                                break
                        if compatible:
                            candidates.append(idx_add)
                    
                    # Try pairs from candidates
                    for i1 in range(len(candidates)):
                        if time.time() > deadline:
                            break
                        for i2 in range(i1 + 1, len(candidates)):
                            a, b = candidates[i1], candidates[i2]
                            if not conflict[a, b]:
                                new_set = remaining | {a, b}
                                if len(new_set) > len(current_set):
                                    current_set = new_set
                                    improved = True
                                    if len(current_set) > best_size:
                                        best_size = len(current_set)
                                        best_set = set(current_set)
                                    break
                        if improved:
                            break
                    if improved:
                        break
    
    # Return the best set found
    if best_set:
        result = mat[list(best_set)]
    else:
        # Fallback: just return the Type B vectors (220)
        result = mat[:220]
    
    return result
# EVOLVE-BLOCK-END
