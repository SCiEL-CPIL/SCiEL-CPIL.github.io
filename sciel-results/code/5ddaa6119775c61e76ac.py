import numpy as np
import itertools
import time
import random

D = 11

def generate_vectors_with_norm_sq(N):
    """Generate all integer vectors in R^D with norm squared N."""
    vectors = []
    # Recursive generation
    def gen(idx, remaining, current):
        if idx == D:
            if remaining == 0:
                vectors.append(tuple(current))
            return
        # Try all possible values for current[idx]
        # Max value is sqrt(remaining)
        max_val = int(np.sqrt(remaining))
        for val in range(-max_val, max_val + 1):
            new_remaining = remaining - val*val
            if new_remaining < 0:
                continue
            current[idx] = val
            gen(idx + 1, new_remaining, current)
    gen(0, N, [0]*D)
    return vectors

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_vectors = []
    best_score = 0
    
    # Try different norm squares
    # N=2 is the seed (110)
    # We try N=2 to 12
    
    for N in range(2, 13):
        if time.time() > deadline:
            break
            
        candidates = generate_vectors_with_norm_sq(N)
        if not candidates:
            continue
        
        # Constraint: v_i . v_j <= floor(N/2)
        max_dot = N // 2
        
        # Run greedy search multiple times
        best_local = []
        for _ in range(5): # 5 restarts
            if time.time() > deadline:
                break
            
            random.shuffle(candidates)
            current_set = []
            current_np = [] # Store as numpy arrays for speed? Or just lists.
            
            for v in candidates:
                # Check against all in current_set
                valid = True
                for u in current_set:
                    dot = np.dot(v, u)
                    if dot > max_dot:
                        valid = False
                        break
                if valid:
                    current_set.append(v)
            
            if len(current_set) > len(best_local):
                best_local = current_set
        
        if len(best_local) > len(best_vectors):
            best_vectors = best_local
            best_score = len(best_local)
            
    # Convert to numpy array
    if not best_vectors:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_vectors, dtype=np.int64)
