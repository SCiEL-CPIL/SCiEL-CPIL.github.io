import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Helper to evaluate score
    def get_score(U):
        if len(U) < 2:
            return 1.0
        U_arr = np.array(U, dtype=np.int64)
        # Sort and unique to ensure set properties
        U_arr = np.unique(U_arr)
        
        sums = np.add.outer(U_arr, U_arr)
        diffs = np.subtract.outer(U_arr, U_arr)
        
        num_sums = len(np.unique(sums))
        num_diffs = len(np.unique(diffs))
        
        if num_sums == 0:
            return 1.0
            
        max_u = np.max(U_arr)
        denom = np.log(2 * max_u + 1)
        
        if denom == 0:
            return 1.0
            
        ratio = num_diffs / num_sums
        score = 1.0 + np.log(ratio) / denom
        return score

    # Initial candidates
    candidates = []
    
    # Seed
    candidates.append([0, 1, 3])
    
    # Heuristic 1: Binary-like (powers of 2)
    candidates.append(list(np.array([0, 1, 2, 4, 8, 16, 32, 64])))
    
    # Heuristic 2: Squares
    candidates.append(list(np.arange(0, 20)**2))
    
    # Heuristic 3: Range
    candidates.append(list(range(0, 20)))
    
    # Heuristic 4: "Binary" sparse (numbers with <= 2 bits set, up to some bound)
    def gen_binary_sparse(limit):
        s = set([0])
        for i in range(limit):
            for j in range(i+1, limit):
                s.add(2**i + 2**j)
            s.add(2**i)
        return sorted(list(s))
    candidates.append(gen_binary_sparse(6))
    
    # Evaluate initial candidates
    best_set = candidates[0]
    best_score = get_score(best_set)
    
    for c in candidates[1:]:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_set = c
            
    # Local Search
    current_set = best_set[:]
    current_score = best_score
    max_val = max(current_set) if current_set else 10
    limit_val = max_val * 2 + 10
    min_val = 0
    
    step_count = 0
    
    while time.time() < deadline:
        step_count += 1
        
        # Generate a move
        move_type = random.random()
        new_set = current_set[:]
        
        if move_type < 0.5:
            # Replace an element
            if len(new_set) > 0:
                idx = random.randint(0, len(new_set) - 1)
                # Try to keep values within a reasonable range to control max U
                # Sample from a distribution centered around current max
                new_val = random.randint(0, limit_val)
                new_set[idx] = new_val
        elif move_type < 0.7:
            # Add an element
            new_val = random.randint(0, limit_val)
            new_set.append(new_val)
            # Prevent set from growing too large (expensive to eval)
            if len(new_set) > 100:
                # If too big, maybe remove a random one later or just skip this move type if too big
                # For now, just allow it but evaluation will be slower
                pass
        elif move_type < 0.9:
            # Remove an element
            if len(new_set) > 2:
                idx = random.randint(0, len(new_set) - 1)
                new_set.pop(idx)
        else:
            # Perturb max value? Or shift?
            # Just random replacement for now
            if len(new_set) > 0:
                idx = random.randint(0, len(new_set) - 1)
                new_val = random.randint(0, limit_val)
                new_set[idx] = new_val
                
        # Ensure 0 is in the set? The task says "containing 0".
        # If we removed 0, we must add it back or reject.
        if 0 not in new_set:
            if len(new_set) > 1:
                # Replace a random non-zero element with 0? Or just add 0?
                # Adding 0 increases size. Replacing keeps size.
                # Let's replace a random element with 0 if 0 is missing.
                idx = random.randint(0, len(new_set) - 1)
                new_set[idx] = 0
            else:
                new_set = [0]
                
        # Make it a set (remove duplicates)
        new_set = list(set(new_set))
        
        # Evaluate
        new_score = get_score(new_set)
        
        if new_score > current_score:
            current_score = new_score
            current_set = new_set
            if current_score > best_score:
                best_score = current_score
                best_set = current_set
                # Update limit_val if max increased significantly
                new_max = max(current_set)
                if new_max > limit_val:
                    limit_val = int(new_max * 1.5)
        else:
            # Simulated Annealing step?
            # For simplicity and stability, use Hill Climbing (reject worse moves)
            # This might get stuck in local optima, but is safe.
            pass
            
    return best_set
