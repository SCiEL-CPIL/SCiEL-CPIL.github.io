import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s buffer
    
    def score(U):
        if not U:
            return 0
        U = sorted(set(U))
        if 0 not in U:
            U.append(0)
            U = sorted(U)
        
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        # Calculate sumset and diffset sizes
        # Using sets for efficiency for small U
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        
        s_size = len(sums)
        d_size = len(diffs)
        
        if s_size == 0:
            return 0
            
        ratio = d_size / s_size
        # log base 2 or e? Prompt says log. Usually natural log in math, but ratio is same for any base.
        # Denominator is log(2 max U + 1)
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        val = 1 + math.log(ratio) / denom
        return val

    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 2, 4, 8],
        [0, 1, 4, 16, 64],
        [0, 1, 3, 9, 27],
    ]
    
    best_score = -1
    best_set = [0, 1, 3]
    
    for c in candidates:
        s = score(c)
        if s > best_score:
            best_score = s
            best_set = list(c)
            
    # Local Search
    current_set = list(best_set)
    current_score = best_score
    
    # Try adding elements
    # Range for new elements: up to 200
    max_val = max(current_set)
    
    iterations = 0
    while time.time() < deadline:
        improved = False
        
        # 1. Try adding a new element
        # Search space: 0 to max_val * 2
        for x in range(0, max_val * 2 + 10):
            if x in current_set:
                continue
            new_set = current_set + [x]
            s = score(new_set)
            if s > current_score:
                current_score = s
                current_set = new_set
                max_val = max(current_set)
                improved = True
                break # Restart search with new best
        
        # 2. Try removing an element (if size > 2)
        if len(current_set) > 2:
            for i in range(len(current_set)):
                new_set = [x for j, x in enumerate(current_set) if j != i]
                s = score(new_set)
                if s > current_score:
                    current_score = s
                    current_set = new_set
                    max_val = max(current_set) if current_set else 0
                    improved = True
                    break
        
        # 3. Try swapping an element
        if len(current_set) > 2:
            for i in range(len(current_set)):
                old_val = current_set[i]
                # Try replacing old_val with something else
                # Small range around old_val
                for new_val in range(max(0, old_val - 5), old_val + 10):
                    if new_val == old_val or new_val in current_set:
                        continue
                    new_set = list(current_set)
                    new_set[i] = new_val
                    s = score(new_set)
                    if s > current_score:
                        current_score = s
                        current_set = new_set
                        max_val = max(current_set)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Stagnation: try a random restart or larger jump
            # Add a random element
            if len(current_set) < 20:
                rand_x = random.randint(0, max_val * 3)
                if rand_x not in current_set:
                    new_set = current_set + [rand_x]
                    s = score(new_set)
                    if s > current_score:
                        current_score = s
                        current_set = new_set
                        max_val = max(current_set)
                    else:
                        # If no improvement in a while, break to save time
                        # Or just continue
                        pass
            
            iterations += 1
            if iterations > 100:
                break
                
    return current_set
