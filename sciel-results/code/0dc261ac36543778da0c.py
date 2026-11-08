import time
import math
import random

def score_set(U):
    if not U or 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    # Compute U+U and U-U
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    if num <= 1:
        return 1.0
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 1.0
    return 1 + math.log(num) / den

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial best
    # Start with a geometric progression which often performs well
    best_set = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_score = score_set(best_set)
    
    # Local search function
    def local_search(current, current_score, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping each element with a new value
            # Range of values to try: 0 to 1000 (widened from 30)
            for i in range(len(current)):
                old_val = current[i]
                # Try values in range 0..1000
                # To speed up, we can sample or iterate. Let's iterate but with a larger range.
                # If n is small, 1000 is fine. If n is large, it might be slow.
                # Let's cap n to 12 for local search to keep it fast.
                for new_val in range(0, 1001):
                    if new_val == old_val:
                        continue
                    # Ensure 0 is in set
                    if old_val == 0 and new_val != 0:
                        continue
                    if 0 not in current and new_val != 0:
                        continue
                    
                    # Create new set
                    new_set = current[:]
                    new_set[i] = new_val
                    # Remove duplicates and sort
                    new_set = sorted(set(new_set))
                    
                    if 0 not in new_set:
                        continue
                    
                    # Check size
                    if len(new_set) < 3 or len(new_set) > 12:
                        continue
                    
                    s = score_set(new_set)
                    if s > current_score + 1e-9:
                        current_score = s
                        current = new_set
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_set = new_set[:]
                
                # Check time periodically inside the inner loop to avoid long stalls
                if time.time() > deadline:
                    return current, current_score
        
        # Try adding an element
        if len(current) < 12:
            for new_val in range(0, 1001):
                if new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                if 0 not in new_set:
                    continue
                s = score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]
                
                if time.time() > deadline:
                    return current, current_score
        
        # Try removing an element (if size > 3)
        if len(current) > 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                new_set = current[:i] + current[i+1:]
                s = score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]
                
                if time.time() > deadline:
                    return current, current_score

        return current, current_score

    # Global search: Random restarts
    # We will try different initial configurations
    # 1. Geometric
    # 2. Arithmetic
    # 3. Random
    # 4. Sparse random
    
    # Run local search on the initial best
    current = best_set[:]
    current_score = best_score
    current, current_score = local_search(current, current_score, deadline)
    
    # Try a few random restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate a random set
        # Size 5 to 10
        size = random.randint(5, 10)
        # Max value up to 500
        max_val = random.randint(50, 500)
        
        # Ensure 0 is included
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        current = [0] + candidates[:size-1]
        current = sorted(current)
        
        # Initial score
        current_score = score_set(current)
        
        # Local search
        current, current_score = local_search(current, current_score, deadline)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current[:]
            
    return best_set
