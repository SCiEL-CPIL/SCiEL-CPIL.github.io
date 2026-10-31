import random
import time
import math

def calculate_score(U):
    if not U:
        return 0.0
    U_set = set(U)
    if 0 not in U_set:
        return 0.0 # Must contain 0
    
    max_u = max(U_set)
    if max_u == 0:
        return 1.0 # log(1)/log(1) is undefined, but limit is 1? 
                   # Actually 2*0+1 = 1, log(1)=0. Division by zero.
                   # But U={0} -> U+U={0}, U-U={0}. Ratio 1. Log 1 = 0.
                   # Score 1 + 0/0. Let's assume 1.0 for single element.
    
    sum_set = set()
    diff_set = set()
    
    # Optimization: use list for speed if small, but set is fine for small N
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    
    if n_sum == 0:
        return 0.0
        
    ratio = n_diff / n_sum
    if ratio <= 1.0:
        # If ratio < 1, log is negative, score < 1. 
        # We want to maximize, so we prefer ratio > 1.
        # But mathematically we just compute.
        pass
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    score = 1.0 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = 10.0 # Use 10 seconds for search to be safe
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 4, 6],
        [0, 2, 3, 5, 7]
    ]
    
    best_score = -1.0
    best_set = [0, 1, 3]
    
    # Evaluate initial candidates
    for c in candidates:
        s = calculate_score(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    # Local Search
    # We will try to improve best_set
    current_set = best_set[:]
    current_score = best_score
    
    # Set of elements to try adding/swapping
    # Keep it small to ensure speed
    max_val = 50
    max_size = 10
    
    while time.time() - start_time < deadline:
        improved = False
        
        # Generate neighbors
        neighbors = []
        
        # 1. Try adding a new element
        if len(current_set) < max_size:
            for x in range(1, max_val + 1):
                if x not in current_set:
                    new_set = current_set + [x]
                    new_set.sort()
                    neighbors.append(new_set)
        
        # 2. Try removing an element (keep at least 1 element? No, must contain 0. So remove non-zero)
        for i in range(len(current_set)):
            if current_set[i] != 0:
                new_set = current_set[:i] + current_set[i+1:]
                neighbors.append(new_set)
        
        # 3. Try swapping an element
        for i in range(len(current_set)):
            if current_set[i] == 0: continue
            for x in range(1, max_val + 1):
                if x not in current_set:
                    new_set = current_set[:]
                    new_set[i] = x
                    new_set.sort()
                    neighbors.append(new_set)
        
        # Evaluate neighbors
        for n in neighbors:
            s = calculate_score(n)
            if s > current_score:
                current_score = s
                current_set = n[:]
                improved = True
                # If we found an improvement, we could break and restart from here,
                # or continue to see if there's a better one in the current neighborhood.
                # For efficiency, let's just take the first improvement and loop again.
                # But evaluating all neighbors is better to find the steepest ascent.
                # Given the small size, let's find the best neighbor in this batch.
        
        if not improved:
            # Try random restarts if stuck
            if len(current_set) < max_size:
                # Randomly add a small element
                x = random.randint(1, max_val)
                if x not in current_set:
                    new_set = current_set + [x]
                    new_set.sort()
                    s = calculate_score(new_set)
                    if s > current_score:
                        current_score = s
                        current_set = new_set
                        improved = True
            
            if not improved:
                # Try a completely random small set
                size = random.randint(3, 6)
                rand_set = {0}
                while len(rand_set) < size:
                    rand_set.add(random.randint(1, max_val))
                rand_list = sorted(rand_set)
                s = calculate_score(rand_list)
                if s > current_score:
                    current_score = s
                    current_set = rand_list
                    improved = True
            
            if not improved:
                # No improvement found in this round. 
                # To avoid infinite loop on local maxima, we can break or try larger max_val.
                # For simplicity, break if no improvement after several tries?
                # The prompt says "Keep an EXPLICIT deadline guard".
                # We are within deadline. Let's just break to save time if stuck.
                # Or increase max_val slightly?
                # Let's just break to ensure we return a valid result quickly.
                break
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]
            
    return best_set
