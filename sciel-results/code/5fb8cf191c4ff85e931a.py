import math
import time
import random

def _score(U):
    U = sorted(list(set(U)))
    if 0 not in U:
        return 0.0
    if len(U) < 2:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    plus_set = set()
    minus_set = set()
    for a in U:
        for b in U:
            plus_set.add(a + b)
            minus_set.add(a - b)
            
    n_plus = len(plus_set)
    n_minus = len(minus_set)
    
    if n_plus == 0:
        return 0.0
        
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
        
    try:
        val = 1.0 + math.log(ratio) / math.log(2 * mx + 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return val

def _local_search(U, max_iter=500, deadline=None):
    """
    Local search without Sidon constraint.
    Tries to move any element to any other value in a reasonable range.
    """
    current_U = sorted(U)
    current_score = _score(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    n = len(current_U)
    max_val = max(current_U)
    # Search space: values from 0 to max_val * 1.5
    # But we want to keep span small.
    # Let's restrict moves to [0, max_val * 2]
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            original_val = current_U[i]
            # Try moving element i to a new value
            # Range: 0 to max_val * 2
            # Skip values already in U (except original)
            existing = set(current_U)
            existing.remove(original_val)
            
            # To be efficient, don't iterate all 0..max*2.
            # Try neighbors and random spots?
            # Or just iterate 0..max_val*2 if max_val is small (< 100).
            # For n=6, max_val ~ 20-30. 0..60 is small.
            
            upper_bound = max_val * 2
            if upper_bound > 200:
                upper_bound = 200 # Cap search space
            
            for x in range(0, upper_bound + 1):
                if x in existing:
                    continue
                
                new_U = list(current_U)
                new_U[i] = x
                new_U.sort()
                
                # Check if 0 is still in U (it is, unless i=0 and x!=0)
                # If i==0, we must ensure 0 is in new_U.
                # Since we are replacing U[0]=0 with x, if x!=0, 0 is lost.
                # So if i==0, x must be 0? No, we can move 0.
                # But U must contain 0.
                # So if i==0, we can only move to 0? No, we can swap.
                # Actually, just check if 0 in new_U.
                
                if 0 not in new_U:
                    continue
                
                new_score = _score(new_U)
                if new_score > best_score + 1e-9:
                    best_score = new_score
                    best_U = new_U
                    current_U = new_U
                    max_val = max(current_U)
                    improved = True
                    break # Restart loop with new current_U
        
        if not improved:
            break
            
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Known optimal Golomb rulers for small n
    # These provide good initial structures
    candidates = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 28],
        [0, 1, 4, 10, 18, 24, 28, 34],
        [0, 1, 4, 10, 18, 24, 28, 34, 43],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65, 77],
    ]
    
    best_U = candidates[0]
    best_score = _score(best_U)
    
    for U in candidates:
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
            
        # Local search from this candidate
        if time.time() < deadline - 5:
            improved_U, improved_score = _local_search(U, max_iter=100, deadline=deadline)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
                
    # Random search
    # Focus on sizes 4-10, span < 50
    if time.time() < deadline - 10:
        for _ in range(200):
            n = random.randint(4, 10)
            max_span = random.randint(10, 60)
            # Generate random set of size n in [0, max_span] containing 0
            if max_span < n:
                continue
            U = [0]
            for _ in range(n - 1):
                x = random.randint(1, max_span)
                if x not in U:
                    U.append(x)
            
            if len(U) < 2:
                continue
                
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
                
            if time.time() < deadline - 5:
                improved_U, improved_score = _local_search(U, max_iter=50, deadline=deadline)
                if improved_score > best_score:
                    best_score = improved_score
                    best_U = improved_U
                    
    return best_U
