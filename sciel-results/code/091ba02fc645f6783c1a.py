import math
import itertools
import numpy as np
import time

def score_set(U):
    """Compute score using NumPy for speed."""
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    U_arr = np.array(U, dtype=np.int64)
    n = len(U_arr)
    
    # Compute all pairwise sums and differences using broadcasting
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    size_sum = len(np.unique(sums))
    size_diff = len(np.unique(diffs))
    
    if size_sum == 0:
        return 0
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
    return 1 + math.log(ratio) / denom

def generate_golomb_ruler(n, max_val=100):
    """Generate a Golomb ruler of length n with values in [0, max_val]."""
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    # Use a greedy/backtracking approach for small n
    def backtrack(pos, current, used_diffs):
        if pos == n:
            return current[:]
        
        start = 0 if pos == 0 else current[-1] + 1
        for v in range(start, max_val + 1):
            # Check if v creates duplicate differences
            valid = True
            for i in range(len(current)):
                d = v - current[i]
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                new_diffs = set(used_diffs)
                for i in range(len(current)):
                    new_diffs.add(v - current[i])
                result = backtrack(pos + 1, current + [v], new_diffs)
                if result is not None:
                    return result
        return None
    
    result = backtrack(0, [0], set())
    if result is None:
        # Fallback: simple arithmetic progression
        return list(range(n))
    return result

def local_search(U, max_val, iterations=5000, deadline=None):
    """Local search to improve the score by swapping elements."""
    U = sorted(set(U))
    if 0 not in U:
        U = sorted(set([0] + U))
    
    current_score = score_set(U)
    best_score = current_score
    best_set = U[:]
    
    for i in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try swapping a random element with a random new value
        idx = np.random.randint(0, len(U))
        new_val = np.random.randint(1, max_val + 1)
        
        if new_val in U:
            continue
        
        old_val = U[idx]
        new_U = U[:idx] + [new_val] + U[idx+1:]
        new_U = sorted(set(new_U))
        
        if 0 not in new_U or len(new_U) != len(U):
            continue
        
        new_score = score_set(new_U)
        
        if new_score > current_score + 1e-12:
            current_score = new_score
            U = new_U
            if new_score > best_score:
                best_score = new_score
                best_set = U[:]
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Generate Golomb rulers as starting points for various sizes
    for n in range(3, 12):
        if time.time() > deadline:
            break
        
        # Try to generate a Golomb ruler
        max_val = 50 if n <= 6 else 100
        golomb = generate_golomb_ruler(n, max_val)
        
        if len(golomb) != n:
            continue
        
        # Score the Golomb ruler
        s = score_set(golomb)
        if s > best_score:
            best_score = s
            best_set = golomb[:]
        
        # Do local search from this starting point
        if time.time() > deadline:
            break
        
        # Limit iterations based on remaining time
        remaining = deadline - time.time()
        iters = min(2000, int(remaining * 200))
        if iters < 100:
            continue
        
        improved_set, improved_score = local_search(
            golomb, 
            max_val=max_val + 20, 
            iterations=iters,
            deadline=deadline
        )
        
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set[:]
    
    # Also try some random starts
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline:
                break
            n = np.random.randint(3, 10)
            max_val = np.random.randint(20, 80)
            random_set = [0] + sorted(set(np.random.choice(range(1, max_val+1), size=n-1, replace=False).tolist()))
            random_set = sorted(random_set)
            
            s = score_set(random_set)
            if s > best_score:
                best_score = s
                best_set = random_set[:]
            
            if time.time() > deadline:
                break
            
            remaining = deadline - time.time()
            iters = min(1000, int(remaining * 100))
            if iters < 50:
                continue
            
            improved_set, improved_score = local_search(
                random_set,
                max_val=max_val + 10,
                iterations=iters,
                deadline=deadline
            )
            
            if improved_score > best_score:
                best_score = improved_score
                best_set = improved_set[:]
    
    return best_set
