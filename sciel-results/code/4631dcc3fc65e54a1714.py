import math
import time
import random
import numpy as np

def _score(U):
    """Compute the score for a set U."""
    U = sorted(list(set(U)))
    if 0 not in U:
        return 0.0
    if len(U) < 2:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    # Compute U+U and U-U
    U_arr = np.array(U, dtype=np.int64)
    
    # Sumset
    sums = U_arr[:, None] + U_arr[None, :]
    n_plus = len(np.unique(sums))
    
    # Difference set
    diffs = U_arr[:, None] - U_arr[None, :]
    n_minus = len(np.unique(diffs))
    
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

def _greedy_construct(target_size, max_span, time_limit=10.0):
    """
    Greedy constructive algorithm:
    Start with U = {0}. Iteratively add the element that maximizes the score.
    """
    start = time.time()
    U = [0]
    
    # Precompute what we need
    # For each candidate x, we need to compute new U+U and U-U sizes
    # This is O(n * span) per step, which is fine for small n and span
    
    for step in range(target_size - 1):
        if time.time() - start > time_limit:
            break
            
        best_x = None
        best_score = -1
        
        # Try all possible x from 1 to max_span
        # We can be smarter: try x values that are likely good
        # But for now, brute force over [1, max_span]
        
        U_set = set(U)
        
        for x in range(1, max_span + 1):
            if x in U_set:
                continue
                
            new_U = U + [x]
            new_U_sorted = sorted(new_U)
            
            # Quick check: is 0 still in U? Yes, since we only add.
            # Compute score
            s = _score(new_U_sorted)
            
            if s > best_score:
                best_score = s
                best_x = x
        
        if best_x is None:
            break
            
        U.append(best_x)
        U.sort()
        
    return U

def _local_search_numpy(U, max_iter=100, deadline=None):
    """
    Local search using NumPy for efficient score computation.
    Tries to move any element to any other value.
    """
    current_U = sorted(U)
    current_score = _score(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    n = len(current_U)
    max_val = max(current_U)
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            original_val = current_U[i]
            existing = set(current_U)
            existing.remove(original_val)
            
            upper_bound = max_val * 2
            if upper_bound > 300:
                upper_bound = 300
            
            # Try a subset of candidates to save time
            # First try neighbors, then random
            candidates = set()
            # Neighbors
            for dx in range(-5, 6):
                x = original_val + dx
                if 0 <= x <= upper_bound and x not in existing:
                    candidates.add(x)
            # Random
            for _ in range(20):
                x = random.randint(0, upper_bound)
                if x not in existing:
                    candidates.add(x)
            
            for x in candidates:
                new_U = list(current_U)
                new_U[i] = x
                new_U.sort()
                
                if 0 not in new_U:
                    continue
                
                new_score = _score(new_U)
                if new_score > best_score + 1e-9:
                    best_score = new_score
                    best_U = new_U
                    current_U = new_U
                    max_val = max(current_U)
                    improved = True
                    break
        
        if not improved:
            break
            
    return best_U, best_score

def _simulated_annealing(U, deadline):
    """
    Simulated annealing to escape local optima.
    """
    current_U = sorted(U)
    current_score = _score(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    n = len(current_U)
    max_val = max(current_U)
    
    T = 0.1  # Initial temperature
    cooling = 0.99
    
    while time.time() < deadline - 5:
        if T < 1e-6:
            break
            
        i = random.randint(0, n - 1)
        original_val = current_U[i]
        existing = set(current_U)
        existing.remove(original_val)
        
        upper_bound = max_val * 2
        if upper_bound > 300:
            upper_bound = 300
        
        x = random.randint(0, upper_bound)
        if x in existing:
            continue
        
        new_U = list(current_U)
        new_U[i] = x
        new_U.sort()
        
        if 0 not in new_U:
            continue
        
        new_score = _score(new_U)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_U = new_U
            current_score = new_score
            max_val = max(current_U)
            
            if current_score > best_score:
                best_score = current_score
                best_U = list(current_U)
        
        T *= cooling
        
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_U = [0, 1]
    best_score = _score(best_U)
    
    # 1. Greedy construction for various sizes and spans
    for target_size in range(3, 15):
        for max_span in [10, 20, 30, 50, 80]:
            if time.time() > deadline - 20:
                break
                
            time_per_greedy = min(5.0, (deadline - time.time() - 10) / 10)
            U = _greedy_construct(target_size, max_span, time_limit=time_per_greedy)
            
            if len(U) >= 2:
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_U = list(U)
                
                # Local search from this point
                if time.time() < deadline - 10:
                    improved_U, improved_score = _local_search_numpy(U, max_iter=50, deadline=deadline)
                    if improved_score > best_score:
                        best_score = improved_score
                        best_U = improved_U
    
    # 2. Structured seeds
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
        # Powers of 2
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 2, 4, 8, 16, 32, 64],
        # Geometric
        [0, 1, 3, 9, 27, 81],
        [0, 1, 3, 9, 27, 81, 243],
        # Mixed
        [0, 1, 3, 7, 15, 31, 63],
    ]
    
    for U in candidates:
        if time.time() > deadline - 15:
            break
            
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
            
        if time.time() < deadline - 10:
            improved_U, improved_score = _local_search_numpy(U, max_iter=80, deadline=deadline)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # 3. Random search
    if time.time() < deadline - 15:
        random.seed(42)
        for _ in range(300):
            if time.time() > deadline - 10:
                break
            
            n = random.randint(3, 12)
            max_span = random.randint(10, 100)
            
            if max_span < n:
                continue
            
            U = [0]
            used = {0}
            for _ in range(n - 1):
                x = random.randint(1, max_span)
                if x not in used:
                    U.append(x)
                    used.add(x)
            
            if len(U) < 2:
                continue
                
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
                
            if time.time() < deadline - 10:
                improved_U, improved_score = _local_search_numpy(U, max_iter=30, deadline=deadline)
                if improved_score > best_score:
                    best_score = improved_score
                    best_U = improved_U
    
    # 4. Simulated annealing from best
    if time.time() < deadline - 10:
        improved_U, improved_score = _simulated_annealing(best_U, deadline)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # 5. Final local search polish
    if time.time() < deadline - 5:
        improved_U, improved_score = _local_search_numpy(best_U, max_iter=100, deadline=deadline)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    return best_U
