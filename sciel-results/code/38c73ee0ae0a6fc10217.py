import time
import math
import random
import numpy as np

def compute_score(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    
    # Compute sumset and difference set efficiently
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    sum_set = set(sums.flatten().tolist())
    diff_set = set(diffs.flatten().tolist())
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def fast_compute_score(U):
    """Faster score computation using numpy for larger sets."""
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start = time.time()
    deadline = start + 130  # Safety margin before 150s limit
    
    best_score = 0
    best_U = [0, 1, 3]
    
    # Phase 1: Try structured families
    # Powers of 2
    for n in range(2, 25):
        U = [0] + [2**i for i in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Mersenne numbers
    for n in range(2, 25):
        U = [0] + [2**i - 1 for i in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Powers of 3
    for n in range(2, 15):
        U = [0] + [3**i for i in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Fibonacci-like
    for n in range(2, 20):
        fibs = [0, 1]
        for i in range(2, n):
            fibs.append(fibs[-1] + fibs[-2])
        U = fibs
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Arithmetic progressions
    for a in range(1, 20):
        for n in range(2, 20):
            U = [a * i for i in range(n)]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Geometric sequences with various bases
    for base in [2, 3, 4, 5]:
        for n in range(2, 15):
            U = [0] + [base**i for i in range(1, n)]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Phase 2: Random search with local refinement
    random.seed(42)
    
    # Try random sets of various sizes and ranges
    for trial in range(3000):
        if time.time() > deadline - 10:
            break
        
        n = random.randint(3, 20)
        max_val = random.randint(10, 2000)
        U = set([0])
        for _ in range(n - 1):
            U.add(random.randint(1, max_val))
        U = sorted(U)
        
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Phase 3: Local search / hill climbing
    if time.time() < deadline - 15:
        U = list(best_U)
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            improved = False
            U = U[:]
            
            # Try various mutations
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                
                U_new = U[:]
                op = random.choice(['add', 'remove', 'modify', 'swap'])
                
                if op == 'add' and len(U_new) < 30:
                    # Add a new element
                    max_curr = max(U_new)
                    new_val = random.randint(1, 2 * max_curr + 50)
                    if new_val not in U_new:
                        U_new.append(new_val)
                        U_new = sorted(set(U_new))
                
                elif op == 'remove' and len(U_new) > 2:
                    # Remove an element (not 0)
                    idx = random.randint(1, len(U_new) - 1)
                    U_new.pop(idx)
                    U_new = sorted(set(U_new))
                
                elif op == 'modify' and len(U_new) > 1:
                    # Modify an element (not 0)
                    idx = random.randint(1, len(U_new) - 1)
                    old_val = U_new[idx]
                    new_val = max(1, old_val + random.randint(-max(1, old_val//2), max(1, old_val//2)))
                    if new_val not in U_new:
                        U_new[idx] = new_val
                        U_new = sorted(set(U_new))
                
                elif op == 'swap' and len(U_new) > 2:
                    # Swap two elements
                    i, j = random.sample(range(1, len(U_new)), 2)
                    U_new[i], U_new[j] = U_new[j], U_new[i]
                    U_new = sorted(set(U_new))
                
                if len(U_new) >= 2:
                    s = compute_score(U_new)
                    if s > best_score:
                        best_score = s
                        best_U = U_new[:]
                        U = U_new[:]
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
    
    # Phase 4: Final polish - try to optimize the best set found
    if time.time() < deadline - 5:
        U = list(best_U)
        
        # Try adding elements at strategic positions
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            
            U_new = U[:]
            max_curr = max(U_new)
            
            # Try adding elements near existing ones or at strategic gaps
            candidates = []
            for x in U_new:
                candidates.append(x + 1)
                candidates.append(x - 1)
                candidates.append(2 * x)
                candidates.append(x // 2)
            candidates.append(max_curr + 1)
            candidates.append(max_curr * 2)
            candidates = [c for c in candidates if c > 0 and c not in U_new]
            
            if candidates:
                new_val = random.choice(candidates)
                U_new.append(new_val)
                U_new = sorted(set(U_new))
                
                s = compute_score(U_new)
                if s > best_score:
                    best_score = s
                    best_U = U_new[:]
                    U = U_new[:]
        
        # Try removing elements
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            if len(U) <= 2:
                break
            
            idx = random.randint(1, len(U) - 1)
            U_new = U[:]
            U_new.pop(idx)
            U_new = sorted(set(U_new))
            
            s = compute_score(U_new)
            if s > best_score:
                best_score = s
                best_U = U_new[:]
                U = U_new[:]
    
    return best_U
