"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score for a set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using greedy construction."""
    if n <= 1:
        return [0]
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
    }
    if n in known:
        return known[n]
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler


def _local_search(U, max_iterations=5000, time_limit=10.0):
    """Local search to improve the set U."""
    U = sorted(set(U))
    n = len(U)
    max_u = U[-1]
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    
    # Try adding new elements
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try adding a new element
        margin = max(10, max_u // 5)
        new_val = random.randint(0, max_u + margin)
        if new_val in U:
            continue
        
        new_U = sorted(U + [new_val])
        new_score = _compute_score(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
            U = new_U
            max_u = U[-1]
    
    return best_U, best_score


def _local_search_swap(U, max_iterations=5000, time_limit=10.0):
    """Local search with swap operations."""
    U = sorted(set(U))
    n = len(U)
    max_u = U[-1]
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try swapping one element
        if n > 1:
            i = random.randint(1, n - 1)  # Don't swap 0
            old_val = U[i]
            margin = max(10, max_u // 5)
            new_val = random.randint(0, max_u + margin)
            if new_val != old_val and new_val not in U:
                new_U = list(U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                new_score = _compute_score(new_U)
                
                if new_score > best_score:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
                    max_u = U[-1]
    
    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit for safety
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Generate Golomb rulers of various sizes
    for n in range(3, 25):
        if time.time() > deadline - 60:
            break
        
        try:
            U = _golomb_ruler(n)
        except Exception:
            continue
        
        if not U or 0 not in U:
            continue
        
        U = sorted(U)
        score = _compute_score(U)
        
        if score > best_score:
            best_score = score
            best_U = list(U)
    
    # Try some random sets
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 50:
            break
        n = random.randint(3, 20)
        max_val = random.randint(10, 200)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        score = _compute_score(U)
        
        if score > best_score:
            best_score = score
            best_U = list(U)
    
    # Local search from the best seed
    if time.time() < deadline - 20:
        U, score = _local_search(best_U, max_iterations=15000, time_limit=30)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Try swap-based local search
    if time.time() < deadline - 15:
        U, score = _local_search_swap(best_U, max_iterations=15000, time_limit=25)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Try a few more local search rounds with different random seeds
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            U = list(best_U)
            # Remove a random element (not 0)
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                U.pop(idx)
            # Add a random element
            max_u = max(U) if U else 0
            new_val = random.randint(0, max_u + max(10, max_u // 3))
            if new_val not in U:
                U.append(new_val)
            
            U, score = _local_search(U, max_iterations=3000, time_limit=10)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
