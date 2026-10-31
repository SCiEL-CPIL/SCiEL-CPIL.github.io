import time
import math
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _score_set_np(U):
    """Compute the score for a given set U using numpy for speed."""
    if not U or U[0] != 0:
        return -1.0
    if len(U) < 2:
        return 0.0
        
    U_arr = np.array(U, dtype=np.int64)
    maxU = U_arr[-1]
    if maxU == 0:
        return 0.0
    
    # Compute sum set and diff set
    # For n elements, outer sum/diff is n x n
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    n_sum = len(np.unique(sums))
    n_diff = len(np.unique(diffs))
    
    if n_sum == 0:
        return -1.0
        
    # score = 1 + log(n_diff / n_sum) / log(2 * maxU + 1)
    log_denom = math.log(2 * maxU + 1)
    if log_denom == 0:
        return 1.0
        
    return 1.0 + math.log(n_diff / n_sum) / log_denom

def _local_search(U, deadline, max_iter=500):
    """Local search with add, remove, and swap moves."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    
    current_score = _score_set_np(U)
    best_score = current_score
    best_set = U[:]
    
    n_iter = 0
    while time.time() < deadline and n_iter < max_iter:
        n_iter += 1
        improved = False
        maxU = max(U)
        
        # 1. Try adding a new element
        # Range: 1 to maxU + 20
        candidates_add = list(range(1, maxU + 21))
        random.shuffle(candidates_add)
        
        for new_elem in candidates_add:
            if new_elem in U:
                continue
            new_U = sorted(U + [new_elem])
            s = _score_set_np(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_set = new_U
                U = new_U
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try removing an element (except 0)
        if len(U) > 2:
            candidates_remove = [x for x in U if x != 0]
            random.shuffle(candidates_remove)
            for elem in candidates_remove:
                new_U = [x for x in U if x != elem]
                s = _score_set_np(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    improved = True
                    break
        
        if improved:
            continue
            
        # 3. Try swapping an element
        if len(U) > 2:
            # Pick a random element to remove and a random new element to add
            removable = [x for x in U if x != 0]
            if removable:
                old_elem = random.choice(removable)
                # New element range: 1 to max(U) + 30
                new_elem = random.randint(1, max(U) + 30)
                if new_elem not in U:
                    new_U = sorted([x for x in U if x != old_elem] + [new_elem])
                    s = _score_set_np(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        improved = True
                        continue
        
        if not improved:
            # Stagnation: try a random jump
            if len(U) > 2:
                # Remove a random non-zero element
                removable = [x for x in U if x != 0]
                if removable:
                    old_elem = random.choice(removable)
                    new_elem = random.randint(1, max(U) + 50)
                    if new_elem not in U:
                        new_U = sorted([x for x in U if x != old_elem] + [new_elem])
                        s = _score_set_np(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = new_U
                            U = new_U
                            continue
            break
            
    return best_score, best_set

def construct():
    deadline = _START_TIME + _TIME_LIMIT
    
    # Initial candidates
    initial_sets = [
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 4, 9, 16, 25],
        [0, 2, 3, 5, 7, 11],
        [0, 1, 3, 4, 9, 11],
        [0, 1, 2, 5, 10, 20],
        [0, 1, 4, 5, 12, 15],
        [0, 1, 3, 6, 10, 15],
    ]
    
    best_score = -1.0
    best_set = [0, 1]
    
    # Run local search from each initial set
    for start_set in initial_sets:
        if time.time() > deadline - 5.0:
            break
        s, U = _local_search(start_set, deadline - 5.0, max_iter=200)
        if s > best_score:
            best_score = s
            best_set = U
        
    # Random restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a random initial set
        size = random.randint(3, 8)
        max_val = random.randint(20, 100)
        random_elems = random.sample(range(1, max_val + 1), size - 1)
        start_set = sorted([0] + random_elems)
        
        s, U = _local_search(start_set, deadline - 5.0, max_iter=200)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Final polish with more iterations
    if time.time() < deadline - 10.0:
        s, U = _local_search(best_set, deadline - 5.0, max_iter=1000)
        if s > best_score:
            best_score = s
            best_set = U
            
    return best_set
