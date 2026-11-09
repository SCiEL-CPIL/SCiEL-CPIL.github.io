# EVOLVE-BLOCK-START
import time
import random
from math import log
import numpy as np

def _compute_score(U):
    """Compute the score for a given set U using NumPy vectorization."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute U+U
    # This creates an array of all sums
    sumset_arr = U_arr[:, None] + U_arr[None, :]
    sumset = np.unique(sumset_arr)
    
    # Compute U-U
    diffset_arr = U_arr[:, None] - U_arr[None, :]
    diffset = np.unique(diffset_arr)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
    
    score = 1 + log(ratio) / denominator
    return score

def _local_search(U, max_iterations=50000, timeout=10.0):
    """Local search to improve the set."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        
        # Try adding a new element
        candidates = []
        # Try small additions near existing elements
        for x in list(U_set):
            candidates.append(x + 1)
            candidates.append(max(0, x - 1))
        # Try powers of 2 or 3
        p2 = 1
        while p2 <= 2 * max_u + 50:
            candidates.append(p2)
            p2 *= 2
        p3 = 1
        while p3 <= 2 * max_u + 50:
            candidates.append(p3)
            p3 *= 3
        # Random candidates
        for _ in range(20):
            candidates.append(random.randint(0, 2 * max_u + 100))
        
        random.shuffle(candidates)
        improved = False
        for cand in candidates:
            if time.time() - start_time > timeout:
                break
            new_set = U_set | {cand}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-9:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removing an element
            removable = [x for x in current if x != 0]
            random.shuffle(removable)
            for cand in removable[:30]:
                if time.time() - start_time > timeout:
                    break
                new_set = set(current) - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-9:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                    improved = True
                    break
        
        if not improved and i > 100:
            # Try a bigger jump
            jump_size = random.randint(50, max(100, max_u // 10))
            new_elem = random.randint(max_u, max_u + jump_size)
            new_set = set(current) | {new_elem}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-9:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Start with a good initial set
    best_result = [0, 1, 3]
    best_score = _compute_score(best_result)
    
    # Try several initial structures
    initial_sets = []
    
    # Powers of 2 based
    p2 = [0]
    v = 1
    while v <= 10000:
        p2.append(v)
        v *= 2
    initial_sets.append(p2)
    
    # Powers of 3 based
    p3 = [0]
    v = 1
    while v <= 10000:
        p3.append(v)
        v *= 3
    initial_sets.append(p3)
    
    # Mixed: 0, 1, then exponential
    mixed = [0, 1]
    v = 3
    while v <= 10000:
        mixed.append(v)
        v *= 2
    initial_sets.append(mixed)
    
    # Try a set with dense low part and sparse high part
    dense_sparse = list(range(20))
    v = 40
    while v <= 10000:
        dense_sparse.append(v)
        v *= 2
    initial_sets.append(dense_sparse)
    
    # Small structured sets known to be good
    initial_sets.append([0, 1, 3])
    initial_sets.append([0, 1, 3, 9])
    initial_sets.append([0, 1, 3, 9, 27])
    initial_sets.append([0, 1, 2, 4])
    initial_sets.append([0, 1, 3, 6])
    
    for init_set in initial_sets:
        if time.time() > deadline - 20:
            break
        
        # Do local search on this initial set
        improved, score = _local_search(init_set, max_iterations=100000, timeout=min(15.0, deadline - time.time() - 5))
        
        if score > best_score:
            best_score = score
            best_result = improved
    
    # Final check
    final_score = _compute_score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 3]
    
    return best_result
# EVOLVE-BLOCK-END
