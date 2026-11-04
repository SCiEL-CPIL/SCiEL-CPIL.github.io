import time
import random
import numpy as np
from itertools import combinations

def _score_set(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return -1.0
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return -1.0
    denom = np.log(2 * max_u + 1)
    if denom <= 0:
        return -1.0
    score = 1.0 + np.log(ratio) / denom
    return score

def _score_set_fast(U_arr):
    """Faster score computation using numpy arrays."""
    U = U_arr.tolist()
    if not U or 0 not in U:
        return -1.0
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    # Use numpy for sumset
    U_arr = np.array(U)
    sumset = set()
    for a in U:
        sumset.update(a + U_arr)
    
    diffset = set()
    for a in U:
        diffset.update(a - U_arr)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return -1.0
    denom = np.log(2 * max_u + 1)
    if denom <= 0:
        return -1.0
    score = 1.0 + np.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    random.seed(42)
    
    # Start with a known good structure: use sets with geometric or structured spacing
    # Try several initial candidates
    candidates = []
    
    # Candidate 1: Small dense set
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Candidate 2: Spaced set
    candidates.append([0, 1, 3, 7, 15, 31, 63, 127, 255])
    
    # Candidate 3: Mixed
    candidates.append([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    
    # Candidate 4: From paper hint - try size ~2003
    # Use a structured set
    for n in [50, 100, 200, 500, 1000, 2000]:
        # Try arithmetic progression
        U = list(range(n))
        candidates.append(U)
        
        # Try sparse set
        if n <= 200:
            U = [i * 2 for i in range(n)]
            candidates.append(U)
    
    # Evaluate initial candidates
    best_score = -1.0
    best_U = [0, 1, 3]
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Local search: try adding/removing elements
    current_U = best_U[:]
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        U = current_U[:]
        U_set = set(U)
        max_u = max(U)
        n = len(U)
        
        # Try adding a random element
        if random.random() < 0.7:
            # Add element in range [0, 2*max_u + 10]
            new_elem = random.randint(0, max(1, 2 * max_u + 10))
            if new_elem not in U_set:
                U.append(new_elem)
        else:
            # Remove a random non-zero element
            removable = [x for x in U if x != 0]
            if removable:
                rem_elem = random.choice(removable)
                U.remove(rem_elem)
        
        if len(U) < 2 or 0 not in U:
            no_improve_count += 1
            continue
        
        s = _score_set(U)
        if s > current_score:
            current_score = s
            current_U = U[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Final polish: try small perturbations
    if time.time() < deadline:
        U = current_U[:]
        U_set = set(U)
        max_u = max(U)
        
        for _ in range(100):
            if time.time() > deadline:
                break
            # Try adding nearby elements
            candidates_to_add = []
            for x in U:
                for delta in [-2, -1, 1, 2, 3, 5]:
                    y = x + delta
                    if 0 <= y <= 2 * max_u + 50 and y not in U_set:
                        candidates_to_add.append(y)
            
            if candidates_to_add:
                new_elem = random.choice(candidates_to_add)
                U_test = U + [new_elem]
                s = _score_set(U_test)
                if s > current_score:
                    current_score = s
                    current_U = U_test[:]
                    U_set.add(new_elem)
                    max_u = max(current_U)
    
    return sorted(set(current_U))

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
