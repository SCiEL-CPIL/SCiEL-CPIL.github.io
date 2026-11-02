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
    # Compute U+U and U-U
    # For efficiency, use numpy
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
    """Generate a Golomb ruler with n marks using greedy approach."""
    if n <= 1:
        return [0]
    marks = [0]
    for i in range(1, n):
        # Find the smallest position that doesn't create duplicate differences
        pos = 0
        while True:
            pos += 1
            valid = True
            # Check differences with all existing marks
            diffs = set()
            for m in marks:
                d = pos - m
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if valid:
                # Also check that this doesn't create duplicate differences among existing marks
                # Actually we need to check all pairwise differences
                all_diffs = set()
                for j in range(len(marks) + 1):
                    if j == len(marks):
                        mj = pos
                    else:
                        mj = marks[j]
                    for k in range(j + 1, len(marks) + 1):
                        if k == len(marks):
                            mk = pos
                        else:
                            mk = marks[k]
                        d = mk - mj
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                    if not valid:
                        break
            if valid:
                marks.append(pos)
                break
    return marks


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
        # Choose a random position between 0 and max_u + some margin
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


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with a Golomb ruler as seed
    # Try different sizes to find a good starting point
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Generate Golomb rulers of increasing size
    for n in range(3, 30):
        if time.time() > deadline - 60:  # Leave 60 seconds for local search
            break
        
        try:
            gr = _golomb_ruler(n)
        except Exception:
            continue
        
        if not gr or 0 not in gr:
            continue
        
        gr = sorted(gr)
        score = _compute_score(gr)
        
        if score > best_score:
            best_score = score
            best_U = list(gr)
    
    # Local search from the best Golomb ruler
    if time.time() < deadline - 30:  # Leave 30 seconds margin
        U, score = _local_search(best_U, max_iterations=10000, time_limit=60)
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
            
            U, score = _local_search(U, max_iterations=2000, time_limit=15)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
