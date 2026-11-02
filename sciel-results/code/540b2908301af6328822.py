"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _score_set(U):
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    n_sums = len(sums)
    n_diffs = len(diffs)
    if n_sums == 0 or n_diffs == 0:
        return -1
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1 + math.log(ratio) / denom

def _local_search(U, deadline, max_iter=1000):
    """Local search: add, remove, or swap elements to improve score."""
    U = sorted(set(U))
    current_score = _score_set(U)
    if current_score < 0:
        return U, current_score
    
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
        improved = False
        max_u = max(U)
        
        # Try adding an element
        for new_elem in range(1, max_u + 12):
            if new_elem in U:
                continue
            new_U = sorted(U + [new_elem])
            s = _score_set(new_U)
            if s > current_score + 1e-12:
                U = new_U
                current_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing an element (keep 0 and at least one other)
        for i in range(1, len(U)):
            new_U = U[:i] + U[i+1:]
            if len(new_U) < 2:
                continue
            s = _score_set(new_U)
            if s > current_score + 1e-12:
                U = new_U
                current_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping an element
        for i in range(1, len(U)):
            for new_elem in range(1, max_u + 18):
                if new_elem in U:
                    continue
                new_U = sorted(U[:i] + [new_elem] + U[i+1:])
                s = _score_set(new_U)
                if s > current_score + 1e-12:
                    U = new_U
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Systematic search for size 3
    for a in range(1, 20):
        for b in range(a + 1, 35):
            if time.time() > deadline - 5:
                return best_set
            U = [0, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Systematic search for size 4
    for a in range(1, 15):
        for b in range(a + 1, 25):
            for c in range(b + 1, 40):
                if time.time() > deadline - 5:
                    return best_set
                U = [0, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Local search from best found
    if time.time() < deadline - 15:
        improved_set, improved_score = _local_search(best_set, deadline, max_iter=2000)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Try local search from several promising starting points
    starting_points = [
        [0, 1, 4, 6],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 2, 5, 9],
        [0, 1, 5, 8],
        [0, 1, 4, 7],
        [0, 2, 4, 7],
        [0, 1, 3, 5],
    ]
    
    for start in starting_points:
        if time.time() > deadline - 5:
            break
        s = _score_set(start)
        if s > best_score - 0.08:
            improved_set, improved_score = _local_search(start, deadline, max_iter=1000)
            if improved_score > best_score:
                best_set = improved_set
                best_score = improved_score
    
    if time.time() > deadline - 1:
        return best_set
    
    return best_set

# EVOLVE-BLOCK-END
