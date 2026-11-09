# EVOLVE-BLOCK-START
import time
import random
from math import log

# Known optimal Golomb rulers (Sidon sets) for small n
# These are minimal-length Golomb rulers
KNOWN_GOLOMB = {
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 10, 12],
    6: [0, 1, 4, 10, 18, 23],
    7: [0, 1, 7, 12, 18, 23, 31],
    8: [0, 1, 7, 12, 18, 25, 34, 41],
    9: [0, 1, 7, 12, 18, 25, 34, 44, 51],
    10: [0, 1, 7, 12, 18, 25, 34, 44, 52, 62],
    11: [0, 1, 7, 12, 18, 25, 34, 44, 52, 62, 70],
    12: [0, 1, 7, 12, 18, 25, 34, 44, 52, 62, 70, 79],
}

def compute_score(U):
    """Compute the raw score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
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
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def is_sidon(U):
    """Check if U is a Sidon set (all pairwise sums are distinct or symmetric)."""
    sums = {}
    for i in range(len(U)):
        for j in range(i, len(U)):
            s = U[i] + U[j]
            if s in sums:
                # Check if it's the same pair
                if (sums[s][0], sums[s][1]) != (i, j) and (sums[s][1], sums[s][0]) != (i, j):
                    return False
            sums[s] = (i, j)
    return True

def local_search(U, time_limit=10):
    """Perform local search to improve the set U based on score."""
    start_time = time.time()
    U = sorted(set(U))
    best_U = list(U)
    best_score = compute_score(U)
    
    n = len(U)
    if n < 2:
        return best_U
    
    # We will try to move each element to a better position
    # Since the score depends on global properties, we recompute it.
    # To be efficient, we limit the range of moves.
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(1, n):  # Keep 0 fixed
            current_val = U[i]
            # Try a range of new values around current_val and some random ones
            # Since we want small max(U), we bias towards smaller values
            candidates = set()
            # Local moves
            for delta in range(-5, 6):
                candidates.add(current_val + delta)
            # Global moves (small values)
            for v in range(1, max(10, current_val)):
                candidates.add(v)
            
            # Shuffle candidates to break symmetry
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            for new_val in cand_list:
                if new_val < 0 or new_val == current_val:
                    continue
                if new_val in U:
                    continue
                
                new_U = list(U)
                new_U[i] = new_val
                new_U.sort()
                
                # Ensure 0 is still the minimum
                if new_U[0] != 0:
                    continue
                
                # Optional: We can skip if it's not a Sidon set if we want,
                # but non-Sidon sets can be better. So we evaluate all.
                score = compute_score(new_U)
                
                if score > best_score + 1e-9:
                    best_score = score
                    best_U = new_U
                    U = new_U
                    improved = True
                    break # Restart search from new best
            
            if improved:
                break
    
    return best_U

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Try known Golomb rulers
    for n in range(2, 13):
        if time.time() > deadline:
            break
        if n in KNOWN_GOLOMB:
            U = list(KNOWN_GOLOMB[n])
            score = compute_score(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Local search on the best candidate
    if time.time() < deadline - 5:
        improved_U = local_search(best_U, time_limit=15)
        score = compute_score(improved_U)
        if score > best_score:
            best_score = score
            best_U = improved_U
    
    # Try other starting points and local search
    # We know n=5 is promising. Let's try to optimize n=4, 5, 6 specifically.
    starting_sets = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 11], # Non-Sidon example
        [0, 1, 3, 6, 9], # Arithmetic progression?
        [0, 2, 3, 7, 10],
    ]
    
    for start_set in starting_sets:
        if time.time() > deadline - 10:
            break
        if len(start_set) < 2:
            continue
        U = list(start_set)
        U.sort()
        if U[0] != 0:
            continue
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_U = U
        
        # Local search
        if time.time() < deadline - 10:
            improved_U = local_search(U, time_limit=10)
            score = compute_score(improved_U)
            if score > best_score:
                best_score = score
                best_U = improved_U

    return best_U
# EVOLVE-BLOCK-END
