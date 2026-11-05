import time
import itertools
import random

def _is_sidon(subset):
    """Check if a subset has all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    if n <= 2:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(arr):
    """Find the largest Sidon subset size by exhaustive search."""
    n = len(arr)
    for size in range(n, 0, -1):
        for subset in itertools.combinations(arr, size):
            if _is_sidon(list(subset)):
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    best_score = 12
    best_set = list(range(12))
    
    def try_candidate(candidate):
        nonlocal best_score, best_set
        if len(set(candidate)) != 12:
            return
        if min(candidate) < 0 or max(candidate) > 128:
            return
        score = _max_sidon_subset_size(candidate)
        if score < best_score:
            best_score = score
            best_set = sorted(candidate)
    
    # Consecutive integers
    for start in range(0, 117):
        if time.time() > deadline:
            break
        try_candidate(list(range(start, start + 12)))
    
    # Two blocks of 6
    if time.time() < deadline:
        for gap in range(1, 30):
            for offset in range(0, 10):
                if time.time() > deadline:
                    break
                candidate = list(range(offset, offset + 6)) + list(range(offset + gap, offset + gap + 6))
                try_candidate(candidate)
    
    # Three blocks of 4
    if time.time() < deadline:
        for g1 in range(1, 20):
            for g2 in range(1, 20):
                for offset in range(0, 5):
                    if time.time() > deadline:
                        break
                    candidate = (list(range(offset, offset + 4)) + 
                                list(range(offset + g1, offset + g1 + 4)) + 
                                list(range(offset + g1 + g2, offset + g1 + g2 + 4)))
                    try_candidate(candidate)
    
    # Four blocks of 3
    if time.time() < deadline:
        for g1 in range(1, 15):
            for g2 in range(1, 15):
                for g3 in range(1, 15):
                    for offset in range(0, 5):
                        if time.time() > deadline:
                            break
                        candidate = (list(range(offset, offset + 3)) + 
                                    list(range(offset + g1, offset + g1 + 3)) + 
                                    list(range(offset + g1 + g2, offset + g1 + g2 + 3)) + 
                                    list(range(offset + g1 + g2 + g3, offset + g1 + g2 + g3 + 3)))
                        try_candidate(candidate)
    
    # Random search with local optimization
    if time.time() < deadline:
        random.seed(42)
        for _ in range(500):
            if time.time() > deadline:
                break
            candidate = sorted(random.sample(range(129), 12))
            score = _max_sidon_subset_size(candidate)
            if score < best_score:
                best_score = score
                best_set = candidate[:]
            # Local search: try replacing one element
            for i in range(12):
                if time.time() > deadline:
                    break
                for new_val in range(129):
                    if new_val in candidate:
                        continue
                    trial = candidate[:]
                    trial[i] = new_val
                    trial.sort()
                    t_score = _max_sidon_subset_size(trial)
                    if t_score < score:
                        score = t_score
                        candidate = trial
                        if score < best_score:
                            best_score = score
                            best_set = candidate[:]
    
    return best_set
