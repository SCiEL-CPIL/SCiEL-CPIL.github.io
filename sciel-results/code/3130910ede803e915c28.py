"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import math
import random
from collections import Counter

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the lower bound score for a finite set U containing 0."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return 0.0
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sum_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
    
    # Compute difference set U-U
    diff_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
    
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    ratio = len(diff_set) / len(sum_set)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    score = 1 + math.log(ratio) / log_denom
    return score


def _try_add_element(U, candidate):
    """Try adding candidate to U and return (new_U, new_score) if improved."""
    new_U = sorted(set(U) | {candidate})
    return new_U, _score_set(new_U)


def _exhaustive_small_search(max_size=6, max_val=50, time_limit=30):
    """Exhaustively search for optimal small sets."""
    start_time = time.time()
    best_U = [0]
    best_score = _score_set([0])
    
    # Try all subsets of {1, ..., max_val} of size up to max_size
    # For efficiency, use iterative deepening
    for size in range(1, min(max_size, max_val) + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Use combinations for small sizes
        if size <= 4:
            for combo in itertools.combinations(range(1, max_val + 1), size):
                U = [0] + list(combo)
                score = _score_set(U)
                if score > best_score:
                    best_score = score
                    best_U = U
        else:
            # For larger sizes, use random sampling
            for _ in range(10000):
                if time.time() - start_time > time_limit:
                    break
                combo = random.sample(range(1, max_val + 1), size)
                U = [0] + sorted(combo)
                score = _score_set(U)
                if score > best_score:
                    best_score = score
                    best_U = U
    
    return best_U, best_score


def _greedy_construct(max_size=50, max_val=200, time_limit=100):
    """Greedy construction starting from {0}."""
    start_time = time.time()
    U = [0]
    current_score = _score_set(U)
    
    # Precompute candidate values
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    
    while len(U) < max_size and (time.time() - start_time) < time_limit:
        best_candidate = None
        best_score = current_score
        best_new_U = None
        
        for cand in candidates:
            if cand in U:
                continue
            new_U, new_score = _try_add_element(U, cand)
            if new_score > best_score:
                best_score = new_score
                best_candidate = cand
                best_new_U = new_U
        
        if best_candidate is None:
            break
        
        U = best_new_U
        current_score = best_score
        candidates.remove(best_candidate)
    
    return U, current_score


def _local_search(U, time_limit=80):
    """Local search: try adding and removing elements."""
    start_time = time.time()
    U = sorted(set(U))
    current_score = _score_set(U)
    max_val = max(U) * 2 + 100
    if max_val > 5000:
        max_val = 5000
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try adding
        for cand in range(1, max_val + 1):
            if cand in U:
                continue
            new_U, new_score = _try_add_element(U, cand)
            if new_score > current_score + 1e-12:
                U = new_U
                current_score = new_score
                improved = True
                max_val = max(max_val, cand + 50)
        
        # Try removing
        for i, elem in enumerate(U):
            if elem == 0:
                continue
            new_U = U[:i] + U[i+1:]
            new_score = _score_set(new_U)
            if new_score > current_score + 1e-12:
                U = new_U
                current_score = new_score
                improved = True
    
    return U, current_score


def _swap_local_search(U, time_limit=60):
    """Local search with swap operations: remove one, add another."""
    start_time = time.time()
    U = sorted(set(U))
    current_score = _score_set(U)
    max_val = max(U) * 3 + 50
    if max_val > 3000:
        max_val = 3000
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try swap: remove elem, add cand
        for i, elem in enumerate(U):
            if elem == 0:
                continue
            for cand in range(1, max_val + 1):
                if cand in U:
                    continue
                new_U = U[:i] + U[i+1:] + [cand]
                new_U = sorted(set(new_U))
                new_score = _score_set(new_U)
                if new_score > current_score + 1e-12:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    
    return U, current_score


def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize score."""
    global_start = time.time()
    time_limit = 140  # Safety margin before 150s limit
    
    best_U = [0]
    best_score = _score_set(best_U)
    
    # Phase 1: Exhaustive search for small optimal sets
    if time.time() - global_start < 30:
        U, score = _exhaustive_small_search(max_size=6, max_val=50, time_limit=25)
        if score > best_score:
            best_U = U
            best_score = score
    
    # Phase 2: Try multiple random seeds for greedy construction
    for seed in range(8):
        if time.time() - global_start > time_limit - 30:
            break
        
        random.seed(seed * 12345 + 67890)
        U, score = _greedy_construct(max_size=35, max_val=120, time_limit=min(15, time_limit - 30 - (time.time() - global_start)))
        
        if score > best_score:
            best_U = U
            best_score = score
    
    # Phase 3: Local search on the best result
    if time.time() - global_start < time_limit - 20:
        U, score = _local_search(best_U, time_limit=min(60, time_limit - 20 - (time.time() - global_start)))
        if score > best_score:
            best_U = U
            best_score = score
    
    # Phase 4: Swap-based local search
    if time.time() - global_start < time_limit - 10:
        U, score = _swap_local_search(best_U, time_limit=min(40, time_limit - 10 - (time.time() - global_start)))
        if score > best_score:
            best_U = U
            best_score = score
    
    # Phase 5: Final local search polish
    if time.time() - global_start < time_limit - 5:
        U, score = _local_search(best_U, time_limit=min(20, time_limit - 5 - (time.time() - global_start)))
        if score > best_score:
            best_U = U
            best_score = score
    
    return sorted(best_U)
# EVOLVE-BLOCK-END
