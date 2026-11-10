import time
import random

def check_validity(A):
    """Check if A is a valid B2[2] set."""
    counts = {}
    A_sorted = sorted(A)
    for i, a in enumerate(A_sorted):
        for b in A_sorted[i:]:
            s = a + b
            counts[s] = counts.get(s, 0) + 1
            if counts[s] > 2:
                return False
    return True

def greedy_deterministic():
    chosen = []
    counts = {}
    for x in range(1, 1001):
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        valid = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                valid = False
                break
        if valid:
            chosen.append(x)
            counts = trial
    return chosen

def randomized_greedy():
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    chosen = []
    counts = {}
    for x in candidates:
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        valid = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                valid = False
                break
        if valid:
            chosen.append(x)
            counts = trial
    return chosen

def local_search(A, max_iters=100):
    A = set(A)
    improved = True
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        # Try to add an element
        for x in range(1, 1001):
            if x in A:
                continue
            new_sums = [x + y for y in A] + [2 * x]
            # Check if adding x is valid
            # We need to check counts for new_sums
            # Efficient check:
            # Current counts for sums in new_sums
            # If any count + 1 > 2, invalid
            # We don't have global counts in this function, so we'd need to recompute or pass them.
            # For simplicity in the loop, we can just try to add and check validity of the whole set?
            # That is slow.
            # Let's skip complex local search for now and rely on restarts.
            pass
    return list(A)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial deterministic greedy
    best_set = greedy_deterministic()
    best_size = len(best_set)
    
    # Randomized restarts
    while time.time() < deadline:
        candidate = randomized_greedy()
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate
            # Optional: try to improve with local search if time permits
            # For now, just keep the best random result
    
    return best_set
