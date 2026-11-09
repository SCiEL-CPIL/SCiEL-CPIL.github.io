"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _is_sidon(subset):
    """Return True if subset has all pair sums (incl. diagonals) unique."""
    subset = sorted(subset)
    k = len(subset)
    sums = set()
    for i in range(k):
        for j in range(i, k):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient):
    """Find the largest subset of ambient with all pair sums (incl. diagonals) unique."""
    ambient = sorted(ambient)
    n = len(ambient)
    # Check from 6 downwards
    for size in range(6, 3, -1):
        for combo in itertools.combinations(ambient, size):
            if _is_sidon(combo):
                return size
    return 3

def _evaluate(ambient):
    """Evaluate a candidate ambient set. Returns the max Sidon-with-diagonals subset size."""
    if len(ambient) != 12:
        return 999
    if len(set(ambient)) != 12:
        return 999
    if any(x < 0 or x > 128 for x in ambient):
        return 999
    return _max_sidon_size(ambient)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    # Candidate 1: Almost AP with a break
    # The AP 0..11 has max Sidon size 5.
    # Try perturbing the last element to break the specific Sidon sets.
    base_ap = list(range(12))
    
    best_set = base_ap
    best_score = _evaluate(base_ap)
    
    # Try shifting the last element
    for delta in range(1, 10):
        candidate = base_ap.copy()
        candidate[-1] = 11 + delta
        if candidate[-1] <= 128:
            score = _evaluate(candidate)
            if score < best_score:
                best_score = score
                best_set = candidate
                if best_score == 4:
                    return best_set
    
    # Try shifting other elements
    # Local search: perturb one element at a time
    current_set = best_set[:]
    current_score = best_score
    
    rng = random.Random(42)
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        # Pick a random index to perturb
        idx = rng.randint(0, 11)
        orig_val = current_set[idx]
        
        # Try small perturbations
        for delta in [-2, -1, 1, 2, -3, 3]:
            new_val = orig_val + delta
            if 0 <= new_val <= 128:
                trial_set = current_set[:]
                trial_set[idx] = new_val
                if len(set(trial_set)) == 12:
                    score = _evaluate(trial_set)
                    if score < current_score:
                        current_score = score
                        current_set = trial_set
                        if current_score == 4:
                            return current_set
                    elif score == current_score:
                        # Accept with some probability to escape local minima?
                        # Or just keep best.
                        pass
    
    return current_set
# EVOLVE-BLOCK-END
