import time
import random
import numpy as np

N = 16
DEADLINE = 85.0  # Safety margin before 90s kill

def _apply_comparators(comps, x):
    """Apply compare-exchange comparators to input vector x."""
    x = x.copy()
    for (i, j) in comps:
        a, b = x[i], x[j]
        if a > b:
            x[i], x[j] = b, a
    return x

def _verify_network(comps):
    """Verify sorting network using zero-one principle (all 2^N binary inputs)."""
    # Generate all 2^16 binary inputs as a (65536, 16) array
    # Using a trick to generate efficiently
    all_inputs = np.array([list(map(int, format(i, '016b'))) for i in range(65536)], dtype=np.int8)
    
    # Apply all comparators
    x = all_inputs.copy()
    for (i, j) in comps:
        a = x[:, i].copy()
        b = x[:, j].copy()
        # min goes to i, max goes to j
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
    
    # Check if each row is sorted (non-decreasing)
    diffs = np.diff(x, axis=1)
    return bool(np.all(diffs >= 0))

def _greedy_reduce(comps, deadline):
    """Try to remove redundant comparators greedily."""
    current = list(comps)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            # Try removing comparator at index i
            trial = current[:i] + current[i+1:]
            if _verify_network(trial):
                current = trial
                changed = True
                break # Restart loop as indices changed
    return current

def _local_search(comps, deadline):
    """Local search to minimize comparator count."""
    best = list(comps)
    best_count = len(best)
    random.seed(42)
    
    start_time = time.time()
    no_improve = 0
    
    while time.time() < deadline:
        # Try random removal
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                best_count = len(best)
                no_improve = 0
                continue
        
        # Try moving a comparator to a different position
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            comp = best[idx]
            trial = best[:idx] + best[idx+1:]
            new_idx = random.randint(0, len(trial))
            trial.insert(new_idx, comp)
            if _verify_network(trial):
                best = trial
                no_improve = 0
                continue
        
        no_improve += 1
        if no_improve > 5000:
            break
            
    return best

def construct():
    # Start with Odd-Even Transposition (Valid, 120 comparators)
    comps = [(i, i + 1) for phase in range(N)
             for i in range(phase % 2, N - 1, 2)]
    
    if not _verify_network(comps):
        # Should not happen
        return comps
    
    # 1. Greedy reduction
    deadline_greedy = time.time() + 20.0
    comps = _greedy_reduce(comps, deadline_greedy)
    
    # 2. Local search with moves
    deadline_local = time.time() + (DEADLINE - time.time() - 5.0)
    comps = _local_search(comps, deadline_local)
    
    # 3. Final greedy sweep
    deadline_final = time.time() + (DEADLINE - time.time() - 2.0)
    comps = _greedy_reduce(comps, deadline_final)
    
    return comps
