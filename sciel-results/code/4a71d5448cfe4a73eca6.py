import time
import numpy as np

DEADLINE_SECONDS = 140.0  # safety margin before 150s kill

def _greedy_construct():
    """Deterministic ascending greedy baseline (matches seed behavior)."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        ok = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                ok = False
                break
        if ok:
            chosen.append(x)
            counts = trial
    return chosen

def _count_sums_numpy(mask):
    """Count sum representations a+b (a<=b) for all sums using convolution.
    mask is a boolean array of length 1001 (index 0 unused).
    Returns an array where index s = number of pairs (a,b) with a<=b, a+b=s,
    both in the set.
    """
    # Use convolution: c = mask * mask gives counts of ordered pairs (a,b) with a+b=s
    # For a<=b: count = (c[s] + (s even and s//2 in set ? 1 : 0)) / 2
    # Actually: unordered pairs with a<=b:
    #   if s is even and s/2 is in set: the pair (s/2, s/2) is one unordered pair
    #   other pairs come in (a,b) and (b,a) with a != b, so each contributes 2 to c[s]
    # So: count_unordered = (c[s] - diag[s]) / 2 + diag[s] where diag[s]=1 if s even and s/2 in set
    # Simplified: count = (c[s] + diag[s]) / 2
    m = mask.astype(np.float64)
    c = np.convolve(m, m)  # c[s] = number of ordered pairs (a,b) with a+b=s
    # diag: for each s, 1 if s is even and s//2 is in the set (i.e., the pair (s/2,s/2))
    # s ranges from 2 to 2000
    diag = np.zeros(len(c))
    # s even, s//2 in [1,1000], mask[s//2] is True
    even_s = np.arange(2, 2001, 2)
    half = even_s // 2
    diag[even_s] = mask[half].astype(np.float64)
    count = (c + diag) / 2.0
    return count

def _violations(mask):
    """Return number of sums with > 2 representations."""
    c = _count_sums_numpy(mask)
    return int(np.sum(c > 2))

def _local_search(mask, time_limit):
    """Randomized local search to improve the set. Returns best mask found."""
    best_mask = mask.copy()
    best_size = int(np.sum(mask))
    best_viol = _violations(mask)
    
    if best_viol == 0:
        return best_mask, best_size
    
    rng = np.random.default_rng(42)
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50000
    
    while time.time() - start_time < time_limit:
        # Randomly pick an element to toggle
        x = rng.integers(1, 1001)
        if mask[x]:
            # Try removing x
            mask[x] = False
            v = _violations(mask)
            size = int(np.sum(mask))
            # Accept if violations decreased (size can go down)
            if v < best_viol:
                best_mask = mask.copy()
                best_size = size
                best_viol = v
                no_improve_count = 0
            else:
                mask[x] = True  # revert
                no_improve_count += 1
        else:
            # Try adding x
            mask[x] = True
            v = _violations(mask)
            size = int(np.sum(mask))
            # Accept if size increased and violations didn't increase
            if size > best_size and v <= best_viol:
                best_mask = mask.copy()
                best_size = size
                best_viol = v
                no_improve_count = 0
            else:
                mask[x] = False  # revert
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
    
    return best_mask, best_size

def construct():
    start = time.time()
    
    # Step 1: Greedy baseline
    chosen = _greedy_construct()
    best_size = len(chosen)
    best_set = set(chosen)
    
    # Build mask for local search
    mask = np.zeros(1001, dtype=bool)
    for x in chosen:
        mask[x] = True
    
    # Step 2: Local search with time budget
    remaining = DEADLINE_SECONDS - (time.time() - start)
    if remaining > 5:
        best_mask, best_size = _local_search(mask, remaining)
        best_set = set(int(x) for x in np.where(best_mask)[0])
    
    # Step 3: Try a few more random restarts if time permits
    rng = np.random.default_rng(123)
    while time.time() - start < DEADLINE_SECONDS - 10:
        # Start from current best, perturb
        trial_mask = best_mask.copy() if 'best_mask' in dir() else mask.copy()
        # Randomly flip a few elements
        n_flips = rng.integers(1, 5)
        for _ in range(n_flips):
            x = rng.integers(1, 1001)
            trial_mask[x] = not trial_mask[x]
        
        # Local search from perturbed state
        remaining = DEADLINE_SECONDS - (time.time() - start)
        if remaining < 2:
            break
        new_mask, new_size = _local_search(trial_mask, remaining)
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask
            best_set = set(int(x) for x in np.where(new_mask)[0])
    
    return sorted(best_set)
