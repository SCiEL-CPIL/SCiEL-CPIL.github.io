import time
import random
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
NUM_TARGETS = 129

_START_TIME = time.time()
DEADLINE = 85.0

def _elapsed():
    return time.time() - _START_TIME

def _pair_counts_fast(A):
    """Fast pair counts using numpy."""
    A = np.asarray(A, dtype=np.int32)
    n = len(A)
    if n == 0:
        return np.zeros(NUM_TARGETS, dtype=np.int32)
    outer = A[:, None] + A[None, :]
    mask = (outer >= TARGET_LO) & (outer <= TARGET_HI)
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=0)
    mask = mask & triu_mask
    if np.any(mask):
        idx = outer[mask] - TARGET_LO
        counts = np.bincount(idx, minlength=NUM_TARGETS).astype(np.int32)[:NUM_TARGETS]
    else:
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
    return counts

def _is_valid(counts):
    return int(np.min(counts)) > 0

def _get_peak(counts):
    return int(np.max(counts))

def _delta_swap(A_arr, i, x_new):
    """Compute delta in counts when replacing A[i] with x_new."""
    n = len(A_arr)
    ai = A_arr[i]
    delta = np.zeros(NUM_TARGETS, dtype=np.int32)
    
    # Remove contributions of ai with all other elements
    for j in range(n):
        s = ai + A_arr[j]
        if TARGET_LO <= s <= TARGET_HI:
            delta[s - TARGET_LO] -= 1
    
    # Add contributions of x_new with all other elements
    for j in range(n):
        s = x_new + A_arr[j]
        if TARGET_LO <= s <= TARGET_HI:
            delta[s - TARGET_LO] += 1
    
    return delta

def _try_swap(A, A_set, counts, i, x_new):
    """Try replacing A[i] with x_new. Returns (new_counts, new_peak) or None."""
    if x_new == A[i] or x_new in A_set:
        return None
    
    A_arr = np.asarray(A, dtype=np.int32)
    delta = _delta_swap(A_arr, i, x_new)
    new_counts = counts + delta
    
    if np.min(new_counts) <= 0:
        return None
    
    new_peak = int(np.max(new_counts))
    return (new_counts, new_peak)

def _local_search_peak(A, max_time=40.0, rng=None):
    """Swap-based local search to minimize peak."""
    if rng is None:
        rng = random.Random()
    
    A = sorted(list(A))
    A_set = set(A)
    counts = _pair_counts_fast(A)
    
    if not _is_valid(counts):
        return (A, float('inf'), counts)
    
    current_peak = _get_peak(counts)
    best_A = list(A)
    best_peak = current_peak
    best_counts = counts.copy()
    
    n = len(A)
    stall = 0
    max_stall = 30
    
    while _elapsed() < max_time and stall < max_stall:
        stall += 1
        improved = False
        
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if _elapsed() > max_time:
                break
            ai = A[i]
            
            # Try all possible replacements
            for x_new in range(0, N + 1):
                if x_new == ai or x_new in A_set:
                    continue
                
                result = _try_swap(A, A_set, counts, i, x_new)
                if result is None:
                    continue
                
                new_counts, new_peak = result
                
                if new_peak < current_peak:
                    A[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A.sort()
                    counts = new_counts
                    current_peak = new_peak
                    improved = True
                    stall = 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                        best_counts = counts.copy()
                    break
        
        if not improved:
            stall += 5
    
    return (best_A, best_peak, best_counts)

def _balance_polish(A, counts, peak, max_time=20.0, rng=None):
    """
    Reduce variance of counts while keeping peak <= current peak.
    This can enable subsequent peak reductions.
    """
    if rng is None:
        rng = random.Random()
    
    A = sorted(list(A))
    A_set = set(A)
    n = len(A)
    
    current_peak = peak
    best_A = list(A)
    best_peak = peak
    best_counts = counts.copy()
    best_var = float(np.var(counts.astype(float)))
    
    max_iter = 200
    for _ in range(max_iter):
        if _elapsed() > max_time:
            break
        
        found = False
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if _elapsed() > max_time:
                break
            ai = A[i]
            
            for x_new in range(0, N + 1):
                if x_new == ai or x_new in A_set:
                    continue
                
                result = _try_swap(A, A_set, counts, i, x_new)
                if result is None:
                    continue
                
                new_counts, new_peak = result
                
                # Accept if peak decreases, or peak same but variance decreases
                if new_peak < current_peak:
                    A[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A.sort()
                    counts = new_counts
                    current_peak = new_peak
                    found = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                        best_counts = counts.copy()
                        best_var = float(np.var(counts.astype(float)))
                    break
                elif new_peak == current_peak:
                    new_var = float(np.var(new_counts.astype(float)))
                    if new_var < best_var - 0.01:
                        A[i] = x_new
                        A_set.discard(ai)
                        A_set.add(x_new)
                        A.sort()
                        counts = new_counts
                        found = True
                        best_var = new_var
                        best_A = list(A)
                        best_counts = counts.copy()
                        break
        
        if not found:
            break
    
    return (best_A, best_counts, best_peak)

def _try_remove(A, A_set, counts, i):
    """Try removing A[i]. Returns new_counts or None."""
    ai = A[i]
    n = len(A)
    new_counts = counts.copy()
    
    for j in range(n):
        s = ai + A[j]
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] -= 1
    
    if np.min(new_counts) <= 0:
        return None
    
    return new_counts

def _reduce_size(A, max_time=15.0):
    """Remove redundant elements."""
    A = sorted(list(A))
    A_set = set(A)
    counts = _pair_counts_fast(A)
    
    if not _is_valid(counts):
        return A, counts
    
    changed = True
    while changed and _elapsed() < max_time and len(A) > 10:
        changed = False
        for i in range(len(A)):
            if _elapsed() > max_time:
                break
            result = _try_remove(A, A_set, counts, i)
            if result is not None:
                ai = A[i]
                A.pop(i)
                A_set.discard(ai)
                counts = result
                changed = True
                break
    
    return A, counts

def _greedy_build(rng, max_time=30.0):
    """Greedy build to cover all targets."""
    A = []
    A_set = set()
    
    # Start with structured anchors
    for x in [0, 64, 128]:
        if x not in A_set:
            A.append(x)
            A_set.add(x)
    
    A.sort()
    counts = _pair_counts_fast(A)
    
    while _elapsed() < max_time:
        if _is_valid(counts):
            break
        
        uncovered = np.where(counts == 0)[0]
        if len(uncovered) == 0:
            break
        
        # Pick the most uncovered target
        target_idx = int(rng.choice(uncovered))
        
        best_x = None
        best_gain = -1
        
        for x in range(0, N + 1):
            if x in A_set:
                continue
            gain = 0
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                    gain += 1
            s = x + x
            if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None or best_gain <= 0:
            break
        
        A.append(best_x)
        A_set.add(best_x)
        A.sort()
        counts = _pair_counts_fast(A)
    
    return A, counts

def _structured_init(rng):
    """Generate structured initializations."""
    inits = []
    
    # Arithmetic progressions
    for step in [2, 3, 4, 5, 6, 8]:
        A = list(range(0, N + 1, step))
        inits.append(sorted(A))
    
    # Two interleaved progressions
    for step in [4, 6, 8, 10]:
        A1 = list(range(0, N + 1, step))
        A2 = list(range(step // 2, N + 1, step))
        A = sorted(set(A1 + A2))
        inits.append(A)
    
    # Random sets of various sizes
    for size in [20, 24, 28, 32, 36, 40, 48, 56, 64]:
        A = sorted(rng.sample(range(0, N + 1), size))
        inits.append(A)
    
    # Hybrid: dense at ends, sparse in middle
    for mid_size in [8, 12, 16]:
        A = list(range(0, 16)) + list(range(N - 15, N + 1))
        mid = rng.sample(range(16, N - 15), min(mid_size, N - 32))
        A = sorted(set(A + mid))
        inits.append(A)
    
    return inits

def _fix_coverage(A, counts, rng, max_adds=20):
    """Add elements to fix uncovered targets."""
    A = sorted(list(A))
    A_set = set(A)
    
    for _ in range(max_adds):
        if _is_valid(counts):
            break
        
        uncovered = np.where(counts == 0)[0]
        if len(uncovered) == 0:
            break
        
        target_idx = int(rng.choice(uncovered))
        
        best_x = None
        best_gain = -1
        
        for x in range(0, N + 1):
            if x in A_set:
                continue
            gain = 0
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                    gain += 1
            s = x + x
            if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None or best_gain <= 0:
            candidates = [x for x in range(0, N + 1) if x not in A_set]
            if not candidates:
                break
            best_x = rng.choice(candidates)
        
        A.append(best_x)
        A_set.add(best_x)
        A.sort()
        counts = _pair_counts_fast(A)
    
    return A, counts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    best_A = None
    best_peak = float('inf')
    
    # Phase 1: Generate diverse initializations
    inits = _structured_init(rng)
    
    # Also add greedy build
    A_greedy, _ = _greedy_build(rng, max_time=DEADLINE - 40)
    if len(A_greedy) > 0:
        inits.append(A_greedy)
    
    # Phase 2: For each init, repair if needed, then optimize
    for A_init in inits:
        if _elapsed() > DEADLINE - 10:
            break
        
        A_init = sorted(set(A_init))
        counts = _pair_counts_fast(A_init)
        
        if not _is_valid(counts):
            A_init, counts = _fix_coverage(A_init, counts, rng)
            if not _is_valid(counts):
                continue
        
        # Phase 3: Local search to minimize peak
        A_opt, peak, counts_opt = _local_search_peak(A_init, max_time=min(25.0, DEADLINE - _elapsed() - 10), rng=rng)
        
        if peak < best_peak:
            best_peak = peak
            best_A = A_opt
        
        # Phase 4: Balance polish (reduce variance, may enable further peak reduction)
        if best_A is not None and _elapsed() < DEADLINE - 8:
            counts_best = _pair_counts_fast(best_A)
            peak_best = _get_peak(counts_best)
            A_bal, counts_bal, peak_bal = _balance_polish(
                best_A, counts_best, peak_best,
                max_time=min(10.0, DEADLINE - _elapsed() - 5), rng=rng
            )
            if peak_bal < best_peak:
                best_peak = peak_bal
                best_A = A_bal
            
            # Phase 5: Another peak reduction after balancing
            if _elapsed() < DEADLINE - 5:
                A_opt2, peak2, _ = _local_search_peak(
                    best_A, max_time=min(8.0, DEADLINE - _elapsed() - 3), rng=rng
                )
                if peak2 < best_peak:
                    best_peak = peak2
                    best_A = A_opt2
        
        # Phase 6: Try to remove elements
        if best_A is not None and _elapsed() < DEADLINE - 3:
            A_red, counts_red = _reduce_size(best_A, max_time=min(5.0, DEADLINE - _elapsed() - 2))
            if _is_valid(counts_red):
                peak_red = _get_peak(counts_red)
                if peak_red <= best_peak:
                    best_peak = peak_red
                    best_A = A_red
        
        # Early termination if we reach a good peak
        if best_peak <= 5:
            break
    
    # Fallback
    if best_A is None or not _is_valid(_pair_counts_fast(best_A)):
        best_A = list(range(0, 129))
    
    return best_A
