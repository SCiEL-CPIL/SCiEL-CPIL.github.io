"""Model-and-optimize approach for distinct subset sums with small max element."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import math

def _all_subset_sums_distinct(seq):
    """Check if all subset sums are distinct. Returns True if valid."""
    n = len(seq)
    # Use bitset via Python int for subset sums
    bits = 1  # only sum 0
    for a in seq:
        if a < 0:
            return False
        bits |= (bits << a)
    # Count set bits
    cnt = bin(bits).count('1')
    return cnt == (1 << n)

def _greedy_construction(n=12, max_val=1048576):
    """
    Greedy: start with smallest possible, then for each next element,
    pick the smallest integer that keeps all subset sums distinct.
    """
    seq = []
    for i in range(n):
        start = 1
        found = False
        for a in range(start, max_val + 1):
            candidate = seq + [a]
            if _all_subset_sums_distinct(candidate):
                seq.append(a)
                found = True
                break
        if not found:
            return None
    return seq

def _local_search(seq, n=12, max_val=1048576, time_limit=60.0, seed=42):
    """
    Local search to minimize the largest element.
    Start from a valid sequence, try to reduce elements via swaps/moves.
    """
    rng = random.Random(seed)
    best_seq = list(seq)
    best_max = max(best_seq)
    start_time = time.time()
    
    def is_valid(s):
        return _all_subset_sums_distinct(s)
    
    current = list(best_seq)
    cur_max = max(current)
    
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2:
            break
        
        for idx in range(n):
            if elapsed > time_limit - 2:
                break
            old_val = current[idx]
            for new_val in range(1, old_val):
                if elapsed > time_limit - 2:
                    break
                candidate = list(current)
                candidate[idx] = new_val
                if len(set(candidate)) != n:
                    continue
                if max(candidate) > max_val:
                    continue
                if is_valid(candidate):
                    new_max = max(candidate)
                    if new_max < cur_max or (new_max == cur_max and sum(candidate) < sum(current)):
                        current = candidate
                        cur_max = new_max
                        improved = True
                        break
            if improved:
                break
    
    if cur_max < best_max:
        best_seq = list(current)
        best_max = cur_max
    
    return best_seq

def local_search_refined(seq, n=12, max_val=1048576, time_limit=10.0, rng=None):
    """Refined local search with multiple strategies."""
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    current = list(seq)
    cur_max = max(current)
    best_seq = list(current)
    best_max = cur_max
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit - 1:
            break
        improved = False
        for idx in range(n):
            if time.time() - start_time > time_limit - 1:
                break
            old = current[idx]
            # Try smaller values first (to reduce max)
            for new in range(1, old):
                if time.time() - start_time > time_limit - 1:
                    break
                cand = list(current)
                cand[idx] = new
                if len(set(cand)) != n:
                    continue
                if max(cand) > max_val:
                    continue
                if _all_subset_sums_distinct(cand):
                    nm = max(cand)
                    if nm < cur_max or (nm == cur_max and sum(cand) < sum(current)):
                        current = cand
                        cur_max = nm
                        if cur_max < best_max:
                            best_seq = list(current)
                            best_max = cur_max
                        improved = True
                        break
        # Try random perturbations to escape local optima
        if not improved and time.time() - start_time < time_limit - 1:
            for _ in range(100):
                if time.time() - start_time > time_limit - 1:
                    break
                idx = rng.randint(0, n - 1)
                delta = rng.randint(-50, 50)
                new_val = current[idx] + delta
                if new_val < 1 or new_val > max_val:
                    continue
                cand = list(current)
                cand[idx] = new_val
                if len(set(cand)) != n:
                    continue
                if _all_subset_sums_distinct(cand):
                    nm = max(cand)
                    if nm < cur_max or (nm == cur_max and sum(cand) < sum(current)):
                        current = cand
                        cur_max = nm
                        if cur_max < best_max:
                            best_seq = list(current)
                            best_max = cur_max
                        improved = True
                        break
    
    return best_seq

def _simulated_annealing(seq, n=12, max_val=1048576, time_limit=20.0, rng=None):
    """
    Simulated annealing to minimize max element.
    Uses a smooth surrogate: minimize max element primarily, then sum.
    """
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    current = list(seq)
    cur_max = max(current)
    best_seq = list(current)
    best_max = cur_max
    
    # Temperature schedule
    T0 = 100.0
    T_final = 1.0
    total_steps = 5000
    
    for step in range(total_steps):
        if time.time() - start_time > time_limit - 1:
            break
        
        # Temperature cooling
        progress = step / total_steps
        T = T0 * (T_final / T0) ** progress
        
        # Pick a random element to modify
        idx = rng.randint(0, n - 1)
        # Random new value in a neighborhood
        radius = max(1, int(50 * (1 - progress)))
        delta = rng.randint(-radius, radius)
        new_val = current[idx] + delta
        
        if new_val < 1 or new_val > max_val:
            continue
        
        cand = list(current)
        cand[idx] = new_val
        if len(set(cand)) != n:
            continue
        
        if not _all_subset_sums_distinct(cand):
            # Invalid: reject with high probability
            continue
        
        cand_max = max(cand)
        cand_sum = sum(cand)
        
        # Objective: primarily max, secondarily sum
        cur_obj = cur_max * 10000 + sum(current)
        cand_obj = cand_max * 10000 + cand_sum
        delta_obj = cand_obj - cur_obj
        
        if delta_obj <= 0:
            # Improvement
            current = cand
            cur_max = cand_max
            if cur_max < best_max:
                best_seq = list(current)
                best_max = cur_max
        else:
            # Worse: accept with probability exp(-delta_obj / T)
            if rng.random() < math.exp(-delta_obj / max(T, 1e-10)):
                current = cand
                cur_max = cand_max
    
    return best_seq

def construct():
    """
    Construct 12 distinct positive integers <= 1048576 with all subset sums distinct.
    Minimize the largest element.
    """
    deadline = 85.0  # safety margin from 90s
    start_time = time.time()
    
    n = 12
    max_val = 1048576
    
    # Step 1: Greedy construction for a quick valid solution
    greedy_seq = _greedy_construction(n, max_val)
    if greedy_seq is None:
        # Fallback to powers of 2
        return [1 << i for i in range(n)]
    
    best_seq = greedy_seq
    best_max = max(best_seq)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_seq
    
    # Step 2: Local search to improve
    remaining = deadline - elapsed - 2
    if remaining > 1:
        improved = local_search_refined(best_seq, n, max_val, min(remaining, 15.0), random.Random(42))
        if improved and max(improved) < best_max:
            best_seq = improved
            best_max = max(improved)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_seq
    
    # Step 3: Simulated annealing for more aggressive search
    remaining = deadline - elapsed - 2
    if remaining > 5:
        rng = random.Random(999)
        sa_seq = _simulated_annealing(best_seq, n, max_val, min(remaining, 30.0), rng)
        if sa_seq and max(sa_seq) < best_max:
            best_seq = sa_seq
            best_max = max(sa_seq)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 3:
        return best_seq
    
    # Step 4: More local search rounds with random restarts
    remaining = deadline - elapsed - 2
    if remaining > 3:
        rng = random.Random(777)
        for attempt in range(5):
            if time.time() - start_time > deadline - 3:
                break
            # Perturb best sequence
            base = list(best_seq)
            for _ in range(rng.randint(2, 5)):
                idx = rng.randint(0, n - 1)
                offset = rng.randint(-100, 100)
                new_val = base[idx] + offset
                if new_val >= 1 and new_val <= max_val:
                    base[idx] = new_val
            base.sort()
            if len(set(base)) == n and _all_subset_sums_distinct(base):
                time_left = deadline - (time.time() - start_time) - 3
                if time_left > 1:
                    seq = local_search_refined(base, n, max_val, min(time_left, 5.0), rng)
                    if seq and max(seq) < best_max:
                        best_seq = seq
                        best_max = max(seq)
    
    return best_seq
# EVOLVE-BLOCK-END
