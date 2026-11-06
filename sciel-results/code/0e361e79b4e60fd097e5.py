"""Combined approach: fast bitset validation + structured local search for distinct subset sums."""
# EVOLVE-BLOCK-START
import time
import random

def _subset_sums_distinct(seq):
    """
    Check if all 2^n subset sums are distinct using bitset.
    Returns True if valid.
    """
    n = len(seq)
    bits = 1  # sum 0 achievable
    for a in seq:
        bits |= (bits << a)
    # Count set bits; must equal 2^n
    return bin(bits).count('1') == (1 << n)

def _greedy_construct(n=12, max_val=1048576):
    """
    Greedy construction: pick smallest possible next element that keeps all subset sums distinct.
    Returns a valid sequence or None.
    """
    seq = []
    for i in range(n):
        start = 1 if not seq else seq[-1] + 1
        found = False
        for a in range(start, max_val + 1):
            candidate = seq + [a]
            if _subset_sums_distinct(candidate):
                seq.append(a)
                found = True
                break
        if not found:
            return None
    return seq

def _local_search(seq, n=12, max_val=1048576, time_limit=55.0, seed=42):
    """
    Local search: iteratively try to reduce elements while maintaining validity.
    Objective: minimize max element, then sum.
    """
    rng = random.Random(seed)
    current = sorted(seq)
    cur_max = max(current)
    best_seq = list(current)
    best_max = cur_max
    start_time = time.time()
    
    def is_valid(s):
        return _subset_sums_distinct(s)
    
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Try reducing each element
        for idx in range(n):
            if time.time() - start_time > time_limit:
                break
            old_val = current[idx]
            # Try smaller values in random order to escape local minima
            candidates = list(range(1, old_val))
            rng.shuffle(candidates)
            for new_val in candidates:
                if time.time() - start_time > time_limit:
                    break
                candidate = list(current)
                candidate[idx] = new_val
                # Ensure distinct values
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
                        # Update best
                        if cur_max < best_max or (cur_max == best_max and sum(current) < sum(best_seq)):
                            best_seq = list(current)
                            best_max = cur_max
                        break
            if improved:
                # After an improvement, re-sort to maintain structure
                current.sort()
    
    return best_seq

def _randomized_local_search(seq, n=12, max_val=1048576, time_limit=55.0, seed=42):
    """
    Enhanced local search with random restarts and perturbation.
    """
    rng = random.Random(seed)
    start_time = time.time()
    best_seq = sorted(seq)
    best_max = max(best_seq)
    
    # First, do a standard local search
    result = _local_search(best_seq, n, max_val, min(10.0, time_limit - 5.0), seed=seed)
    if result:
        best_seq = result
        best_max = max(best_seq)
    
    # Now try random restarts with perturbation
    attempts = 0
    while time.time() - start_time < time_limit - 2:
        attempts += 1
        # Perturb the best sequence
        base = list(best_seq)
        num_perturb = rng.randint(1, 3)
        for _ in range(num_perturb):
            idx = rng.randint(0, n - 1)
            offset = rng.randint(-100, 100)
            new_val = base[idx] + offset
            if 1 <= new_val <= max_val:
                base[idx] = new_val
        base.sort()
        
        # Check if perturbed sequence is valid
        if len(set(base)) == n and _subset_sums_distinct(base):
            # Do local search from this point
            remaining = time_limit - (time.time() - start_time) - 2
            if remaining > 1.0:
                result = _local_search(base, n, max_val, remaining, seed=seed + attempts)
                if result and max(result) < best_max:
                    best_seq = result
                    best_max = max(best_seq)
                    if best_max <= 2000:
                        break
        else:
            # Try to fix the perturbed sequence by local search anyway
            # First, find a valid neighborhood
            for idx in range(n):
                for delta in range(-50, 51):
                    candidate = list(base)
                    candidate[idx] += delta
                    if 1 <= candidate[idx] <= max_val and len(set(candidate)) == n:
                        if _subset_sums_distinct(candidate):
                            base = sorted(candidate)
                            break
                if _subset_sums_distinct(base):
                    break
            if _subset_sums_distinct(base):
                remaining = time_limit - (time.time() - start_time) - 2
                if remaining > 1.0:
                    result = _local_search(base, n, max_val, remaining, seed=seed + attempts)
                    if result and max(result) < best_max:
                        best_seq = result
                        best_max = max(best_seq)
                        if best_max <= 2000:
                            break
    
    return best_seq

def construct():
    """
    Construct 12 distinct positive integers <= 1048576 with all subset sums distinct.
    Minimize the largest integer.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Step 1: Get a valid starting point via greedy
    seq = _greedy_construct(12, 1048576)
    if seq is None:
        # Fallback: powers of 2
        return [1 << i for i in range(12)]
    
    # Step 2: Local search to minimize max element
    best_seq = _randomized_local_search(seq, 12, 1048576, time_limit=min(55.0, deadline - time.time() - 5.0), seed=42)
    
    # Step 3: Final polish with tighter local search
    remaining = deadline - time.time() - 2.0
    if remaining > 5.0:
        final_seq = _local_search(best_seq, 12, 1048576, time_limit=remaining, seed=999)
        if final_seq and max(final_seq) < max(best_seq):
            best_seq = final_seq
    
    return best_seq
# EVOLVE-BLOCK-END
