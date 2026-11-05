import time
import random
import bisect

def _check_insertion(values, sums, x, pos):
    """Check if inserting x at position pos in values creates valid sums.
    Returns (valid, new_sums_set) if valid, else (False, None).
    values is sorted list, sums is set of all existing consecutive block sums.
    """
    n = len(values)
    if n == 0:
        if x in sums:
            return False, None
        return True, {x}
    
    # left_sums[i] = sum(values[i:pos]) for i in 0..pos
    left_sums = [0] * (pos + 1)
    for i in range(pos - 1, -1, -1):
        left_sums[i] = left_sums[i + 1] + values[i]
    
    # right_sums[j] = sum(values[pos:j+1]) for j in pos..n-1
    right_sums = {}
    s = 0
    for j in range(pos, n):
        s += values[j]
        right_sums[j] = s
    
    new_sums = set()
    for i in range(pos + 1):
        ls = left_sums[i]
        for j in range(pos, n):
            rs = right_sums[j]
            total = ls + x + rs
            if total in sums:
                return False, None
            new_sums.add(total)
    
    return True, new_sums

def _incremental_valid_append(values, sums, x):
    """Check if x can be appended to values. Returns (valid, new_sums_to_add)."""
    if values and x <= values[-1]:
        return False, None
        
    tails = []
    total = x
    tails.append(total)
    for a in reversed(values):
        total += a
        tails.append(total)
    for t in tails:
        if t in sums:
            return False, None
    return True, tails

def _compute_sums(seq):
    """Compute all consecutive block sums for a sequence."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _greedy_append(order):
    """Build a valid sequence by appending in given order."""
    values = []
    sums = set()
    for x in order:
        valid, new_sums = _incremental_valid_append(values, sums, x)
        if valid:
            values.append(x)
            sums.update(new_sums)
    return values, sums

def _local_search_insertion(values, sums, available, deadline, rng):
    """Try to improve by inserting available numbers at various positions."""
    current = values[:]
    current_sums = sums.copy()
    avail = set(available)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        avail_list = list(avail)
        rng.shuffle(avail_list)
        
        for x in avail_list:
            if time.time() >= deadline:
                break
            if x in current:
                continue
            
            pos_hint = bisect.bisect_left(current, x)
            
            positions_to_try = set()
            for d in range(0, min(len(current) + 1, 4)):
                p1 = max(0, pos_hint - d)
                p2 = min(len(current) + 1, pos_hint + d + 1)
                positions_to_try.add(p1)
                positions_to_try.add(p2)
            
            for _ in range(2):
                positions_to_try.add(rng.randint(0, len(current)))
            
            for pos in sorted(positions_to_try):
                valid, new_sums = _check_insertion(current, current_sums, x, pos)
                if valid:
                    current = current[:pos] + [x] + current[pos:]
                    current_sums.update(new_sums)
                    avail.discard(x)
                    improved = True
                    break
    
    return current, current_sums

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    best_sums = set()
    
    # 1. Deterministic Baselines
    seq, sums = _greedy_append(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq
        best_sums = sums
    
    seq, sums = _greedy_append(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq
        best_sums = sums
    
    # 2. Local search on best baseline
    available = set(range(1, 257)) - set(best_seq)
    rng = random.Random(42)
    improved_seq, improved_sums = _local_search_insertion(best_seq, best_sums, available, deadline, rng)
    if len(improved_seq) > len(best_seq):
        best_seq = improved_seq
        best_sums = improved_sums
    
    # 3. Random restarts with local search
    base_range = list(range(1, 257))
    
    while time.time() < deadline:
        rng.shuffle(base_range)
        seq, sums = _greedy_append(base_range)
        
        if len(seq) > len(best_seq):
            best_seq = seq
            best_sums = sums
        
        available = set(range(1, 257)) - set(seq)
        improved_seq, improved_sums = _local_search_insertion(seq, sums, available, deadline, rng)
        
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq
            best_sums = improved_sums
    
    return best_seq
