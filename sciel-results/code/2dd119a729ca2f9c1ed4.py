import time
import random

def _all_subset_sums_distinct(seq):
    """
    Check if all subset sums are distinct using a bitset.
    Returns True if valid.
    """
    n = len(seq)
    bits = 1
    for a in seq:
        bits |= (bits << a)
    return bin(bits).count('1') == (1 << n)

def _is_complete(seq):
    """
    Check if seq is a complete sequence: a[0]=1 and a[i] <= 1 + sum(a[0..i-1]).
    This guarantees all subset sums from 0 to total_sum are achievable.
    """
    if not seq or seq[0] != 1:
        return False
    s = 0
    for a in seq:
        if a > s + 1:
            return False
        s += a
    return True

def _construct_complete(n, target_max=None):
    """
    Construct a complete sequence of length n.
    If target_max is given, try to keep the largest element <= target_max.
    """
    if target_max is None:
        # Use powers of 2
        return [1 << i for i in range(n)]
    
    # Greedy: build sequence keeping max <= target_max
    # a[0] = 1
    seq = [1]
    s = 1
    for i in range(1, n):
        # We need a[i] <= s + 1 (completeness) and a[i] <= target_max
        # Also, to minimize max, we want to keep elements as small as possible
        # but also ensure we can reach the target_max constraint
        # Actually, for minimizing max, we want to spread the sum
        # Let's try: a[i] = min(s + 1, target_max)
        # But this might not work if target_max is too small
        # Better: try to keep elements balanced
        # The sum so far is s, we need n-i more elements
        # Total sum must be at least 2^n - 1
        # Let's just use a simple greedy: a[i] = min(s+1, target_max)
        a_i = min(s + 1, target_max)
        # But we also need a[i] > s (to be distinct from previous? No, just distinct values)
        # Actually, for completeness, a[i] <= s+1 is the constraint
        # And a[i] must be positive and distinct from other elements
        # Let's just try a[i] = s + 1 if that's <= target_max, else target_max
        if a_i <= s:
            a_i = s + 1
        if a_i > target_max:
            return None  # Can't construct with this target_max
        seq.append(a_i)
        s += a_i
    
    # Check if all elements are distinct
    if len(set(seq)) != n:
        return None
    
    # Check validity
    if not _all_subset_sums_distinct(seq):
        return None
    
    return seq

def _local_search(seq, deadline):
    """
    Local search to reduce the maximum element.
    """
    n = len(seq)
    best_seq = list(seq)
    best_max = max(seq)
    
    # Try to reduce the largest element
    # Sort indices by value descending
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        # Pick a random element to try to reduce
        idx = random.randint(0, n - 1)
        old_val = seq[idx]
        
        # Try reducing by a small amount
        for delta in [1, 2, 3, 5, 10]:
            new_val = old_val - delta
            if new_val <= 0:
                continue
            if new_val in seq:
                continue
            
            seq[idx] = new_val
            if _all_subset_sums_distinct(seq):
                new_max = max(seq)
                if new_max < best_max:
                    best_max = new_max
                    best_seq = list(seq)
                    break
            else:
                seq[idx] = old_val
                break
    
    return best_seq, best_max

def construct():
    n = 12
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    # Start with powers of 2
    seq = [1 << i for i in range(n)]
    best_seq = list(seq)
    best_max = max(seq)  # 2048
    
    # Try to construct a complete sequence with smaller max
    # The minimum total sum is 2^n - 1 = 4095
    # If we want max <= M, we need 12 elements summing to at least 4095
    # with each <= M, so M >= ceil(4095/12) = 342
    # But completeness constraints make it harder
    
    # Try various target_max values
    for target_max in range(2047, 340, -1):
        if time.time() > deadline:
            break
        result = _construct_complete(n, target_max)
        if result is not None:
            max_val = max(result)
            if max_val < best_max:
                best_max = max_val
                best_seq = list(result)
                # Don't break; try smaller targets
            else:
                # If we found a valid sequence with max == target_max, 
                # we can stop since we're going downward
                break
    
    # If we found something better, try local search to improve further
    if best_max < 2048:
        improved_seq, improved_max = _local_search(best_seq, deadline)
        if improved_max < best_max:
            best_max = improved_max
            best_seq = improved_seq
    
    return best_seq
