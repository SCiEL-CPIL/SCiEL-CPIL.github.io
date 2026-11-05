"""Improved program: multi-strategy search for max-length increasing sequence with distinct consecutive-block sums."""
import time
import random
import sys

def _check_valid(seq):
    """Check if a strictly increasing sequence has all distinct consecutive-block sums."""
    n = len(seq)
    if n == 0:
        return True
    # Check strictly increasing
    for i in range(n - 1):
        if seq[i + 1] <= seq[i]:
            return False
    # Check all values in [1, 256]
    for x in seq:
        if x < 1 or x > 256:
            return False
    # Check all consecutive block sums are distinct
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build(order, deadline=None):
    """
    Greedy: process candidates in given order, append if it doesn't create duplicate block sums.
    Returns the sequence built.
    """
    values = []
    sums = set()
    start_time = time.time()
    
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        # Compute new consecutive sums ending at x
        # A new sum is: x alone, x + last, x + last + second-to-last, etc.
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        
        # Check if any tail sum is already in sums
        if any(t in sums for t in tails):
            continue
        
        values.append(x)
        sums.update(tails)
    
    return values

def _greedy_build_fast(order, deadline=None):
    """
    Faster greedy using a set for O(1) lookup of conflicts.
    """
    values = []
    sums = set()
    start_time = time.time()
    
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        # Compute new consecutive sums ending at x
        total = x
        conflict = False
        for a in reversed(values):
            if total in sums:
                conflict = True
                break
            total += a
        if total in sums:
            conflict = True
        
        if conflict:
            continue
        
        # Add all new sums
        total = x
        sums.add(total)
        for a in reversed(values):
            total += a
            sums.add(total)
        
        values.append(x)
    
    return values

def _local_search(seq, deadline):
    """
    Try to improve the sequence by swapping or removing elements.
    Returns improved sequence or original.
    """
    best = list(seq)
    best_len = len(best)
    start_time = time.time()
    
    while time.time() < deadline:
        improved = False
        n = len(best)
        
        # Try removing each element and see if we can add a better one
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Remove element i
            candidate = best[:i] + best[i+1:]
            
            # Try to add elements from 1..256 that aren't in candidate
            existing = set(candidate)
            for x in range(1, 257):
                if x in existing:
                    continue
                # Check if adding x at the right position maintains increasing order
                # Find where x would go
                pos = None
                for j in range(len(candidate) + 1):
                    if j == 0 and (j == len(candidate) or x < candidate[j]):
                        pos = j
                        break
                    elif j > 0 and j < len(candidate) and candidate[j-1] < x < candidate[j]:
                        pos = j
                        break
                    elif j == len(candidate) and candidate[j-1] < x:
                        pos = j
                        break
                
                if pos is None:
                    continue
                
                new_seq = candidate[:pos] + [x] + candidate[pos:]
                if _check_valid(new_seq):
                    if len(new_seq) > best_len:
                        best = new_seq
                        best_len = len(best)
                        improved = True
                        break
        
        if not improved:
            # Try pairwise swaps
            n = len(best)
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    # Swap best[i] and best[j] - but must maintain increasing order
                    # This only works if the sequence remains increasing after swap
                    # Actually swapping breaks increasing order unless we re-sort, 
                    # which changes the sequence. Let's try a different approach.
                    pass
        
        if not improved:
            break
    
    return best

def _random_greedy(deadline):
    """
    Run greedy with a random order of candidates.
    """
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    return _greedy_build_fast(candidates, deadline)

def _reverse_greedy(deadline):
    """
    Greedy from largest to smallest.
    """
    candidates = list(range(256, 0, -1))
    return _greedy_build_fast(candidates, deadline)

def _bisection_greedy(deadline):
    """
    Greedy starting from middle values.
    """
    candidates = list(range(128, 257)) + list(range(1, 128))
    return _greedy_build_fast(candidates, deadline)

def construct():
    """
    Return a strictly increasing sequence of integers in [1,256] where all 
    nonempty consecutive block sums are distinct. Maximize sequence length.
    """
    start_time = time.time()
    deadline = start_time + 75.0  # Leave 15s safety margin
    
    best_seq = []
    best_len = 0
    
    # Strategy 1: Standard greedy (1 to 256)
    seq = _greedy_build_fast(range(1, 257), deadline)
    if len(seq) > best_len:
        best_seq = seq
        best_len = len(seq)
    
    # Strategy 2: Reverse greedy (256 to 1)
    if time.time() < deadline:
        seq = _greedy_build_fast(range(256, 0, -1), deadline)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Strategy 3: Bisection greedy
    if time.time() < deadline:
        seq = _greedy_build_fast(list(range(128, 257)) + list(range(1, 128)), deadline)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Strategy 4: Multiple random greedy attempts
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            seq = _random_greedy(deadline)
            if len(seq) > best_len:
                best_seq = seq
                best_len = len(seq)
    
    # Strategy 5: Local search on best sequence
    if time.time() < deadline - 5.0:
        improved = _local_search(best_seq, deadline)
        if len(improved) > best_len:
            best_seq = improved
            best_len = len(improved)
    
    # Final validation
    if not _check_valid(best_seq):
        # Fallback to the simple greedy which is guaranteed valid
        best_seq = _greedy_build_fast(range(1, 257))
    
    return best_seq

if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Sequence: {result}")
