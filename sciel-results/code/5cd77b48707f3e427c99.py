import time
import random
import bisect

def _is_valid_candidate(candidate):
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build(order):
    selected = []
    for x in order:
        candidate = selected + [x]
        candidate.sort()
        if _is_valid_candidate(candidate):
            selected = candidate
    return selected

def _try_insert(selected, x):
    """Try to insert x into sorted selected and check validity."""
    pos = bisect.bisect_left(selected, x)
    if pos < len(selected) and selected[pos] == x:
        return None
    new_candidate = selected[:pos] + [x] + selected[pos:]
    if _is_valid_candidate(new_candidate):
        return new_candidate
    return None

def _local_search(selected, available, deadline):
    """Try to improve the sequence by inserting available numbers."""
    current = selected[:]
    available_set = set(available)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        avail_list = list(available_set)
        random.shuffle(avail_list)
        for x in avail_list:
            if x in current:
                continue
            new_seq = _try_insert(current, x)
            if new_seq:
                current = new_seq
                available_set.remove(x)
                improved = True
    return current

def _dfs_backtrack(seq, available, deadline, best_len):
    """
    DFS with backtracking. State is (tuple(seq), frozenset(available)).
    We try to add elements from available in various orders.
    Memoization: if we've seen this state and didn't find a solution longer than best_len, skip.
    """
    current_len = len(seq)
    if current_len >= best_len:
        return seq[:]
    
    if time.time() > deadline:
        return seq[:]
    
    seq_tuple = tuple(seq)
    avail_frozen = frozenset(available)
    
    # Memo: state -> max length achieved from this state
    if not hasattr(_dfs_backtrack, 'memo'):
        _dfs_backtrack.memo = {}
    
    key = (seq_tuple, avail_frozen)
    if key in _dfs_backtrack.memo:
        if _dfs_backtrack.memo[key] >= best_len:
            return seq[:]
    
    best_from_here = seq[:]
    
    # Try adding each available element
    # Heuristic: try larger elements first to spread sums
    candidates = sorted(available, reverse=True)
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        new_seq = seq + [x]
        # Check validity of new_seq
        if not _is_valid_candidate(new_seq):
            continue
        
        new_available = available - {x}
        
        # Recurse
        result = _dfs_backtrack(new_seq, new_available, deadline, best_len)
        
        if len(result) > len(best_from_here):
            best_from_here = result
        
        # Update memo
        _dfs_backtrack.memo[key] = len(best_from_here)
        
        # If we found a very long sequence, we can prune
        if len(best_from_here) >= len(available) + len(seq):
            break
    
    _dfs_backtrack.memo[key] = len(best_from_here)
    return best_from_here

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    
    # Baselines
    seq = _greedy_build(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    seq = _greedy_build(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    # Random Restarts with Local Search
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Generate a random order
        order = list(range(1, 257))
        rng.shuffle(order)
        
        seq = _greedy_build(order)
        
        if len(seq) > len(best_seq):
            best_seq = seq
            
        # Try to improve the current sequence
        available = set(range(1, 257)) - set(seq)
        improved_seq = _local_search(seq, available, deadline)
        
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq
        
        # DFS backtracking on the improved sequence
        if time.time() < deadline - 5.0:
            _dfs_backtrack.memo = {}
            dfs_result = _dfs_backtrack(improved_seq, set(range(1, 257)) - set(improved_seq), deadline - 3.0, len(best_seq))
            if len(dfs_result) > len(best_seq):
                best_seq = dfs_result
    
    return best_seq
