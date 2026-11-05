import time
import random
import bisect

def _compute_block_sums(candidate):
    """Compute all non-empty consecutive block sums for a candidate sequence."""
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            sums.add(s)
    return sums

def _is_valid_candidate(candidate):
    """Check if all consecutive block sums are distinct."""
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

def _check_insertion(selected, x):
    """
    Efficiently check if inserting x into sorted selected maintains distinct block sums.
    Only checks new sums that involve x.
    Returns True if valid, False otherwise.
    """
    if not selected:
        return True
    
    pos = bisect.bisect_left(selected, x)
    if pos < len(selected) and selected[pos] == x:
        return False
    
    n = len(selected)
    
    # Precompute prefix sums of selected for efficient range sum queries
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + selected[i]
    
    # Existing sums (not involving x)
    existing_sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += selected[j]
            existing_sums.add(s)
    
    # New sums involving x at position pos:
    # For each start i <= pos and end j >= pos, the block sum includes x
    # Block [i..j] where i <= pos <= j (in the new sequence)
    # In terms of selected indices: 
    #   - If i == pos: block is x + selected[pos..j] for j >= pos
    #   - If j == pos: block is selected[i..pos-1] + x for i <= pos
    #   - If i < pos and j > pos: block is selected[i..pos-1] + x + selected[pos..j]
    
    # Let's enumerate all blocks in the new sequence that include x
    # New sequence has length n+1, x is at index pos
    # A block [i..j] includes x iff i <= pos <= j
    
    # For efficiency, we compute these sums using prefix sums of selected
    # selected indices: 0..n-1
    # In new sequence, selected[k] is at index k if k < pos, and index k+1 if k >= pos
    
    for i in range(0, pos + 1):  # start index in new sequence (0 to pos)
        for j in range(pos, n):  # end index in new sequence (pos to n-1)
            # Block from new-seq index i to j
            # This includes x and some elements from selected
            # Elements from selected in this block:
            #   - From selected[0..i-1] if i > 0: no wait, let me think more carefully
            
            # New sequence: selected[0], ..., selected[pos-1], x, selected[pos], ..., selected[n-1]
            # Index in new seq: 0, 1, ..., pos-1, pos, pos+1, ..., n
            # selected[k] is at new-seq index k if k < pos, and k+1 if k >= pos
            
            # Block [i..j] in new seq includes:
            # - selected[k] for k in [0, pos-1] where new-seq index k is in [i, j], i.e., k in [i, min(j, pos-1)]
            # - x if pos is in [i, j] (which it always is since i <= pos <= j)
            # - selected[k] for k in [pos, n-1] where new-seq index k+1 is in [i, j], i.e., k+1 in [i, j], so k in [max(i-1, pos), j-1]
            
            # Simpler: compute the sum directly
            total = x
            # Add selected elements before x that are in the block
            # selected[k] is at new-seq index k, so if i <= k <= pos-1 and k <= j, include it
            # Since j >= pos, all k in [i, pos-1] are included (if i <= pos-1)
            if i < pos:
                total += prefix[pos] - prefix[i]  # sum of selected[i..pos-1]
            
            # Add selected elements after x that are in the block
            # selected[k] is at new-seq index k+1, so if k+1 >= i and k+1 <= j, include it
            # k+1 <= j => k <= j-1
            # k+1 >= i => k >= i-1
            # But k >= pos always
            # So k in [pos, j-1] if j-1 >= pos
            if j > pos:
                total += prefix[j] - prefix[pos]  # sum of selected[pos..j-1]
            
            if total in existing_sums:
                return False
            existing_sums.add(total)
    
    return True

def _greedy_build(order):
    """Greedy build: iterate through order, add x if it maintains validity."""
    selected = []
    for x in order:
        if _check_insertion(selected, x):
            selected.append(x)
            selected.sort()
    return selected

def _try_insert(selected, x):
    """Try to insert x into sorted selected and check validity."""
    if _check_insertion(selected, x):
        pos = bisect.bisect_left(selected, x)
        return selected[:pos] + [x] + selected[pos:]
    return None

def _local_search(selected, available, deadline):
    """Local search: try to insert available numbers and swap to escape local optima."""
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
        
        # Try swaps: remove one element, try to insert two different ones
        if time.time() < deadline:
            for i in range(len(current)):
                removed = current[i]
                without = current[:i] + current[i+1:]
                for x in list(available_set):
                    new_seq = _try_insert(without, x)
                    if new_seq:
                        # Try to also insert removed or another
                        current = new_seq
                        available_set.remove(x)
                        # Try to re-insert removed
                        new_seq2 = _try_insert(current, removed)
                        if new_seq2:
                            current = new_seq2
                        else:
                            # Try to insert another available element
                            for y in list(available_set):
                                if y == removed:
                                    continue
                                new_seq3 = _try_insert(current, y)
                                if new_seq3:
                                    current = new_seq3
                                    available_set.remove(y)
                                    break
                        available_set.add(removed)
                        improved = True
                        break
                if time.time() > deadline:
                    break
    return current

def _dfs_backtrack(selected, available, max_depth, deadline, memo=None):
    """
    DFS backtracking search with incremental sum checking.
    """
    if memo is None:
        memo = {}
    
    if len(selected) >= max_depth:
        return selected[:]
    
    if time.time() > deadline:
        return selected[:]
    
    # Memoization key: use last few elements and length as a heuristic key
    key = (len(selected), tuple(selected[-3:]) if len(selected) >= 3 else tuple(selected))
    
    if key in memo:
        return selected[:]
    
    best_result = selected[:]
    avail_list = sorted(available)
    
    for x in avail_list:
        if time.time() > deadline:
            break
        if x in selected:
            continue
        if selected and x <= selected[-1]:
            continue
        
        new_selected = selected + [x]
        if _is_valid_candidate(new_selected):
            new_available = available - {x}
            result = _dfs_backtrack(new_selected, new_available, max_depth, deadline, memo)
            if len(result) > len(best_result):
                best_result = result
    
    memo[key] = True
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    
    # Baseline: greedy builds with different orders
    seq = _greedy_build(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    seq = _greedy_build(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq
    
    # DFS backtracking from a good starting point
    if len(best_seq) > 0:
        available = set(range(1, 257)) - set(best_seq)
        target_len = len(best_seq) + 1
        dfs_result = _dfs_backtrack(best_seq, available, target_len, deadline)
        if len(dfs_result) > len(best_seq):
            best_seq = dfs_result
    
    # Random restarts with greedy + local search
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        order = list(range(1, 257))
        rng.shuffle(order)
        
        seq = _greedy_build(order)
        
        if len(seq) > len(best_seq):
            best_seq = seq
        
        available = set(range(1, 257)) - set(seq)
        improved_seq = _local_search(seq, available, deadline)
        
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq
        
        # Try DFS extension from this improved sequence
        if len(improved_seq) > 0 and time.time() < deadline - 5:
            avail2 = set(range(1, 257)) - set(improved_seq)
            target2 = len(improved_seq) + 1
            dfs_res2 = _dfs_backtrack(improved_seq, avail2, target2, deadline)
            if len(dfs_res2) > len(best_seq):
                best_seq = dfs_res2
    
    return best_seq
