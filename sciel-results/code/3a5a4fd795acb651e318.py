import time
import random
import bisect

def _check_append_validity(selected, x, sums):
    """Check if appending x to sorted selected is valid. Returns (valid, new_sums) or (False, None)."""
    # New sums: x itself, and x + each suffix sum of selected
    # Suffix sums of selected: selected[-1], selected[-1]+selected[-2], ...
    new_sums = [x]
    current_sum = x
    for v in reversed(selected):
        current_sum += v
        new_sums.append(current_sum)
    
    for s in new_sums:
        if s in sums:
            return False, None
    return True, new_sums

def _check_insert_validity(selected, pos, x, sums):
    """Check if inserting x at position pos in sorted selected is valid.
    Returns (valid, new_sums_to_add, sums_to_remove) or (False, None, None).
    
    When inserting x at pos, the affected consecutive blocks are those that
    span the insertion point. The new sums are:
    - x itself
    - x + suffix of selected[pos:]
    - prefix of selected[:pos] + x
    - prefix of selected[:pos] + x + suffix of selected[pos:]
    - Also, old sums that are no longer valid need to be removed, but for
      simplicity we just check if new sums collide with existing sums.
    
    Actually, for a proper check we need to consider that some old sums
    remain valid and some new sums are created. The cleanest approach:
    compute all sums of the new sequence and check for duplicates.
    But that's O(n^2). Let's be smarter.
    
    When we insert x at position pos in a sequence S:
    - Old consecutive blocks that don't span the insertion point remain unchanged
    - New consecutive blocks are those that include x
    - A new block includes x iff it starts at i <= pos and ends at j >= pos
    - The sum of such a block = (sum of S[i:pos]) + x + (sum of S[pos:j+1])
    
    So new sums = {prefix_i + x + suffix_j : 0 <= i <= pos, pos <= j < n}
    where prefix_i = sum of S[0:i] (with prefix_0 = 0) and suffix_j = sum of S[j:n]
    Wait, let me redefine. Let S = [s0, s1, ..., s_{n-1}].
    Insert x at position pos (0-indexed), so new sequence is S[0:pos] + [x] + S[pos:n].
    
    A consecutive block in the new sequence that includes x:
    starts at index i (0 <= i <= pos) and ends at index j (pos <= j <= n)
    in the new sequence. In terms of original S:
    - The part before x is S[i:pos], sum = P[pos] - P[i] where P is prefix sum of S
    - x itself
    - The part after x is S[pos:j], sum = P[j] - P[pos] where j goes from pos to n
    
    So new_sum = (P[pos] - P[i]) + x + (P[j] - P[pos]) = P[j] - P[i] + x
    
    where 0 <= i <= pos and pos <= j <= n.
    
    We need to check if any of these new sums collide with existing sums (that don't
    include x in the original) or with each other.
    
    For efficiency, we can:
    1. Compute prefix sums of S
    2. For each i in [0, pos], compute base = P[pos] - P[i] + x
       For each j in [pos, n], check if base + (P[j] - P[pos]) is in sums
    3. Also check that new sums don't collide with each other
    """
    n = len(selected)
    # Compute prefix sums
    P = [0] * (n + 1)
    for i in range(n):
        P[i + 1] = P[i] + selected[i]
    
    new_sums = set()
    # For each start i in [0, pos] and end j in [pos, n]
    for i in range(pos + 1):
        base = P[pos] - P[i] + x
        for j in range(pos, n + 1):
            s = base + (P[j] - P[pos])
            if s in sums:
                return False, None, None
            if s in new_sums:
                return False, None, None
            new_sums.add(s)
    
    return True, new_sums, None

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

def _get_all_sums(candidate):
    """Compute all consecutive block sums of candidate."""
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            sums.add(s)
    return sums

def _greedy_build(order):
    """Greedy build: for each x in order, try to insert into sorted selected."""
    selected = []
    sums = set()
    for x in order:
        if x in sums:
            continue
        # Find insertion point
        pos = bisect.bisect_left(selected, x)
        if pos < len(selected) and selected[pos] == x:
            continue
        
        if pos == len(selected):
            # Appending
            valid, new_sums = _check_append_validity(selected, x, sums)
            if valid:
                selected.append(x)
                sums.update(new_sums)
        else:
            # Inserting in the middle
            valid, new_sums, _ = _check_insert_validity(selected, pos, x, sums)
            if valid:
                selected.insert(pos, x)
                sums.update(new_sums)
    return selected

def _local_search(selected, available, deadline, rng):
    """Try to improve the sequence by inserting available numbers."""
    current = selected[:]
    sums = _get_all_sums(current)
    current_set = set(current)
    available_set = set(available)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        avail_list = list(available_set)
        rng.shuffle(avail_list)
        
        # Sort by some heuristic: try numbers that are "less conflicting" first
        # For now, just try in random order
        for x in avail_list:
            if x in current_set:
                continue
            
            pos = bisect.bisect_left(current, x)
            if pos < len(current) and current[pos] == x:
                continue
            
            if pos == len(current):
                valid, new_sums = _check_append_validity(current, x, sums)
            else:
                valid, new_sums, _ = _check_insert_validity(current, pos, x, sums)
            
            if valid:
                if pos == len(current):
                    current.append(x)
                else:
                    current.insert(pos, x)
                sums.update(new_sums)
                current_set.add(x)
                available_set.remove(x)
                improved = True
    
    return current

def _try_remove_and_reinsert(current, deadline, rng):
    """Try removing one element and reinserting it elsewhere, or replacing it."""
    if len(current) < 2:
        return current
    
    sums = _get_all_sums(current)
    current_set = set(current)
    
    # Try removing each element and see if we can add something better
    indices = list(range(len(current)))
    rng.shuffle(indices)
    
    best = current[:]
    best_sums = sums
    
    for idx in indices:
        if time.time() > deadline:
            break
        
        # Remove element at idx
        removed = current[idx]
        new_current = current[:idx] + current[idx+1:]
        
        if not new_current:
            continue
        
        # Recompute sums for new_current
        new_sums = _get_all_sums(new_current)
        
        # Try to add back removed or try other available numbers
        available = set(range(1, 257)) - set(new_current)
        
        # First try adding back the removed element
        pos = bisect.bisect_left(new_current, removed)
        if pos < len(new_current) and new_current[pos] == removed:
            # Can't add back in same position, skip
            pass
        else:
            if pos == len(new_current):
                valid, add_sums = _check_append_validity(new_current, removed, new_sums)
            else:
                valid, add_sums, _ = _check_insert_validity(new_current, pos, removed, new_sums)
            
            if valid:
                # Same length, no improvement
                continue
        
        # Try adding other available numbers to new_current
        avail_list = list(available)
        rng.shuffle(avail_list)
        
        for x in avail_list:
            pos = bisect.bisect_left(new_current, x)
            if pos < len(new_current) and new_current[pos] == x:
                continue
            
            if pos == len(new_current):
                valid, add_sums = _check_append_validity(new_current, x, new_sums)
            else:
                valid, add_sums, _ = _check_insert_validity(new_current, pos, x, new_sums)
            
            if valid:
                if pos == len(new_current):
                    candidate = new_current + [x]
                else:
                    candidate = new_current[:pos] + [x] + new_current[pos:]
                
                if len(candidate) > len(best):
                    best = candidate
                    best_sums = new_sums | add_sums
                break  # Only add one element per removal
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    rng = random.Random(42)
    
    # 1. Deterministic baselines with optimized greedy
    for order in [list(range(1, 257)), list(range(256, 0, -1))]:
        seq = _greedy_build(order)
        if len(seq) > len(best_seq):
            best_seq = seq
    
    # 2. Biased random orders
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Mix of strategies
        if trial % 3 == 0:
            # Random
            order = list(range(1, 257))
            rng.shuffle(order)
        elif trial % 3 == 1:
            # Descending with some randomness
            order = list(range(256, 0, -1))
            for i in range(len(order) - 1):
                if rng.random() < 0.1:
                    j = rng.randint(i + 1, len(order) - 1)
                    order[i], order[j] = order[j], order[i]
        else:
            # Ascending with some randomness
            order = list(range(1, 257))
            for i in range(len(order) - 1):
                if rng.random() < 0.1:
                    j = rng.randint(i + 1, len(order) - 1)
                    order[i], order[j] = order[j], order[i]
        
        seq = _greedy_build(order)
        if len(seq) > len(best_seq):
            best_seq = seq
    
    # 3. Local search on best sequences
    if time.time() < deadline:
        available = set(range(1, 257)) - set(best_seq)
        improved = _local_search(best_seq, available, deadline, rng)
        if len(improved) > len(best_seq):
            best_seq = improved
    
    # 4. More random restarts with local search
    while time.time() < deadline:
        # Generate random order
        order = list(range(1, 257))
        rng.shuffle(order)
        
        seq = _greedy_build(order)
        
        if len(seq) > len(best_seq):
            best_seq = seq
        
        # Local search on this sequence
        if time.time() < deadline:
            available = set(range(1, 257)) - set(seq)
            improved = _local_search(seq, available, deadline, rng)
            if len(improved) > len(best_seq):
                best_seq = improved
        
        # Also try local search on best_seq periodically
        if len(best_seq) > len(seq) and rng.random() < 0.3:
            if time.time() < deadline:
                available = set(range(1, 257)) - set(best_seq)
                improved = _local_search(best_seq, available, deadline, rng)
                if len(improved) > len(best_seq):
                    best_seq = improved
    
    # 5. Final polish: try remove-and-reinsert on best_seq
    if time.time() < deadline - 5:
        polished = _try_remove_and_reinsert(best_seq, deadline, rng)
        if len(polished) > len(best_seq):
            best_seq = polished
    
    return best_seq
