"""Combining greedy baseline with backtracking refinement to find
longer strictly increasing sequences in [1,256] with distinct
consecutive-block sums."""
import time

def _can_append(values, x, sums):
    """Check if x can be appended without creating duplicate block sums.
    Returns the new sums if valid, else None."""
    new_sums = set()
    total = x
    new_sums.add(total)
    for a in reversed(values):
        total += a
        new_sums.add(total)
    if new_sums & sums:
        return None
    return new_sums

def _greedy(values, sums, start_val, max_val):
    """Greedy: pick smallest valid x >= start_val."""
    for x in range(start_val, max_val + 1):
        new_sums = _can_append(values, x, sums)
        if new_sums is not None:
            values.append(x)
            sums.update(new_sums)
            return True
    return False

def _backtrack_extend(values, sums, min_val, max_val, deadline, best_len):
    """Try to extend the sequence via backtracking. Returns best found."""
    best = list(values)
    best_sums = set(sums)
    
    if time.time() > deadline:
        return best
    
    # Try each candidate in order, with backtracking
    for x in range(min_val, max_val + 1):
        if time.time() > deadline:
            break
        
        new_sums = _can_append(values, x, sums)
        if new_sums is None:
            continue
        
        # Tentatively add x
        values.append(x)
        sums.update(new_sums)
        
        # Recurse: try to extend further
        if len(values) > len(best):
            best = list(values)
            best_sums = set(sums)
        
        # Try to extend further with backtracking
        deeper_best = _backtrack_extend(values, sums, x + 1, max_val, deadline, len(best))
        if len(deeper_best) > len(best):
            best = deeper_best
            best_sums = set(sums)  # approximate; we'll recompute if needed
        
        # Undo
        values.pop()
        for s in new_sums:
            sums.discard(s)
    
    return best

def _backtrack_with_choice(values, sums, min_val, max_val, deadline, best_len, depth=0):
    """Backtracking that tries multiple choices at each level, keeping the best."""
    best = list(values)
    
    if time.time() > deadline:
        return best
    
    if depth > 20:  # Safety limit on recursion depth
        return best
    
    candidates = []
    for x in range(min_val, max_val + 1):
        if time.time() > deadline:
            break
        new_sums = _can_append(values, x, sums)
        if new_sums is not None:
            candidates.append((x, new_sums))
    
    # Try each candidate, pick the one leading to longest sequence
    for x, new_sums in candidates:
        if time.time() > deadline:
            break
        
        values.append(x)
        sums.update(new_sums)
        
        if len(values) > len(best):
            best = list(values)
        
        # Recurse
        result = _backtrack_with_choice(values, sums, x + 1, max_val, deadline, len(best), depth + 1)
        if len(result) > len(best):
            best = result
        
        # Undo
        values.pop()
        for s in new_sums:
            sums.discard(s)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Phase 1: Fast greedy baseline
    values = []
    sums = set()
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        new_sums = _can_append(values, x, sums)
        if new_sums is not None:
            values.append(x)
            sums.update(new_sums)
    
    best = list(values)
    best_len = len(best)
    
    # Phase 2: Backtracking to extend beyond greedy trap
    if len(values) < 256:  # There's room to improve
        min_val = values[-1] + 1 if values else 1
        result = _backtrack_extend(list(values), set(sums), min_val, 256, deadline, best_len)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 3: Try backtracking from a shorter prefix to find a longer overall sequence
    if time.time() < deadline - 10:
        # Try truncating the greedy result and backtracking from there
        for trunc in range(len(values) - 1, max(len(values) - 4, 0), -1):
            if time.time() > deadline - 5:
                break
            prefix = list(values[:trunc])
            prefix_sums = set()
            n = len(prefix)
            for i in range(n):
                s = 0
                for j in range(i, n):
                    s += prefix[j]
                    prefix_sums.add(s)
            
            min_val = prefix[-1] + 1 if prefix else 1
            result = _backtrack_with_choice(prefix, prefix_sums, min_val, 256, deadline, best_len)
            if len(result) > best_len:
                best = result
                best_len = len(result)
    
    return best
