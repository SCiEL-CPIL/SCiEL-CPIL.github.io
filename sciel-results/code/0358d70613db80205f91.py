import time
import random

N = 256

def _can_add(x, A, sums):
    """Check if x can be added to A while maintaining Sidon property."""
    if 2 * x in sums:
        return False
    for a in A:
        if x + a in sums:
            return False
    return True

def _get_new_sums(x, A):
    """Get the set of new sums introduced by adding x to A."""
    new_sums = {2 * x}
    for a in A:
        new_sums.add(x + a)
    return new_sums

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered pair sums distinct)."""
    if len(selected) != len(set(selected)):
        return False
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check if selected is inclusion-maximal."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_max_coverage(deadline):
    """
    Construct a maximal Sidon set by greedily picking the element
    that blocks the most other unblocked elements.
    """
    A = []
    sums = set()
    unblocked = set(range(1, N + 1))
    
    while unblocked and time.time() < deadline:
        best_x = None
        max_blocked = -1
        best_new_sums = None
        best_A = None
        
        # Candidate selection: iterate through unblocked elements
        for x in list(unblocked):
            if time.time() > deadline:
                break
            
            if not _can_add(x, A, sums):
                continue
            
            # Calculate blocking power
            new_sums = _get_new_sums(x, A)
            all_sums = sums | new_sums
            new_A = A + [x]
            
            blocked_count = 0
            for y in unblocked:
                if y == x:
                    continue
                # Check if y is blocked by the new configuration
                if 2 * y in all_sums:
                    blocked_count += 1
                    continue
                for a in new_A:
                    if y + a in all_sums:
                        blocked_count += 1
                        break
            
            # Tie-breaking: prefer smaller x or random? 
            if blocked_count > max_blocked or (blocked_count == max_blocked and (best_x is None or random.random() < 0.5)):
                max_blocked = blocked_count
                best_x = x
                best_new_sums = new_sums
                best_A = new_A
        
        if best_x is None:
            # No valid move found? Should not happen if unblocked is not empty and A is Sidon,
            # unless we are in a dead end where no unblocked element can be added?
            break
        
        # Commit the choice
        A = best_A
        sums.update(best_new_sums)
        unblocked.discard(best_x)
        
        # Update unblocked: remove any elements that are now blocked
        new_unblocked = set()
        for y in unblocked:
            if _can_add(y, A, sums):
                new_unblocked.add(y)
        unblocked = new_unblocked
        
    return A

def _local_search_reduce(A, deadline):
    """
    Try to improve the set A by removing elements while maintaining maximality.
    This is the 'lazy deletion' / minimum vertex cover approach.
    """
    A = list(A)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to avoid bias
        indices = list(range(len(A)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            candidate = A[:i] + A[i+1:]
            if not candidate:
                continue
            
            # Check Sidon property
            if not _is_valid_sidon(candidate):
                continue
            
            # Compute sums for candidate
            sums = set()
            n = len(candidate)
            valid = True
            for k in range(n):
                a = candidate[k]
                for m in range(k, n):
                    s = a + candidate[m]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check maximality
            if _is_maximal(candidate, sums):
                A = candidate
                improved = True
                break
                
    return A

def construct():
    start_time = time.time()
    # 60s limit, leave 5s margin
    deadline = start_time + 55.0
    
    best_A = None
    best_size = N + 1
    
    # Run multiple trials of the greedy max-coverage algorithm
    # Randomness in tie-breaking helps explore different solutions
    for trial in range(100):
        if time.time() > deadline:
            break
        
        A = _greedy_max_coverage(deadline)
        
        if not A:
            continue
            
        if not _is_valid_sidon(A):
            continue
            
        # Ensure maximality
        # The greedy process should produce a maximal set, but let's be safe
        # If it's not maximal, we might have a bug or it stopped early.
        # For now, assume the greedy loop handles it.
        
        # Try to reduce size
        A_reduced = _local_search_reduce(A, deadline)
        
        if _is_valid_sidon(A_reduced) and _is_maximal(A_reduced, set()):
            # Recompute sums for maximality check just to be sure
            sums = set()
            n = len(A_reduced)
            for k in range(n):
                a = A_reduced[k]
                for m in range(k, n):
                    sums.add(a + A_reduced[m])
            if _is_maximal(A_reduced, sums):
                if len(A_reduced) < best_size:
                    best_size = len(A_reduced)
                    best_A = A_reduced
            
        # Also consider the unreduced version if reduction failed or didn't improve
        if len(A) < best_size:
            sums = set()
            n = len(A)
            for k in range(n):
                a = A[k]
                for m in range(k, n):
                    sums.add(a + A[m])
            if _is_maximal(A, sums):
                best_size = len(A)
                best_A = A
        
    if best_A is None:
        # Fallback: simple greedy
        A = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add(x, A, sums):
                A.append(x)
                sums.update(_get_new_sums(x, A))
        best_A = A
        best_size = len(A)
        
    return best_A
