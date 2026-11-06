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

def _count_blocked(x, A, sums, unblocked):
    """
    Count how many elements in 'unblocked' (excluding x) would become blocked
    if x were added to A.
    """
    new_sums = _get_new_sums(x, A)
    all_sums = sums | new_sums
    new_A = A + [x]
    
    blocked_count = 0
    for y in unblocked:
        if y == x:
            continue
        # Check if y is blocked by the new configuration
        # y is blocked if 2y in all_sums OR (y+a in all_sums for some a in new_A)
        if 2 * y in all_sums:
            blocked_count += 1
            continue
        for a in new_A:
            if y + a in all_sums:
                blocked_count += 1
                break
    return blocked_count

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
        # To speed up, we can sample if unblocked is large, but N=256 is small.
        for x in list(unblocked):
            if time.time() > deadline:
                break
            
            if not _can_add(x, A, sums):
                continue
            
            # Calculate blocking power
            blocked = _count_blocked(x, A, sums, unblocked)
            
            # Tie-breaking: prefer smaller x or random? 
            # Smaller x might be better for structural reasons? Or random to avoid loops.
            if blocked > max_blocked or (blocked == max_blocked and (best_x is None or random.random() < 0.5)):
                max_blocked = blocked
                best_x = x
                best_new_sums = _get_new_sums(x, A)
                best_A = A + [x]
        
        if best_x is None:
            # No valid move found? Should not happen if unblocked is not empty and A is Sidon,
            # unless we are in a dead end where no unblocked element can be added?
            # But if an element is unblocked, it *can* be added by definition.
            # So this branch is unlikely unless logic error.
            break
        
        # Commit the choice
        A = best_A
        sums.update(best_new_sums)
        unblocked.discard(best_x)
        
        # Update unblocked: remove any elements that are now blocked
        # We need to re-evaluate all remaining unblocked elements
        new_unblocked = set()
        for y in unblocked:
            if _can_add(y, A, sums):
                new_unblocked.add(y)
        unblocked = new_unblocked
        
    return A

def _local_search_improve(A, deadline):
    """
    Try to improve the set A by swapping elements or removing and re-adding.
    Since A is already maximal, we can't just remove.
    We can try to replace an element x with a new element y that allows removing another element?
    This is complex.
    
    Alternative: Randomized restarts of the greedy process.
    """
    return A

def construct():
    start_time = time.time()
    # 60s limit, leave 5s margin
    deadline = start_time + 55.0
    
    best_A = None
    best_size = N + 1
    
    # Run multiple trials of the greedy max-coverage algorithm
    # Randomness in tie-breaking helps explore different solutions
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Optional: Shuffle the order of iteration in unblocked?
        # The _count_blocked loop iterates over unblocked.
        # We can introduce randomness in the tie-breaking in _greedy_max_coverage.
        
        A = _greedy_max_coverage(deadline)
        
        if A and len(A) < best_size:
            best_size = len(A)
            best_A = A
            # If we get a very small set, we might want to stop early or continue refining?
            # Let's continue to see if we can get even smaller.
            
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
