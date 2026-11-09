import random
import time

N = 256

def is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def is_maximal(S, N):
    """Check if S is inclusion-maximal in [1,N]."""
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if x can be added
        # x is addable if no a in S such that x+a in sums, AND 2x not in sums
        valid = True
        if 2 * x in sums:
            valid = False
        else:
            for a in S:
                if (x + a) in sums:
                    valid = False
                    break
        if valid:
            return False
    return True

def greedy_sidon(order, N):
    """Greedy Sidon set construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def compute_blockers(S, N):
    """
    For each y not in S, compute the set of blockers.
    A blocker is a in S such that y+a is in sums(S).
    Also, if 2y is in sums(S), y is blocked by 'diag'.
    Returns a dict: y -> set of blockers (elements of S) or {'diag'}
    """
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    blockers = {}
    for y in range(1, N + 1):
        if y in S_set:
            continue
        bys = set()
        if 2 * y in sums:
            bys.add('diag')
        for a in S:
            if (y + a) in sums:
                bys.add(a)
        blockers[y] = bys
    return blockers

def reduce_maximal(S, N, time_limit=None, start_time=None):
    """
    Optimized reduction of a maximal Sidon set.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 50.0

    S = list(S)
    if not S:
        return S

    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Compute blockers for current S
        blockers = compute_blockers(S, N)
        
        # If any y has no blockers, S is not maximal (should not happen if we start maximal and only remove valid ones)
        # But if it happens, we stop reducing as we can't guarantee maximality easily without adding.
        # Actually, if we remove x and S becomes non-maximal, we should revert.
        # Our loop only removes if is_maximal(S\{x}) is True.
        
        # Identify critical elements
        # An element x is critical if there exists a y such that blockers[y] == {x}
        # If we remove x, y becomes unblocked (unless 'diag' is also there, but {x} implies no diag).
        critical = set()
        for y, bys in blockers.items():
            if len(bys) == 1:
                # The single blocker is either an element or 'diag'
                b = next(iter(bys))
                if b != 'diag':
                    critical.add(b)
        
        # Candidates are non-critical elements
        candidates = [x for x in S if x not in critical]
        
        # If no candidates, we are stuck with single removals.
        # We might try pair removals or just break.
        # For now, break.
        if not candidates:
            break
            
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            # Try removing x
            candidate_S = [s for s in S if s != x]
            
            # Verify maximality of candidate_S
            # This is the expensive part, but necessary.
            if is_maximal(candidate_S, N):
                S = candidate_S
                improved = True
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    best = None
    best_size = N + 1
    
    # Baseline: Ascending greedy
    order = list(range(1, N + 1))
    S = greedy_sidon(order, N)
    S = reduce_maximal(S, N, time_limit=deadline - start_time - 5, start_time=start_time)
    if len(S) < best_size:
        best = S
        best_size = len(S)
        
    random.seed(42)
    
    # Try multiple random orderings
    # We can afford more trials now because reduction is faster
    for trial in range(10000):
        if time.time() > deadline:
            break
            
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_sidon(order, N)
        
        # Reduce with remaining time budget
        remaining = deadline - time.time()
        if remaining > 5:
            S = reduce_maximal(S, N, time_limit=remaining - 5, start_time=start_time)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
            
            # Optional: If we find a very small set, we can try to polish it further
            # by running reduce_maximal again with a longer time limit if time permits
            # But the loop above already does this.
            
    if best is None:
        best = greedy_sidon(list(range(1, N + 1)), N)
        
    return best
