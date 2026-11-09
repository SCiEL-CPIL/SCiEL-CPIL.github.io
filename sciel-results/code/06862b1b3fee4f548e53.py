"""B2[2]-set construction for {1..1000} combining fast greedy, targeted local search, and conflict-aware filling."""
import time
import random

N = 1000
TIME_LIMIT = 115  # seconds; leaves margin before the 150 s kill

def _greedy_build(order, counts, in_S, S, time_limit, start_time):
    """
    Greedy construction with fast array-based insertion check.
    """
    for x in order:
        if (time.time() - start_time) > time_limit:
            break
        if in_S[x]:
            continue
        
        # Check 2*x
        if counts[2 * x] >= 2:
            continue
        
        # Check x + y
        valid = True
        for y in S:
            if y == x:
                continue
            if counts[x + y] >= 2:
                valid = False
                break
        
        if valid:
            S.append(x)
            in_S[x] = True
            counts[2 * x] += 1
            for y in S:
                if y != x:
                    counts[x + y] += 1

def _try_remove(x, S, in_S, counts):
    """Remove x from S and update counts."""
    S.remove(x)
    in_S[x] = False
    counts[2 * x] -= 1
    for y in S:
        if y != x:
            counts[x + y] -= 1

def _try_add(x, S, in_S, counts):
    """Add x to S if valid, update counts. Returns True if added."""
    if counts[2 * x] >= 2:
        return False
    for y in S:
        if y == x:
            continue
        if counts[x + y] >= 2:
            return False
    S.append(x)
    in_S[x] = True
    counts[2 * x] += 1
    for y in S:
        if y != x:
            counts[x + y] += 1
    return True

def _is_valid_swap(a, c, S, in_S, counts):
    """
    Check if swapping a (in S) with c (not in S) is valid.
    O(|S|) without dict overhead.
    """
    # Build delta counts for affected sums
    # We need to check if any sum would exceed 2 after the swap
    
    # Sums that lose a contribution from a:
    #   2*a, a+y for y in S, y != a
    # Sums that gain a contribution from c:
    #   2*c, c+y for y in S, y != a
    
    # For each affected sum, compute new count = old count - (a's contribution) + (c's contribution)
    
    # Use a dict to track net deltas for this specific swap
    deltas = {}
    
    # Subtract a's contributions
    deltas[2 * a] = deltas.get(2 * a, 0) - 1
    for y in S:
        if y == a:
            continue
        s = a + y
        deltas[s] = deltas.get(s, 0) - 1
    
    # Add c's contributions
    deltas[2 * c] = deltas.get(2 * c, 0) + 1
    for y in S:
        if y == a:
            continue
        s = c + y
        deltas[s] = deltas.get(s, 0) + 1
    
    # Check validity
    for s, d in deltas.items():
        new_count = counts[s] + d
        if new_count > 2:
            return False
        if new_count < 0:
            return False
    
    return True

def _perform_swap(a, c, S, in_S, counts):
    """Perform the swap: remove a, add c."""
    _try_remove(a, S, in_S, counts)
    _try_add(c, S, in_S, counts)

def _get_blockers(c, S, in_S, counts):
    """
    Get elements in S that block adding c.
    Returns a list of elements that are part of sums with count=2 that c would contribute to.
    """
    blockers = set()
    
    # Check 2*c
    if counts[2 * c] == 2:
        # Find pairs (a,b) in S with a+b = 2*c
        for i in range(len(S)):
            a = S[i]
            b = 2 * c - a
            if b < a:
                continue
            if b > N:
                continue
            if in_S[b]:
                blockers.add(a)
                blockers.add(b)
    
    # Check c + y for y in S
    for y in S:
        s = c + y
        if counts[s] == 2:
            # Find pairs (a,b) in S with a+b = s
            for i in range(len(S)):
                a = S[i]
                b = s - a
                if b < a:
                    continue
                if b > N:
                    continue
                if in_S[b]:
                    blockers.add(a)
                    blockers.add(b)
    
    return list(blockers)

def _local_search(S, in_S, counts, time_limit, start_time, seed=42):
    """
    Local search using targeted swaps based on blocker analysis.
    """
    rng = random.Random(seed)
    best_size = len(S)
    best_S = list(S)
    best_in_S = list(in_S)
    best_counts = list(counts)
    
    improved = True
    swap_attempts = 0
    max_swaps_per_candidate = 200  # More swap attempts per failed addition
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        swap_attempts = 0
        
        # Try to add random unused numbers
        for _ in range(50):  # Try 50 random candidates per iteration
            if (time.time() - start_time) > time_limit:
                break
            
            c = rng.randint(1, N)
            if in_S[c]:
                continue
            
            # Try to add c directly
            if _try_add(c, S, in_S, counts):
                improved = True
                continue
            
            # If add failed, try targeted swaps
            blockers = _get_blockers(c, S, in_S, counts)
            
            if not blockers:
                continue
            
            # Try swapping with blockers first (targeted)
            rng.shuffle(blockers)
            for a in blockers:
                if (time.time() - start_time) > time_limit:
                    break
                if not in_S[a]:
                    continue
                if _is_valid_swap(a, c, S, in_S, counts):
                    _perform_swap(a, c, S, in_S, counts)
                    improved = True
                    break
            
            # If no blocker swap worked, try random swaps
            if not improved:
                for _ in range(max_swaps_per_candidate):
                    if (time.time() - start_time) > time_limit:
                        break
                    a = rng.choice(S)
                    if _is_valid_swap(a, c, S, in_S, counts):
                        _perform_swap(a, c, S, in_S, counts)
                        improved = True
                        break
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = list(S)
                best_in_S = list(in_S)
                best_counts = list(counts)
    
    # Restore best state
    S[:] = best_S
    for i in range(N + 1):
        in_S[i] = best_in_S[i]
    for i in range(2 * N + 1):
        counts[i] = best_counts[i]
    
    return S, in_S, counts

def _fill_remaining(S, in_S, counts, time_limit, start_time):
    """
    Try to add remaining elements in descending order (conflict-aware filling).
    """
    for x in range(N, 0, -1):
        if (time.time() - start_time) > time_limit:
            break
        if in_S[x]:
            continue
        
        if _try_add(x, S, in_S, counts):
            pass
    
    return S, in_S, counts

def construct():
    start_time = time.time()
    
    S = []
    in_S = [False] * (N + 1)
    counts = [0] * (2 * N + 1)
    
    # Step 1: Multiple greedy constructions with different orderings
    # Try increasing order
    order1 = list(range(1, N + 1))
    _greedy_build(order1, counts, in_S, S, TIME_LIMIT, start_time)
    best_size = len(S)
    best_S = list(S)
    best_in_S = list(in_S)
    best_counts = list(counts)
    
    # Reset and try decreasing order
    if (time.time() - start_time) < TIME_LIMIT - 20:
        S.clear()
        in_S = [False] * (N + 1)
        counts = [0] * (2 * N + 1)
        order2 = list(range(N, 0, -1))
        _greedy_build(order2, counts, in_S, S, TIME_LIMIT, start_time)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            best_in_S = list(in_S)
            best_counts = list(counts)
    
    # Try randomized orders
    rng = random.Random(123)
    for _ in range(10):
        if (time.time() - start_time) > TIME_LIMIT - 30:
            break
        S.clear()
        in_S = [False] * (N + 1)
        counts = [0] * (2 * N + 1)
        order = list(range(1, N + 1))
        if rng.random() < 0.5:
            rng.shuffle(order)
            order.sort(key=lambda x: (-x + rng.random() * 200))
        else:
            rng.shuffle(order)
        
        _greedy_build(order, counts, in_S, S, TIME_LIMIT, start_time)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            best_in_S = list(in_S)
            best_counts = list(counts)
    
    if (time.time() - start_time) > TIME_LIMIT - 15:
        return best_S
    
    # Step 2: Local search to improve via targeted swaps
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5
    if remaining_time > 10:
        S = list(best_S)
        in_S = list(best_in_S)
        counts = list(best_counts)
        S, in_S, counts = _local_search(S, in_S, counts, remaining_time, start_time, seed=42)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            best_in_S = list(in_S)
            best_counts = list(counts)
    
    if (time.time() - start_time) > TIME_LIMIT - 10:
        return best_S
    
    # Step 3: Fill remaining with conflict-aware descending pass
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 3
    if remaining_time > 5:
        S = list(best_S)
        in_S = list(best_in_S)
        counts = list(best_counts)
        S, in_S, counts = _fill_remaining(S, in_S, counts, remaining_time, start_time)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            best_in_S = list(in_S)
            best_counts = list(counts)
    
    # Step 4: One more local search pass if time remains
    if (time.time() - start_time) < TIME_LIMIT - 5:
        remaining_time = TIME_LIMIT - (time.time() - start_time) - 2
        if remaining_time > 5:
            S = list(best_S)
            in_S = list(best_in_S)
            counts = list(best_counts)
            S, in_S, counts = _local_search(S, in_S, counts, remaining_time, start_time, seed=99)
            if len(S) > best_size:
                best_size = len(S)
                best_S = list(S)
                best_in_S = list(in_S)
                best_counts = list(counts)
    
    return best_S
