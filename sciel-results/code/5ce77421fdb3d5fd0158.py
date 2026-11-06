import time
import random

def _valid(S):
    """Check the property for sorted set S. Returns True if valid."""
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i + 1, n):
            b = lst[j]
            for k in range(j, n):
                c = lst[k]
                if (b + c) % a == 0:
                    return False
    return True


def _build_residue_table(n=256):
    """Precompute for each a, a dict mapping residue r -> sorted list of values in [1,n] with that residue mod a."""
    table = {}
    for a in range(1, n + 1):
        table[a] = {}
        for x in range(1, n + 1):
            r = x % a
            if r not in table[a]:
                table[a][r] = []
            table[a][r].append(x)
    return table


def _conflict_fast(a, S, residue_table):
    """Check if adding a to S creates a conflict. S must not contain a.
    Conflict: exists b,c in S with b<=c and (b+c) % a == 0."""
    # For each b in S, we need c in S with c >= b and (b+c) % a == 0
    # (b+c) % a == 0  <=>  c % a == (-b) % a == (a - b%a) % a
    for b in S:
        target_r = (a - b % a) % a
        candidates = residue_table[a].get(target_r, [])
        # Need c in candidates with c >= b and c in S
        # candidates is sorted ascending, so we can break early
        for c in candidates:
            if c < b:
                continue
            if c in S:
                return True
            # Since candidates is sorted and we're looking for c >= b,
            # once c > max(S) we can stop, but let's just check all c >= b
    return False


def _greedy_add(S, candidates, deadline, residue_table):
    """Try to add each candidate in order if conflict-free."""
    S = set(S)
    for a in candidates:
        if time.time() > deadline:
            break
        if a not in S and not _conflict_fast(a, S, residue_table):
            S.add(a)
    return S


def _local_search(S, deadline, max_iter=500, residue_table=None):
    """Local search: try to add elements, then swap to increase set size."""
    S = set(S)
    best = len(S)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        # Phase 1: Try adding a new element (ascending)
        for a in range(1, 257):
            if time.time() > deadline:
                break
            if a not in S and not _conflict_fast(a, S, residue_table):
                S.add(a)
                best = len(S)
                improved = True
                break
        if improved:
            continue
        
        # Phase 2: Try removing one element and adding a different one
        found_swap = False
        for a in sorted(S):
            if time.time() > deadline:
                break
            S.remove(a)
            # Try adding any element not in S
            for b in range(1, 257):
                if time.time() > deadline:
                    break
                if b != a and b not in S and not _conflict_fast(b, S, residue_table):
                    S.add(b)
                    if len(S) > best:
                        best = len(S)
                        improved = True
                    found_swap = True
                    break
            if not found_swap:
                S.add(a)
            if improved:
                break
        if not improved:
            break
    
    return S


def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin
    n = 256

    # Precompute residue table for fast conflict checking
    residue_table = _build_residue_table(n)

    # Base: [171, 256]
    best_set = set(range(171, n + 1))
    best_size = len(best_set)

    # Strategy 1: Descending greedy from 170 down to 129
    S = set(range(171, n + 1))
    S = _greedy_add(S, range(170, 128, -1), deadline, residue_table)
    if _valid(S) and len(S) > best_size:
        best_set = set(S)
        best_size = len(S)

    # Strategy 2: Ascending greedy from 129 to 170
    if time.time() < deadline:
        S = set(range(171, n + 1))
        S = _greedy_add(S, range(129, 171), deadline, residue_table)
        if _valid(S) and len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Strategy 3: Different base thresholds
    for base in [165, 168, 172, 175, 180, 185]:
        if time.time() > deadline:
            break
        S = set(range(base, n + 1))
        below = list(range(base - 1, 128, -1))
        S = _greedy_add(S, below, deadline, residue_table)
        if _valid(S) and len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Strategy 4: Randomized greedy orders (more trials)
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline:
            break
        S = set(range(171, n + 1))
        candidates = list(range(129, 171))
        random.shuffle(candidates)
        S = _greedy_add(S, candidates, deadline, residue_table)
        if _valid(S) and len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Strategy 5: Start from different bases with randomized order
    for base in [168, 171, 174]:
        if time.time() > deadline:
            break
        S = set(range(base, n + 1))
        candidates = list(range(1, base))
        random.shuffle(candidates)
        S = _greedy_add(S, candidates, deadline, residue_table)
        if _valid(S) and len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Local search on best found
    if time.time() < deadline:
        improved = _local_search(set(best_set), deadline, max_iter=500, residue_table=residue_table)
        if _valid(improved) and len(improved) > best_size:
            best_set = improved
            best_size = len(improved)

    # Second round of local search with different starting points
    if time.time() < deadline - 5.0:
        for base in [171, 174, 168]:
            if time.time() > deadline - 3.0:
                break
            S = set(range(base, n + 1))
            S = _local_search(S, deadline - 3.0, max_iter=200, residue_table=residue_table)
            if _valid(S) and len(S) > best_size:
                best_set = set(S)
                best_size = len(S)

    # Final verification
    if not _valid(best_set):
        best_set = set(range(171, n + 1))

    return list(best_set)
