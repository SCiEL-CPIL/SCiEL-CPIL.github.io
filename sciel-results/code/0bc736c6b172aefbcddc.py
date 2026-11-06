import time
import random
import bisect

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
    """For each a in [1,n], map residue r to sorted list of x in [1,n] with x%a==r."""
    table = {}
    for a in range(1, n + 1):
        d = {}
        for x in range(1, n + 1):
            r = x % a
            if r not in d:
                d[r] = []
            d[r].append(x)
        table[a] = d
    return table


def _conflict_add(a, S, residue_table):
    """Check if adding a to S creates a conflict.
    Conflict: exists b,c in S with a < b <= c and a | (b+c).
    Also need to check: a could be b or c in existing triples? No — the property
    is for ALL triples in the final set. When we add a, new triples are:
      - a < b <= c (a is smallest): need a ∤ (b+c) for all b,c in S with b>a, b<=c
      - a = b < c: need a ∤ (a+c) => a ∤ c => c%a != 0
      - a < b = c: need a ∤ (2b) => 2b%a != 0
    But wait: the condition is "for every a<b<=c in the set, a must NOT divide b+c".
    So when we add a new element x, the new triples involving x are:
      x as the 'a' (smallest): x < b <= c in S, x | (b+c)
      x as the 'b': a < x <= c in S, a | (x+c)
      x as the 'c': a < b <= x in S, a | (b+x)
    We need ALL of these to be false.
    """
    # Case 1: a is the smallest (a < b <= c, both in S)
    for b in S:
        if b <= a:
            continue
        target_r = (a - b % a) % a
        candidates = residue_table[a].get(target_r, [])
        # Need c in S with c >= b and c in candidates
        # Use bisect to find first candidate >= b
        idx = bisect.bisect_left(candidates, b)
        for ci in range(idx, len(candidates)):
            if candidates[ci] in S:
                return True
    # Case 2: a is the middle (b < a <= c, i.e., b in S, b < a, c in S, c >= a)
    # Need: b | (a + c) for some b < a, c >= a in S
    for b in S:
        if b >= a:
            continue
        # Need c in S with c >= a and (a + c) % b == 0
        # c % b == (-a) % b = (b - a%b) % b
        target_r = (b - a % b) % b
        candidates = residue_table[b].get(target_r, [])
        idx = bisect.bisect_left(candidates, a)
        for ci in range(idx, len(candidates)):
            if candidates[ci] in S:
                return True
    # Case 3: a is the largest (b < c <= a, i.e., b,c in S, c <= a, b < c)
    # Need: b | (c + a) for some b < c <= a in S
    for b in S:
        if b >= a:
            continue
        for c in S:
            if c <= b or c > a:
                continue
            if (c + a) % b == 0:
                return True
    return False


def _greedy_add(S, candidates, deadline, residue_table):
    """Try to add each candidate in order if conflict-free."""
    S = set(S)
    for a in candidates:
        if time.time() > deadline:
            break
        if a not in S and not _conflict_add(a, S, residue_table):
            S.add(a)
    return S


def _local_search(S, deadline, max_iter=300, residue_table=None):
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
            if a not in S and not _conflict_add(a, S, residue_table):
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
                if b != a and b not in S and not _conflict_add(b, S, residue_table):
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
                best_set = S
                best_size = len(S)

    # Final verification
    if not _valid(best_set):
        best_set = set(range(171, n + 1))

    return list(best_set)
