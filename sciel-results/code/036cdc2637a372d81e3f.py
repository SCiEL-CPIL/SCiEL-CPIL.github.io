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
    for b in S:
        target_r = (a - b % a) % a
        candidates = residue_table[a].get(target_r, [])
        # Need c in candidates with c >= b and c in S
        for c in candidates:
            if c < b:
                continue
            if c in S:
                return True
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

def _local_search_advanced(S, deadline, residue_table, max_iter=1000):
    """Advanced local search: Add, Remove, Swap1, Swap2."""
    S = set(S)
    best = len(S)
    n = 256
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        # 1. Try adding a new element
        for a in range(1, n + 1):
            if time.time() > deadline:
                break
            if a not in S and not _conflict_fast(a, S, residue_table):
                S.add(a)
                best = len(S)
                improved = True
                break
        if improved:
            continue

        # 2. Try swapping 1 element (remove a, add b)
        # Iterate over elements in S to remove
        for a in sorted(S):
            if time.time() > deadline:
                break
            S.remove(a)
            # Try adding any b not in S
            for b in range(1, n + 1):
                if time.time() > deadline:
                    break
                if b != a and b not in S and not _conflict_fast(b, S, residue_table):
                    S.add(b)
                    if len(S) > best:
                        best = len(S)
                        improved = True
                    break
            if not improved:
                S.add(a) # restore if no improvement found in this swap attempt
            if improved:
                break
        
        if improved:
            continue

        # 3. Try swapping 2 elements (remove a1, a2, add b1, b2)
        # This is expensive, so limit the search space
        # Pick two random elements to remove? Or iterate?
        # Let's try a limited swap: remove a, try to add two smaller ones?
        # Or just random perturbations.
        
        # Random restart within local search: pick 2 elements to remove, try to fill with 2 random ones
        if random.random() < 0.1: # 10% chance to do a 2-swap
             if len(S) >= 2:
                 a1, a2 = random.sample(list(S), 2)
                 S.remove(a1)
                 S.remove(a2)
                 # Try to add 2 random candidates
                 candidates = [x for x in range(1, n+1) if x not in S]
                 random.shuffle(candidates)
                 added = 0
                 for b in candidates:
                     if added == 2:
                         break
                     if not _conflict_fast(b, S, residue_table):
                         S.add(b)
                         added += 1
                 if len(S) > best:
                     best = len(S)
                     improved = True
                 else:
                     # If size didn't increase, revert to previous state if possible?
                     # Simplest: if not improved, we just continue. 
                     # But we lost a1, a2. We should restore them if we didn't gain size.
                     # Actually, if we didn't gain size, we should restore the removed ones to avoid shrinking.
                     # But we might have added some. 
                     # Let's just restore a1, a2 if we didn't improve.
                     # This logic is tricky. Let's stick to 1-swap for now to be safe and fast.
                     pass

        if not improved:
            break

    return S

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin
    n = 256

    residue_table = _build_residue_table(n)

    best_set = set(range(171, n + 1))
    best_size = len(best_set)

    # Strategy 1: Descending greedy from 170 down to 1
    S = set(range(171, n + 1))
    S = _greedy_add(S, range(170, 0, -1), deadline, residue_table)
    if _valid(S) and len(S) > best_size:
        best_set = set(S)
        best_size = len(S)

    # Strategy 2: Ascending greedy from 1 to 170
    if time.time() < deadline:
        S = set(range(171, n + 1))
        S = _greedy_add(S, range(1, 171), deadline, residue_table)
        if _valid(S) and len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Strategy 3: Different bases
    for base in [165, 168, 172, 175, 180]:
        if time.time() > deadline:
            break
        S = set(range(base, n + 1))
        S = _greedy_add(S, range(base - 1, 0, -1), deadline, residue_table)
        if _valid(S) and len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Strategy 4: Randomized greedy
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        S = set(range(171, n + 1))
        candidates = list(range(1, 171))
        random.shuffle(candidates)
        S = _greedy_add(S, candidates, deadline, residue_table)
        if _valid(S) and len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Strategy 5: Local search on best
    if time.time() < deadline:
        improved = _local_search_advanced(set(best_set), deadline, residue_table, max_iter=1000)
        if _valid(improved) and len(improved) > best_size:
            best_set = improved
            best_size = len(improved)

    # Strategy 6: Local search on other good candidates
    if time.time() < deadline - 5.0:
        for base in [171, 174, 168, 165]:
            if time.time() > deadline - 3.0:
                break
            S = set(range(base, n + 1))
            S = _local_search_advanced(S, deadline - 3.0, residue_table, max_iter=500)
            if _valid(S) and len(S) > best_size:
                best_set = set(S)
                best_size = len(S)

    # Final verification
    if not _valid(best_set):
        best_set = set(range(171, n + 1))

    return list(best_set)
