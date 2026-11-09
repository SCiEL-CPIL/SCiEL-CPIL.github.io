"""Rewritten program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # seconds, safety margin for 60s/90s limits


def _compute_states(A, B):
    """Compute sums and diffs for A and B."""
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    # A sums and diffs
    for i in range(len(A)):
        a = A[i]
        sums_A.add(2 * a)
        for j in range(i + 1, len(A)):
            b = A[j]
            sums_A.add(a + b)
            diffs_A.add(b - a)
            diffs_A.add(a - b)
            
    # B sums and diffs
    for i in range(len(B)):
        b = B[i]
        sums_B.add(2 * b)
        for j in range(i + 1, len(B)):
            c = B[j]
            sums_B.add(b + c)
            diffs_B.add(c - b)
            diffs_B.add(b - c)
            
    return sums_A, sums_B, diffs_A, diffs_B


def _can_add_A(x, A, sums_A, diffs_A, diffs_B):
    """Check if x can be added to A."""
    # Check sums
    for a in A:
        s = a + x
        if s in sums_A:
            return False
    if 2 * x in sums_A:
        return False
        
    # Check diffs disjointness with B
    for a in A:
        d1 = x - a
        d2 = a - x
        if d1 in diffs_B or d2 in diffs_B:
            return False
    return True


def _can_add_B(x, B, sums_B, diffs_A, diffs_B):
    """Check if x can be added to B."""
    # Check sums
    for b in B:
        s = b + x
        if s in sums_B:
            return False
    if 2 * x in sums_B:
        return False
        
    # Check diffs disjointness with A
    for b in B:
        d1 = x - b
        d2 = b - x
        if d1 in diffs_A or d2 in diffs_A:
            return False
    return True


def _add_to_A(x, A, sums_A, diffs_A):
    """Add x to A and update states."""
    A.append(x)
    for a in A:
        if a != x:
            sums_A.add(a + x)
            diffs_A.add(x - a)
            diffs_A.add(a - x)
    sums_A.add(2 * x)


def _add_to_B(x, B, sums_B, diffs_B):
    """Add x to B and update states."""
    B.append(x)
    for b in B:
        if b != x:
            sums_B.add(b + x)
            diffs_B.add(x - b)
            diffs_B.add(b - x)
    sums_B.add(2 * x)


def _greedy_fill(order, A, B, sums_A, sums_B, diffs_A, diffs_B, deadline):
    """Greedily fill A and B from order."""
    start = time.time()
    for x in order:
        if time.time() - start > deadline:
            break
            
        # Try A
        if _can_add_A(x, A, sums_A, diffs_A, diffs_B):
            _add_to_A(x, A, sums_A, diffs_A)
            continue
            
        # Try B
        if _can_add_B(x, B, sums_B, diffs_A, diffs_B):
            _add_to_B(x, B, sums_B, diffs_B)
            continue
            
    return A, B, sums_A, sums_B, diffs_A, diffs_B


def construct():
    start = time.time()
    best_A = []
    best_B = []
    best_score = 0
    rng = random.Random(42)
    
    # Initial Greedy
    order = list(range(1, N + 1))
    rng.shuffle(order)
    A, B = [], []
    sums_A, sums_B, diffs_A, diffs_B = set(), set(), set(), set()
    
    A, B, sums_A, sums_B, diffs_A, diffs_B = _greedy_fill(
        order, A, B, sums_A, sums_B, diffs_A, diffs_B, DEADLINE - (time.time() - start)
    )
    
    best_score = len(A) + len(B)
    best_A = A[:]
    best_B = B[:]
    
    # Iterated Local Search
    while time.time() - start < DEADLINE - 2.0:
        # Perturb: remove k random elements
        k = rng.randint(1, 5)
        A_curr = best_A[:]
        B_curr = best_B[:]
        
        removed = []
        for _ in range(k):
            if A_curr:
                idx = rng.randrange(len(A_curr))
                removed.append(('A', A_curr[idx]))
                A_curr.pop(idx)
            elif B_curr:
                idx = rng.randrange(len(B_curr))
                removed.append(('B', B_curr[idx]))
                B_curr.pop(idx)
            else:
                break
        
        # Rebuild states for remaining sets
        sums_A, sums_B, diffs_A, diffs_B = _compute_states(A_curr, B_curr)
        
        # Candidates: all numbers not in A_curr or B_curr
        used = set(A_curr) | set(B_curr)
        candidates = [x for x in range(1, N + 1) if x not in used]
        
        # Shuffle candidates, putting removed ones first to try to re-insert them?
        # Or just shuffle.
        rng.shuffle(candidates)
        
        # Greedy fill
        A_new, B_new, _, _, _, _ = _greedy_fill(
            candidates, A_curr, B_curr, sums_A, sums_B, diffs_A, diffs_B, 
            DEADLINE - (time.time() - start)
        )
        
        score = len(A_new) + len(B_new)
        if score > best_score:
            best_score = score
            best_A = A_new[:]
            best_B = B_new[:]
            
            if best_score >= 2 * MAX_EACH:
                break

    # Ensure nonempty
    if not best_A and not best_B:
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
