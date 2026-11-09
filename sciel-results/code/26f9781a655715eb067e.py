"""Combined program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # seconds, safety margin for 90s limit


def _compute_states(A, B):
    """Compute sums and diffs for A and B."""
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for i in range(len(A)):
        a = A[i]
        sums_A.add(2 * a)
        for j in range(i + 1, len(A)):
            b = A[j]
            sums_A.add(a + b)
            diffs_A.add(b - a)
            diffs_A.add(a - b)
            
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
    for a in A:
        s = a + x
        if s in sums_A:
            return False
    if 2 * x in sums_A:
        return False
        
    for a in A:
        d1 = x - a
        d2 = a - x
        if d1 in diffs_B or d2 in diffs_B:
            return False
    return True


def _can_add_B(x, B, sums_B, diffs_A, diffs_B):
    """Check if x can be added to B."""
    for b in B:
        s = b + x
        if s in sums_B:
            return False
    if 2 * x in sums_B:
        return False
        
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
            
        if _can_add_A(x, A, sums_A, diffs_A, diffs_B):
            _add_to_A(x, A, sums_A, diffs_A)
            continue
            
        if _can_add_B(x, B, sums_B, diffs_A, diffs_B):
            _add_to_B(x, B, sums_B, diffs_B)
            continue
            
    return A, B, sums_A, sums_B, diffs_A, diffs_B


def _local_improve(A, B, sums_A, sums_B, diffs_A, diffs_B, rng, deadline):
    """Local improvement: try to insert skipped elements."""
    used = set(A) | set(B)
    candidates = [x for x in range(1, N + 1) if x not in used]
    rng.shuffle(candidates)
    
    improved = False
    for x in candidates:
        if time.time() > deadline:
            break
        if _can_add_A(x, A, sums_A, diffs_A, diffs_B):
            _add_to_A(x, A, sums_A, diffs_A)
            improved = True
        elif _can_add_B(x, B, sums_B, diffs_A, diffs_B):
            _add_to_B(x, B, sums_B, diffs_B)
            improved = True
            
    return A, B, sums_A, sums_B, diffs_A, diffs_B, improved


def _validate(A, B):
    """Final validation of the solution."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
        
    sums_A = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums_A:
                return False
            sums_A.add(s)
            
    sums_B = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sums_B:
                return False
            sums_B.add(s)
        
    diffs_A = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                d = A[i] - A[j]
                if d != 0:
                    diffs_A.add(d)
    
    diffs_B = set()
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                d = B[i] - B[j]
                if d != 0:
                    diffs_B.add(d)
    
    if diffs_A & diffs_B:
        return False
        
    return True


def construct():
    start = time.time()
    best_A = []
    best_B = []
    best_score = 0
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy restarts
    for trial in range(50):
        if time.time() - start > DEADLINE - 10.0:
            break
            
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        A, B = [], []
        sums_A, sums_B, diffs_A, diffs_B = set(), set(), set(), set()
        
        A, B, sums_A, sums_B, diffs_A, diffs_B = _greedy_fill(
            order, A, B, sums_A, sums_B, diffs_A, diffs_B,
            DEADLINE - (time.time() - start) - 10.0
        )
        
        # Local improvement
        A, B, sums_A, sums_B, diffs_A, diffs_B, _ = _local_improve(
            A, B, sums_A, sums_B, diffs_A, diffs_B, rng,
            DEADLINE - (time.time() - start) - 5.0
        )
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            
            if best_score >= 2 * MAX_EACH:
                break
    
    # Phase 2: Iterated local search with perturbations
    while time.time() - start < DEADLINE - 2.0:
        if best_score >= 2 * MAX_EACH:
            break
            
        # Perturb: remove k random elements
        k = rng.randint(1, 5)
        A_curr = best_A[:]
        B_curr = best_B[:]
        
        for _ in range(k):
            if A_curr:
                idx = rng.randrange(len(A_curr))
                A_curr.pop(idx)
            elif B_curr:
                idx = rng.randrange(len(B_curr))
                B_curr.pop(idx)
            else:
                break
        
        sums_A, sums_B, diffs_A, diffs_B = _compute_states(A_curr, B_curr)
        
        # Candidates: all numbers not in current sets
        used = set(A_curr) | set(B_curr)
        candidates = [x for x in range(1, N + 1) if x not in used]
        rng.shuffle(candidates)
        
        A_new, B_new, _, _, _, _ = _greedy_fill(
            candidates, A_curr, B_curr, sums_A, sums_B, diffs_A, diffs_B,
            DEADLINE - (time.time() - start) - 1.0
        )
        
        # Local improvement
        A_new, B_new, _, _, _, _ = _local_improve(
            A_new, B_new, sums_A, sums_B, diffs_A, diffs_B, rng,
            DEADLINE - (time.time() - start) - 0.5
        )
        
        score = len(A_new) + len(B_new)
        if score > best_score:
            best_score = score
            best_A = A_new[:]
            best_B = B[:]
    
    # Ensure nonempty
    if not best_A and not best_B:
        best_A = [1]
        best_B = [2]
    
    # Final validation
    if not _validate(best_A, best_B):
        # Fallback to simple valid solution
        best_A = [1, 2, 4, 8, 16, 32, 64, 128]
        best_B = [3, 6, 12, 24, 48, 96]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
