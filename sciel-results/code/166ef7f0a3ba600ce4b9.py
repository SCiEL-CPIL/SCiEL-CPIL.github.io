# EVOLVE-BLOCK-START
import time
import random

def _try_add_fast(S, x, sum_set, used_diffs):
    """
    Try to add x to Sidon set S given precomputed sum_set (all pair sums in S) 
    and used_diffs (differences that conflict with the other set).
    Returns (new_S, new_sum_set, new_diffs) or (None, None, None).
    """
    # Check Sidon: all sums x + s for s in S must be unique and not in sum_set
    # Also x + x must be unique
    for s in S:
        val = x + s
        if val in sum_set:
            return None, None, None
        sum_set.add(val)
    val = x + x
    if val in sum_set:
        return None, None, None
    sum_set.add(val)
    
    # Check differences
    new_diffs = set()
    for s in S:
        d = abs(x - s)
        if d > 0:
            if d in used_diffs:
                # Roll back sum_set
                for s2 in S:
                    sum_set.discard(x + s2)
                sum_set.discard(val)
                return None, None, None
            new_diffs.add(d)
    
    return S + [x], sum_set, new_diffs

def _remove_from_sets(S, x, sum_set, diffs):
    """Remove x from S, returning updated sum_set and diffs."""
    new_S = [s for s in S if s != x]
    # Remove all sums involving x
    for s in new_S:
        sum_set.discard(x + s)
    sum_set.discard(x + x)
    # Remove diffs involving x
    new_diffs = set()
    for s in new_S:
        d = abs(x - s)
        if d > 0:
            new_diffs.discard(d)
    return new_S, sum_set, diffs - new_diffs

def _greedy_interleaved(universe, deadline, prefer_alternate=True):
    """
    Interleaved greedy: process elements one by one, trying to add to A or B.
    prefer_alternate: if True, alternate which set to try first.
    """
    A = []
    B = []
    sum_set_A = set()
    sum_set_B = set()
    diffs_A = set()
    diffs_B = set()
    
    try_A_first = prefer_alternate
    for x in universe:
        if time.time() > deadline:
            break
        
        # Decide which set to try first
        if try_A_first:
            # Try A first
            new_A, new_sumA, new_diffsA = _try_add_fast(A, x, sum_set_A, diffs_B)
            if new_A is not None:
                A = new_A
                sum_set_A = new_sumA
                diffs_A.update(new_diffsA)
            else:
                # Try B
                new_B, new_sumB, new_diffsB = _try_add_fast(B, x, sum_set_B, diffs_A)
                if new_B is not None:
                    B = new_B
                    sum_set_B = new_sumB
                    diffs_B.update(new_diffsB)
        else:
            # Try B first
            new_B, new_sumB, new_diffsB = _try_add_fast(B, x, sum_set_B, diffs_A)
            if new_B is not None:
                B = new_B
                sum_set_B = new_sumB
                diffs_B.update(new_diffsB)
            else:
                # Try A
                new_A, new_sumA, new_diffsA = _try_add_fast(A, x, sum_set_A, diffs_B)
                if new_A is not None:
                    A = new_A
                    sum_set_A = new_sumA
                    diffs_A.update(new_diffsA)
        
        try_A_first = not try_A_first
    
    return A, B

def _greedy_sequential(order_a, order_b, deadline):
    """Original sequential: build A fully, then B."""
    A = []
    B = []
    sum_set_A = set()
    sum_set_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for x in order_a:
        if time.time() > deadline:
            break
        new_A, new_sumA, new_diffsA = _try_add_fast(A, x, sum_set_A, diffs_B)
        if new_A is not None:
            A = new_A
            sum_set_A = new_sumA
            diffs_A.update(new_diffsA)
    
    for x in order_b:
        if time.time() > deadline:
            break
        new_B, new_sumB, new_diffsB = _try_add_fast(B, x, sum_set_B, diffs_A)
        if new_B is not None:
            B = new_B
            sum_set_B = new_sumB
            diffs_B.update(new_diffsB)
    
    return A, B

def _local_search(A, B, deadline):
    """
    Local search: try removing an element from one set and adding it to the other,
    or trying to add unused elements.
    """
    A = list(A)
    B = list(B)
    sum_set_A = set()
    sum_set_B = set()
    diffs_A = set()
    diffs_B = set()
    
    # Rebuild sum sets and diffs
    for i in range(len(A)):
        for j in range(i, len(A)):
            sum_set_A.add(A[i] + A[j])
    for i in range(len(B)):
        for j in range(i, len(B)):
            sum_set_B.add(B[i] + B[j])
    
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            d = abs(A[i] - A[j])
            if d > 0:
                diffs_A.add(d)
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            d = abs(B[i] - B[j])
            if d > 0:
                diffs_B.add(d)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding unused elements to A or B
        used = set(A) | set(B)
        for x in range(1, 257):
            if x in used:
                continue
            # Try adding to A
            new_A, new_sumA, new_diffsA = _try_add_fast(A, x, sum_set_A, diffs_B)
            if new_A is not None:
                A = new_A
                sum_set_A = new_sumA
                diffs_A.update(new_diffsA)
                improved = True
                break
            # Try adding to B
            new_B, new_sumB, new_diffsB = _try_add_fast(B, x, sum_set_B, diffs_A)
            if new_B is not None:
                B = new_B
                sum_set_B = new_sumB
                diffs_B.update(new_diffsB)
                improved = True
                break
        
        # Try moving elements from A to B or vice versa
        if time.time() > deadline:
            break
        for idx, elem in enumerate(A):
            if time.time() > deadline:
                break
            # Remove elem from A
            new_A, sum_set_A2, diffs_A2 = _remove_from_sets(A, elem, sum_set_A, diffs_A)
            # Try adding to B
            new_B, new_sumB, new_diffsB = _try_add_fast(B, elem, sum_set_B, diffs_A2)
            if new_B is not None:
                A = new_A
                sum_set_A = sum_set_A2
                diffs_A = diffs_A2
                B = new_B
                sum_set_B = new_sumB
                diffs_B.update(new_diffsB)
                improved = True
                break
        if improved:
            continue
        
        for idx, elem in enumerate(B):
            if time.time() > deadline:
                break
            # Remove elem from B
            new_B, sum_set_B2, diffs_B2 = _remove_from_sets(B, elem, sum_set_B, diffs_B)
            # Try adding to A
            new_A, new_sumA, new_diffsA = _try_add_fast(A, elem, sum_set_A, diffs_B2)
            if new_A is not None:
                B = new_B
                sum_set_B = sum_set_B2
                diffs_B = diffs_B2
                A = new_A
                sum_set_A = new_sumA
                diffs_A.update(new_diffsA)
                improved = True
                break
    
    return A, B

def construct():
    start = time.time()
    deadline = start + 70  # Safety margin: 90s kill, leave 20s buffer
    
    best_total = 0
    best_A = []
    best_B = []
    
    # Strategy 1: Interleaved greedy with random orderings
    for trial in range(500):
        if time.time() > deadline:
            break
        
        universe = list(range(1, 257))
        random.shuffle(universe)
        
        A, B = _greedy_interleaved(universe, deadline, prefer_alternate=True)
        total = len(A) + len(B)
        
        if total > best_total:
            best_total = total
            best_A = list(A)
            best_B = list(B)
    
    # Strategy 2: Sequential greedy with random splits
    for trial in range(300):
        if time.time() > deadline:
            break
        
        universe = list(range(1, 257))
        random.shuffle(universe)
        half = len(universe) // 2
        order_a = universe[:half]
        order_b = universe[half:]
        
        A, B = _greedy_sequential(order_a, order_b, deadline)
        total = len(A) + len(B)
        
        if total > best_total:
            best_total = total
            best_A = list(A)
            best_B = list(B)
    
    # Strategy 3: Baseline sequential in natural order
    if time.time() < deadline:
        A, B = _greedy_sequential(list(range(1, 257)), [], deadline)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = list(A)
            best_B = list(B)
    
    # Local search polish on the best solution
    if best_total > 0 and time.time() < deadline:
        A_ls, B_ls = _local_search(best_A, best_B, deadline)
        total_ls = len(A_ls) + len(B_ls)
        if total_ls > best_total:
            best_total = total_ls
            best_A = A_ls
            best_B = B_ls
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
