"""Two Sidon sets with disjoint difference sets – Bitset + multi-start local search."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # safety margin before 60s kill

def _can_add_to_sidon(elem, set_elems, sum_bitset, diff_bitset, other_diff_bitset):
    """Check if elem can be added to set_elems maintaining Sidon + disjoint diffs."""
    # Check pair sums: elem + a for all a in set_elems, and elem+elem
    # Sum bitset tracks which sums are already used (1-indexed sums from 2 to 512)
    # Diff bitset tracks nonzero differences (1 to 255)
    
    # elem + elem
    s2 = elem + elem
    if s2 <= 512 and (sum_bitset >> s2) & 1:
        return False
    
    # elem + a for each a in set_elems
    for a in set_elems:
        s = elem + a
        if (sum_bitset >> s) & 1:
            return False
        # Check difference: |elem - a|
        d = abs(elem - a)
        if d > 0 and ((diff_bitset >> d) & 1 or (other_diff_bitset >> d) & 1):
            return False
    return True

def _add_to_sidon(elem, set_elems, sum_bitset, diff_bitset):
    """Update bitsets after adding elem to set."""
    s2 = elem + elem
    sum_bitset |= (1 << s2)
    for a in set_elems:
        s = elem + a
        sum_bitset |= (1 << s)
        d = abs(elem - a)
        if d > 0:
            diff_bitset |= (1 << d)
    return sum_bitset, diff_bitset

def _greedy_build(candidates, order, seed_diff_bitset=0):
    """Greedy build using given candidate list and order."""
    A = []
    B = []
    sumA = 0
    sumB = 0
    diffA = 0
    diffB = 0
    
    for x in order:
        # Try adding to A
        if len(A) < MAX_EACH:
            ok = True
            s2 = x + x
            if (sumA >> s2) & 1:
                ok = False
            else:
                for a in A:
                    s = x + a
                    if (sumA >> s) & 1:
                        ok = False
                        break
                    d = abs(x - a)
                    if d > 0 and ((diffA >> d) & 1 or (diffB >> d) & 1):
                        ok = False
                        break
            if ok:
                s2 = x + x
                sumA |= (1 << s2)
                for a in A:
                    s = x + a
                    sumA |= (1 << s)
                    d = abs(x - a)
                    if d > 0:
                        diffA |= (1 << d)
                A.append(x)
                continue
        
        # Try adding to B
        if len(B) < MAX_EACH:
            ok = True
            s2 = x + x
            if (sumB >> s2) & 1:
                ok = False
            else:
                for b in B:
                    s = x + b
                    if (sumB >> s) & 1:
                        ok = False
                        break
                    d = abs(x - b)
                    if d > 0 and ((diffB >> d) & 1 or (diffA >> d) & 1):
                        ok = False
                        break
            if ok:
                s2 = x + x
                sumB |= (1 << s2)
                for b in B:
                    s = x + b
                    sumB |= (1 << s)
                    d = abs(x - b)
                    if d > 0:
                        diffB |= (1 << d)
                B.append(x)
    
    return A, B

def _local_search(A, B, diffA, diffB, sumA, sumB, time_limit):
    """Local search: try swapping elements to increase total size."""
    A = list(A)
    B = list(B)
    diffA = diffA
    diffB = diffB
    sumA = sumA
    sumB = sumB
    start = time.time()
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        total = len(A) + len(B)
        
        # Try adding new elements
        for side in (0, 1):
            if side == 0:
                cur_set = A
                cur_sum = sumA
                cur_diff = diffA
                other_diff = diffB
            else:
                cur_set = B
                cur_sum = sumB
                cur_diff = diffB
                other_diff = diffA
            
            if len(cur_set) >= MAX_EACH:
                continue
                
            for x in range(1, N + 1):
                if x in cur_set:
                    continue
                ok = True
                s2 = x + x
                if (cur_sum >> s2) & 1:
                    ok = False
                else:
                    for a in cur_set:
                        s = x + a
                        if (cur_sum >> s) & 1:
                            ok = False
                            break
                        d = abs(x - a)
                        if d > 0 and ((cur_diff >> d) & 1 or (other_diff >> d) & 1):
                            ok = False
                            break
                if ok:
                    if side == 0:
                        sumA |= (1 << s2)
                        for a in A:
                            sumA |= (1 << (x + a))
                            d = abs(x - a)
                            if d > 0:
                                diffA |= (1 << d)
                        A.append(x)
                    else:
                        sumB |= (1 << s2)
                        for b in B:
                            sumB |= (1 << (x + b))
                            d = abs(x - b)
                            if d > 0:
                                diffB |= (1 << d)
                        B.append(x)
                    improved = True
                    break
        
        if time.time() - start > time_limit:
            break
    
    return A, B

def construct():
    t0 = time.time()
    best_total = 0
    best_result = ([1], [2])
    
    # Multi-start with different random orderings
    random.seed(42)
    
    # First try a deterministic good ordering
    order = list(range(1, N + 1))
    A, B = _greedy_build(None, order)
    # Rebuild diff/sum bitsets
    diffA = 0
    diffB = 0
    sumA = 0
    sumB = 0
    for a in A:
        sumA |= (1 << (a + a))
        for b in A:
            if b >= a:
                sumA |= (1 << (a + b))
            d = abs(a - b)
            if d > 0:
                diffA |= (1 << d)
    for b in B:
        sumB |= (1 << (b + b))
        for c in B:
            if c >= b:
                sumB |= (1 << (b + c))
            d = abs(b - c)
            if d > 0:
                diffB |= (1 << d)
    
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_result = (A[:], B[:])
    
    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed
    if remaining > 5:
        A, B = _local_search(A, B, diffA, diffB, sumA, sumB, remaining - 5)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_result = (A[:], B[:])
    
    # Try several random orderings
    for trial in range(20):
        if time.time() - t0 > DEADLINE - 5:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        A, B = _greedy_build(None, order)
        
        # Rebuild bitsets
        diffA = 0
        diffB = 0
        sumA = 0
        sumB = 0
        for a in A:
            sumA |= (1 << (a + a))
            for b in A:
                if b >= a:
                    sumA |= (1 << (a + b))
                d = abs(a - b)
                if d > 0:
                    diffA |= (1 << d)
        for b in B:
            sumB |= (1 << (b + b))
            for c in B:
                if c >= b:
                    sumB |= (1 << (b + c))
                d = abs(b - c)
                if d > 0:
                    diffB |= (1 << d)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_result = (A[:], B[:])
        
        # Local search on best
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 3
        if remaining > 2 and total >= best_total:
            A2, B2 = _local_search(A, B, diffA, diffB, sumA, sumB, remaining)
            total2 = len(A2) + len(B2)
            if total2 > best_total:
                best_total = total2
                best_result = (A2[:], B2[:])
    
    return list(best_result[0]), list(best_result[1])
# EVOLVE-BLOCK-END
