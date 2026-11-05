# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
K = 3
MAX_CARD = 48

def _find_violations(S):
    """Return list of (element, triple) where element divides sum of triple.
    S is a list of distinct positive integers."""
    violations = []
    n = len(S)
    for i, a in enumerate(S):
        # Check if a divides sum of any 3 distinct OTHER elements
        others = [S[j] for j in range(n) if j != i]
        m = len(others)
        for x in range(m):
            for y in range(x+1, m):
                for z in range(y+1, m):
                    if (others[x] + others[y] + others[z]) % a == 0:
                        violations.append((a, (others[x], others[y], others[z])))
                        break
                else:
                    continue
                break
            else:
                continue
            break
    return violations

def _check_valid(S):
    """Return True if S is a valid set (no element divides sum of any 3 distinct others)."""
    n = len(S)
    for i, a in enumerate(S):
        others = [S[j] for j in range(n) if j != i]
        m = len(others)
        for x in range(m):
            for y in range(x+1, m):
                for z in range(y+1, m):
                    if (others[x] + others[y] + others[z]) % a == 0:
                        return False
    return True

def _try_add(c, S):
    """Check if adding c to S keeps validity. O(n^3) worst case but fast in practice."""
    n = len(S)
    # Check 1: c does not divide sum of any 3 distinct elements of S
    for i in range(n):
        si = S[i]
        for j in range(i+1, n):
            s_ij = si + S[j]
            for k in range(j+1, n):
                if (s_ij + S[k]) % c == 0:
                    return False
    # Check 2: for each a in S, no triple in (S\{a}) U {c} sums to multiple of a
    for a in S:
        # Triples containing c: (c, x, y) where x, y in S, x != a, y != a, x != y
        for i in range(n):
            xi = S[i]
            if xi == a:
                continue
            for j in range(n):
                if j == i or S[j] == a:
                    continue
                if (c + xi + S[j]) % a == 0:
                    return False
    return True

def _repair_backfill(start_set, deadline, rng):
    """Start with start_set, remove elements causing violations, then backfill."""
    S = list(start_set)
    
    # Phase 1: Repair - remove elements that cause violations
    # Iteratively find and remove the most problematic element
    max_repair_iters = 200
    for _ in range(max_repair_iters):
        if time.time() > deadline:
            break
        if not S:
            break
        
        violations = _find_violations(S)
        if not violations:
            break
        
        # Count how many times each element appears as the divisor in violations
        from collections import Counter
        div_counter = Counter()
        for a, triple in violations:
            div_counter[a] += 1
        
        # Remove the element that causes the most violations
        worst_elem = div_counter.most_common(1)[0][0]
        S.remove(worst_elem)
    
    # Phase 2: Backfill - try to add elements from [1, N] not in S
    remaining = [x for x in range(1, N+1) if x not in S]
    rng.shuffle(remaining)
    
    for c in remaining:
        if time.time() > deadline:
            break
        if len(S) >= MAX_CARD:
            break
        if _try_add(c, S):
            S.append(c)
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    rng = random.Random(42)
    
    best_set = []
    best_size = 0
    
    # Strategy: Repair+Backfill from dense upper-half starting sets
    # Upper half numbers are less likely to divide sums of smaller numbers
    
    # Generate several diverse starting sets
    start_sets = []
    
    # Set 1: All numbers from 97 to 192 (96 numbers)
    start_sets.append(list(range(97, 193)))
    
    # Set 2: All numbers from 80 to 192 (113 numbers)
    start_sets.append(list(range(80, 193)))
    
    # Set 3: All numbers from 64 to 192 (129 numbers)
    start_sets.append(list(range(64, 193)))
    
    # Set 4: Random sample of 60 from upper half
    upper = list(range(64, 193))
    rng.shuffle(upper)
    start_sets.append(upper[:60])
    
    # Set 5: Random sample of 50 from upper half
    upper2 = list(range(80, 193))
    rng.shuffle(upper2)
    start_sets.append(upper2[:50])
    
    # Set 6: All numbers from 100 to 192
    start_sets.append(list(range(100, 193)))
    
    # Set 7: Numbers > 96 that are not multiples of small numbers
    start_sets.append([x for x in range(97, 193) if x % 2 != 0])
    
    # Set 8: Random from full range, biased toward upper
    full = list(range(1, 193))
    # Weight toward upper half
    weighted = full + [x for x in range(96, 193)]
    rng.shuffle(weighted)
    start_sets.append(weighted[:70])
    
    for si, ss in enumerate(start_sets):
        if time.time() > deadline:
            break
        
        # Multiple repair passes with different random seeds for backfill order
        for pass_num in range(3):
            if time.time() > deadline:
                break
            
            S = _repair_backfill(ss, deadline, rng)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = list(S)
                
                # If we hit max, we can stop
                if best_size >= MAX_CARD:
                    return best_set
    
    # Final greedy pass if we have time
    if time.time() < deadline - 5:
        candidates = list(range(1, N+1))
        # Bias toward upper half
        candidates = [x for x in candidates if x > 60] + [x for x in candidates if x <= 60]
        rng.shuffle(candidates)
        
        S = []
        for c in candidates:
            if time.time() > deadline:
                break
            if _try_add(c, S):
                S.append(c)
        
        if len(S) > best_size:
            best_set = S
    
    return best_set
# EVOLVE-BLOCK-END
