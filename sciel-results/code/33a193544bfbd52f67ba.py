"""Two Sidon sets with disjoint difference sets via MIS on conflict graph."""
import time
import random

N = 256
MAX_EACH = 64

def _is_sidon(arr):
    """Check Sidon: all pair sums a+b with a<=b are unique."""
    n = len(arr)
    seen = set()
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            s = ai + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(arr):
    d = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            d.add(abs(arr[j] - ai))
    return d

def _diffs_disjoint(A, B):
    """Check nonzero difference sets are disjoint."""
    dA = _diffs(A)
    dB = _diffs(B)
    return dA.isdisjoint(dB)

def _valid(A, B):
    """Full validity check."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    if not _is_sidon(A) or not _is_sidon(B):
        return False
    if not _diffs_disjoint(A, B):
        return False
    return True

def _conflict_graph(pool):
    """Build conflict graph on pool. Edge (i,j) if {pool[i],pool[j]} is not Sidon.
    Also self-conflict if pool[i]+pool[i] collides (impossible since unique)."""
    m = len(pool)
    # For each pair i<j, check if pool[i]+pool[j] creates a duplicate sum
    # A set S is Sidon iff all sums s1+s2 (s1<=s2 in S) are distinct.
    # Conflict between i and j: there exist k,l in S (not necessarily distinct)
    # such that pool[i]+pool[j] == pool[k]+pool[l] with {k,l} != {i,j}.
    # Simpler: build all pair sums, check for duplicates.
    # For incremental: when adding x to S, x conflicts with a in S if a+x already in sums.
    # For graph: edge(i,j) exists if adding both to empty set creates a conflict.
    # Actually for MIS: we want max subset with no conflicts.
    # Conflict(i,j) means {pool[i], pool[j]} cannot both be in a Sidon set.
    # For two elements, {a,b} is Sidon iff a+a, a+b, b+b are all distinct,
    # which is always true for a != b. So no pair conflict for size 2!
    # Conflicts are 3-way: {a,b,c} not Sidon iff a+b==a+c (impossible for b!=c),
    # or a+b==b+c (iff a==c), or a+a==b+c, etc.
    # So pair conflicts: a+a == b+c requires a,c in set. This is a hyperedge, not edge.
    # For MIS formulation, we need pairwise conflicts.
    # Two elements a,b conflict if there's NO way to have both in any Sidon set.
    # But any two distinct elements form a Sidon set. So no pairwise conflicts!
    # The constraint is global. MIS doesn't directly apply.
    # Alternative: use a different conflict notion.
    # Let me use: conflict(i,j) if |pool[i]-pool[j]| is "bad" in some sense.
    # Actually, let's just use greedy with local density scoring.
    return None  # Not using pure MIS

def _greedy_mis(pool, deadline):
    """Greedy weighted maximal independent set heuristic.
    Since Sidon is a global constraint, use greedy with density-based scoring.
    At each step, pick the element that minimizes future conflicts."""
    m = len(pool)
    # Precompute all pair sums
    # For greedy: maintain current set S, sums of S.
    # Score(x) = - (number of existing a in S where a+x in sums) - (number of future conflicts)
    # Simpler: greedy by local density (prefer isolated elements).
    
    remaining = list(range(m))
    S = []
    sums = set()
    
    while remaining:
        if time.time() > deadline:
            break
        
        best_idx = None
        best_score = -float('inf')
        
        for ri, idx in enumerate(remaining):
            x = pool[idx]
            # Check if x can be added to S (Sidon check)
            can_add = True
            for a in S:
                if a + x in sums:
                    can_add = False
                    break
            if x + x in sums:
                can_add = False
            if not can_add:
                continue
            
            # Score: prefer elements with fewer potential conflicts with remaining
            # Count how many remaining elements would conflict with x
            # (i.e., for y in remaining, y+x in sums_after or y+y == a+x for some a in S)
            # Approximate: count remaining y where y is "close" to x in sum space
            score = 0
            for rj in remaining:
                if rj == ri:
                    continue
                y = pool[rj]
                # If both x and y added, need x+y not in sums, x+x not in sums (ok), y+y not in sums
                # and x+y not equal to any a+b for a,b in S
                # Approximate conflict count
                if x + y in sums:
                    score -= 1
                # Also check if y would conflict with S
                for a in S:
                    if a + y in sums:
                        score -= 0.5
                        break
            # Prefer higher score (fewer conflicts)
            if score > best_score:
                best_score = score
                best_idx = ri
        
        if best_idx is None:
            break
        
        idx = remaining.pop(best_idx)
        x = pool[idx]
        S.append(x)
        for a in S:
            sums.add(a + x)
        sums.add(x + x)
    
    return S

def _local_improve(sets, deadline):
    """Try to add elements or swap to improve combined size."""
    best = [list(sets[0]), list(sets[1])]
    best_score = len(best[0]) + len(best[1])
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding an element
        for x in range(1, 257):
            if time.time() > deadline:
                break
            for side in (0, 1):
                if x in best[side]:
                    continue
                candidate = best[side] + [x]
                if not _is_sidon(candidate):
                    continue
                d_new = _diffs(candidate)
                d_other = _diffs(best[1 - side])
                if d_new & d_other:
                    continue
                best[side] = candidate
                best_score += 1
                improved = True
                break
            if improved:
                break
        
        if improved:
            continue
        
        # Try swap: remove one, add another
        for side in (0, 1):
            if time.time() > deadline:
                break
            for i in range(len(best[side])):
                if time.time() > deadline:
                    break
                removed = best[side][i]
                without = best[side][:i] + best[side][i+1:]
                for x in range(1, 257):
                    if x == removed or x in best[side]:
                        continue
                    candidate = without + [x]
                    if not _is_sidon(candidate):
                        continue
                    d_new = _diffs(candidate)
                    d_other = _diffs(best[1 - side])
                    if d_new & d_other:
                        continue
                    best[side] = candidate
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    best_sets = [[], []]
    best_score = 0
    
    random.seed(42)
    
    # Phase 1: Split into odds/evens for automatic difference disjointness
    # Odd numbers for A, even for B. Differences within each are even, across are odd.
    odds = list(range(1, 257, 2))   # 128 odds
    evens = list(range(2, 257, 2))  # 128 evens
    
    # Build each as Sidon set using greedy MIS
    for r in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Randomize order
        odds_shuffled = odds[:]
        evens_shuffled = evens[:]
        random.shuffle(odds_shuffled)
        random.shuffle(evens_shuffled)
        
        A = _greedy_mis(odds_shuffled, deadline)
        B = _greedy_mis(evens_shuffled, deadline)
        
        # Trim to max 64
        A = A[:MAX_EACH]
        B = B[:MAX_EACH]
        
        if _valid(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_sets = [list(A), list(B)]
        
        # Local improvement
        if time.time() < deadline - 5.0:
            improved = _local_improve([A, B], deadline)
            if _valid(improved[0], improved[1]):
                score = len(improved[0]) + len(improved[1])
                if score > best_score:
                    best_score = score
                    best_sets = [list(improved[0]), list(improved[1])]
    
    # Phase 2: Try mixed assignments (not strictly odds/evens)
    if time.time() < deadline - 15.0:
        all_nums = list(range(1, 257))
        for r in range(10):
            if time.time() > deadline - 10.0:
                break
            
            # Random split
            random.shuffle(all_nums)
            split = random.randint(64, 192)
            pool_A = all_nums[:split]
            pool_B = all_nums[split:]
            
            A = _greedy_mis(pool_A, deadline)
            B = _greedy_mis(pool_B, deadline)
            
            A = A[:MAX_EACH]
            B = B[:MAX_EACH]
            
            if _valid(A, B):
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_sets = [list(A), list(B)]
            
            if time.time() < deadline - 5.0:
                improved = _local_improve([A, B], deadline)
                if _valid(improved[0], improved[1]):
                    score = len(improved[0]) + len(improved[1])
                    if score > best_score:
                        best_score = score
                        best_sets = [list(improved[0]), list(improved[1])]
    
    # Final polish
    if time.time() < deadline - 1.0:
        final = _local_improve(best_sets, deadline)
        if _valid(final[0], final[1]):
            if len(final[0]) + len(final[1]) > best_score:
                best_sets = final
                best_score = len(final[0]) + len(final[1])
    
    # Ensure nonempty
    if not best_sets[0]:
        best_sets[0] = [1]
    if not best_sets[1]:
        best_sets[1] = [2]
    
    return best_sets
