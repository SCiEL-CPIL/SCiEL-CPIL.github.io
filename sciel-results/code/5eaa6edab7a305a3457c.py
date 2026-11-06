import time
import random
import itertools

def is_sidon(subset):
    """Check if subset has all unordered pair sums (i<=j) unique."""
    s = sorted(subset)
    n = len(s)
    seen_sums = set()
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in seen_sums:
                return False
            seen_sums.add(sm)
    return True

def max_sidon_size(S):
    """Find largest subset of S that is Sidon (all pair sums i<=j unique)."""
    S = sorted(S)
    for k in range(len(S), 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def count_additive_quadruples(a, b, c, d, S_set, S_sorted):
    """Count number of ways a+b = c+d with {a,b} != {c,d} in S."""
    # For a fixed pair sum s = a+b, count how many other pairs have same sum
    # This is O(n^2) per pair, so total O(n^4) which is fine for n=12
    pass

def collision_score(S):
    """For each element, count how many additive quadruples it participates in.
    Higher score = more Sidon violations = harder for evaluator to find large Sidon subset."""
    S = sorted(S)
    n = len(S)
    # For each pair (i,j) with i<=j, compute sum, then for each other pair (k,l) with k<=l, k!=i or l!=j
    # Count collisions per element
    elem_conflict = [0] * n
    for i in range(n):
        for j in range(i, n):
            s1 = S[i] + S[j]
            for k in range(n):
                for l in range(k, n):
                    if k == i and l == j:
                        continue
                    if S[k] + S[l] == s1:
                        elem_conflict[i] += 1
                        elem_conflict[j] += 1
                        elem_conflict[k] += 1
                        elem_conflict[l] += 1
    return elem_conflict

def min_conflict_degree(S):
    """Minimum over elements of their conflict count - we want to maximize this."""
    scores = collision_score(S)
    return min(scores) if scores else 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # The key insight: we want a set where EVERY 7-subset fails to be Sidon.
    # A Sidon set of size k has k(k+1)/2 distinct pair sums.
    # For k=7: 28 distinct sums needed.
    # The pair sums range from 2*min(S) to 2*max(S), giving at most 2*max(S)-2*min(S)+1 possible sums.
    # But that's a loose bound.
    
    # Better approach: try to find sets where the max Sidon subset is as small as possible.
    # Lower bound: can we get it down to 5? 6?
    
    # For [0,11]: pair sums range 0 to 22, that's 23 values. k=6 needs 21, k=7 needs 28 > 23.
    # So max Sidon for [0,11] is at most 6.
    
    # Can we do better? What if we compress the range more?
    # If we use 12 values in a range of size R, pair sums range over 2R+1 values.
    # For max Sidon size k: need k(k+1)/2 <= 2R+1
    # For k=6: need 21 <= 2R+1, so R >= 10
    # For k=5: need 15 <= 2R+1, so R >= 7
    # But we need 12 distinct integers, so the range must be at least 11.
    # So R >= 11, meaning 2R+1 >= 23, so k=6 (needing 21) is possible.
    # k=7 needs 28 > 23, so max Sidon is at most 6 for any 12-element set in [0,128] 
    # where the range is at most 13.
    
    # Wait, that's not right. The range of pair sums depends on the actual min and max.
    # If S has min=a and max=b, pair sums range from 2a to 2b, that's 2(b-a)+1 values.
    # For 12 distinct integers, b-a >= 11, so at least 23 possible pair sum values.
    # k=7 needs 28 distinct sums, which is > 23, so k=7 is impossible.
    # Therefore, for ANY 12-element set in [0,128], max Sidon subset is at most 6.
    
    # Wait, that's only true if b-a >= 11. Since we need 12 distinct integers,
    # the minimum range is 11 (e.g., 0,1,...,11). So 2(b-a)+1 >= 23.
    # And 28 > 23, so no 7-element Sidon subset is possible.
    # Therefore the answer is always at most 6!
    
    # But wait - is it possible that max Sidon is actually 6 for some sets and less for others?
    # Can we make it so that NO 6-element subset is Sidon?
    # A 6-element Sidon set needs 21 distinct pair sums.
    # We have 2(b-a)+1 >= 23 possible sums.
    # So it's theoretically possible to have 6-element Sidon subsets.
    
    # The question is: can we design a 12-element set where no 6-element subset is Sidon?
    # That would give us max Sidon = 5.
    
    # Strategy: use a "collision graph" approach.
    # Two pairs (a,b) and (c,d) with a+b = c+d create a conflict.
    # For a 6-element subset to NOT be Sidon, it must contain at least one such conflict.
    
    # Let's try: make the set have many additive relations.
    # E.g., an arithmetic progression has many: a+i, a+i+d, etc.
    # For AP with 12 terms: a, a+d, ..., a+11d
    # Pair sums: 2a+id, 2a+(i+1)d, ... up to 2a+21d
    # Number of distinct sums: 22 (from 2a to 2a+21d)
    # Wait, for an AP, pair sums are 2a+(i+j)d where 0<=i<=j<=11
    # i+j ranges from 0 to 22, so 23 distinct values.
    # Can we find a 6-element Sidon subset of an AP?
    # A 6-element subset of an AP is itself in "Sidon position" if the index set is Sidon.
    # The index set {0,...,11} - can we find 6 indices that form a Sidon set?
    # A Sidon set of size 6 in {0,...,11} with index sums 0 to 22 (23 values).
    # We need 21 distinct sums from 23 available, so it's tight but possible.
    # E.g., {0,1,4,9,15,22} - but 22 > 11. Let me think in terms of indices 0-11.
    # {0,1,4,10,11,?} - let me just check computationally.
    
    # Actually, let me just try various structured sets and measure.
    
    # Key structured families to try:
    candidates = []
    
    # Arithmetic progressions
    for d in range(1, 11):
        for start in range(0, 128 - 11*d + 1):
            ap = [start + i*d for i in range(12)]
            if all(0 <= x <= 128 for x in ap):
                candidates.append(ap)
    
    # Quadratic sequences
    for a in range(1, 5):
        for b in range(0, 10):
            for c in range(0, 10):
                quad = [a*i*i + b*i + c for i in range(12)]
                if len(set(quad)) == 12 and all(0 <= x <= 128 for x in quad):
                    candidates.append(quad)
    
    # Geometric-ish
    for r in range(1, 10):
        geo = [min(128, r**i if r == 1 else (r**i - 1)//(r-1)) for i in range(12)]
        if len(set(geo)) == 12 and all(0 <= x <= 128 for x in geo):
            candidates.append(geo)
    
    # Random structured: take a base and add noise
    random.seed(42)
    for _ in range(50):
        base = sorted(random.sample(range(0, 129), 12))
        candidates.append(base)
    
    # Dense low-range sets
    for _ in range(200):
        # 12 random from a small window
        window_start = random.randint(0, 40)
        window_end = window_start + random.randint(11, 25)
        if window_end <= 128:
            cand = sorted(random.sample(range(window_start, window_end + 1), 12))
            candidates.append(cand)
    
    # Very dense: 12 from 12 or 13 consecutive
    for s in range(0, 117):
        candidates.append(list(range(s, s + 12)))
    for s in range(0, 116):
        # 12 from 13 consecutive (skip one)
        full = list(range(s, s + 13))
        for skip in range(13):
            candidates.append([full[j] for j in range(13) if j != skip])
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(sorted(c))
    
    # First pass: use a fast upper bound to filter
    # For a set with min=a, max=b, pair sums in [2a, 2b], range = 2(b-a)+1
    # If 2(b-a)+1 < 21 (needed for k=6), then max Sidon < 6
    # But 2(b-a)+1 >= 23 always for 12 distinct integers, so this bound doesn't help.
    
    # Better fast filter: count how many 6-subsets have a pair-sum collision
    # If ALL 6-subsets have a collision, max Sidon <= 5
    
    def all_6_subsets_conflict(S):
        """Check if every 6-element subset of S has at least one pair-sum collision."""
        S = sorted(S)
        n = len(S)
        for comb in itertools.combinations(range(n), 6):
            subset = [S[i] for i in comb]
            if is_sidon(subset):
                return False
        return True
    
    # This is C(12,6) = 924 subsets to check, each O(21) = fast
    # But we can be smarter: precompute pair sums
    
    def has_sidon_6_subset(S):
        """Check if there exists a 6-element Sidon subset of S."""
        S = sorted(S)
        n = len(S)
        for comb in itertools.combinations(range(n), 6):
            subset = [S[i] for i in comb]
            if is_sidon(subset):
                return True
        return False
    
    # Evaluate candidates, starting with those most likely to have small max Sidon
    # Sort by range (smaller range = more collisions likely)
    unique_candidates.sort(key=lambda c: c[-1] - c[0])
    
    current_best = None
    current_best_val = 7  # Start high
    
    # Quick filter: check if max Sidon could be < 6
    # This requires checking all 6-subsets for Sidon property
    
    evaluated = 0
    for c in unique_candidates:
        if time.time() > deadline:
            break
        
        # Fast check: does this set have a 6-element Sidon subset?
        if has_sidon_6_subset(c):
            # Max Sidon >= 6, skip (we want < 6)
            evaluated += 1
            continue
        else:
            # Max Sidon <= 5! Check if it's exactly 5 or less
            val = max_sidon_size(c)
            if val < current_best_val:
                current_best_val = val
                current_best = c
                if current_best_val <= 4:
                    break  # Can't do much better than 4
            evaluated += 1
    
    if current_best is None:
        # All candidates had 6-element Sidon subsets
        # Fall back to finding the one with minimum max Sidon size (which is >= 6)
        # Try more candidates with different structure
        current_best_val = 12
        for c in unique_candidates[:500]:
            if time.time() > deadline:
                break
            val = max_sidon_size(c)
            if val < current_best_val:
                current_best_val = val
                current_best = c
            if current_best_val <= 6:
                break
    
    if current_best is None:
        current_best = list(range(12))
        current_best_val = max_sidon_size(current_best)
    
    # Now try to improve: local search to reduce max Sidon size
    if time.time() < deadline - 10 and current_best_val > 4:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_cand = current_best
            best_val = current_best_val
            
            # Try single-element mutations
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                old_val = best_cand[i]
                for new_val in range(0, 129):
                    if new_val == old_val or new_val in best_cand:
                        continue
                    cand = list(best_cand)
                    cand[i] = new_val
                    cand = sorted(cand)
                    if len(set(cand)) != 12:
                        continue
                    # Check if this reduces max Sidon
                    if has_sidon_6_subset(cand):
                        continue
                    val = max_sidon_size(cand)
                    if val < best_val:
                        best_val = val
                        best_cand = cand
                        improved = True
                        break
            
            current_best = best_cand
            current_best_val = best_val
    
    return current_best
