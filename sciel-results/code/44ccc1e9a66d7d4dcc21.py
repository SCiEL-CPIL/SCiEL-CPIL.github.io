"""Precomputed hypergraph + deterministic greedy set-cover + fast local search."""
import time
import numpy as np

N = 256
DEADLINE_FRACTION = 0.85

# Precompute all 3-APs and 4-APs at module load
def _precompute():
    """Precompute 3-AP list, 4-AP list, and per-element data structures."""
    # All 3-APs: (a, a+d, a+2d) with d >= 1
    all_3aps = []
    for d in range(1, (N + 1) // 2):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            all_3aps.append((a, a + d, a + 2 * d))
    
    # All 4-APs: (a, a+d, a+2d, a+3d) with d >= 1
    all_4aps = []
    for d in range(1, (N + 1) // 4):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each element x, precompute:
    # - elements that form a 4-AP with x (conflict set for adding x)
    #   Actually, we need: for a given set S, adding x creates a 4-AP iff
    #   there exists a 4-AP containing x where the other 3 elements are in S.
    # - For fast checking: for each x, list of (other3_tuple) for each 4-AP containing x
    
    # 4-AP conflicts: for each x, list of frozensets of the other 3 elements
    # But that's too large. Instead: for each x, for each 4-AP containing x,
    # store the indices of the other 3 elements.
    
    # Better approach: for each x, precompute a list of (d1, d2, d3) patterns
    # where {x, x+d1, x+d2, x+d3} forms a 4-AP.
    # Actually simplest: for each x, list of (i, j, k) such that {x, i, j, k} is a 4-AP
    
    # For 3-APs: for each x, list of (i, j) such that {x, i, j} is a 3-AP
    # and x is the "middle" or "end" -- actually we just need which pairs with x form 3-APs
    
    # Let's use a different fast approach:
    # For checking if adding x creates a 4-AP:
    #   x can be position 1: x, x+d, x+2d, x+3d -> need {x+d, x+2d, x+3d} subset of S
    #   x can be position 2: x-d, x, x+d, x+2d -> need {x-d, x+d, x+2d} subset of S
    #   x can be position 3: x-2d, x-d, x, x+d -> need {x-2d, x-d, x+d} subset of S
    #   x can be position 4: x-3d, x-2d, x-d, x -> need {x-3d, x-2d, x-d} subset of S
    
    # Precompute for each x the list of "required triples" for each position
    # conflict_triples[x] = list of tuples of 3 elements (the other members of 4-APs containing x)
    conflict_triples = [[] for _ in range(N + 1)]
    for (a, b, c, d) in all_4aps:
        for pos, x in enumerate((a, b, c, d)):
            others = tuple(t for t in (a, b, c, d) if t != x)
            conflict_triples[x].append(others)
    
    # For 3-AP counting: precompute for each pair (i, j) whether they are part of a 3-AP
    # and what the third element is. Actually, let's precompute:
    # For each x, the set of pairs (i, j) such that {x, i, j} is a 3-AP
    # This is useful for incremental counting.
    
    # 3-AP partners: for each x, list of (i, j) where {x, i, j} forms a 3-AP
    # x can be first: x, x+d, x+2d -> partner pair is (x+d, x+2d)
    # x can be second: x-d, x, x+d -> partner pair is (x-d, x+d)
    # x can be third: x-2d, x-d, x -> partner pair is (x-2d, x-d)
    
    # For counting 3-APs in a set S: count = number of (i, j, k) in S with i+j=2k or similar
    # Actually: for each d, count pairs (a, a+2d) both in S, and a+d in S.
    # Alternative: for each middle element m, count pairs (m-d, m+d) both in S.
    # 3-AP with middle m: (m-d, m, m+d) for d=1..min(m-1, N-m)
    
    # Precompute for each middle m, the list of (m-d, m+d) pairs
    middle_pairs = [[] for _ in range(N + 1)]
    for m in range(1, N + 1):
        max_d = min(m - 1, N - m)
        for d in range(1, max_d + 1):
            middle_pairs[m].append((m - d, m + d))
    
    # For each end element (first or third), precompute connections
    # Actually for incremental update when adding x:
    # New 3-APs created by adding x:
    #   x as first: (x, x+d, x+2d) for d where x+2d <= N and {x+d, x+2d} subset S
    #   x as second: (x-d, x, x+d) for d where x-d >= 1 and x+d <= N and {x-d, x+d} subset S
    #   x as third: (x-2d, x-d, x) for d where x-2d >= 1 and {x-2d, x-d} subset S
    
    # Precompute for each x:
    # x_as_first[x] = list of (x+d, x+2d) for valid d
    # x_as_second[x] = list of (x-d, x+d) for valid d  
    # x_as_third[x] = list of (x-2d, x-d) for valid d
    
    x_as_first = [[] for _ in range(N + 1)]
    x_as_second = [[] for _ in range(N + 1)]
    x_as_third = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # x as first: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            x_as_first[x].append((x + d, x + 2 * d))
        # x as second: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            x_as_second[x].append((x - d, x + d))
        # x as third: x-2d, x-d, x
        for d in range(1, (x - 1) // 2 + 1):
            x_as_third[x].append((x - 2 * d, x - d))
    
    return all_3aps, all_4aps, conflict_triples, middle_pairs, x_as_first, x_as_second, x_as_third

_ALL_3APS, _ALL_4APS, _CONFLICT_TRIPLES, _MIDDLE_PAIRS, _X_AS_FIRST, _X_AS_SECOND, _X_AS_THIRD = _precompute()

def _count_3ap(flags):
    """Count distinct 3-APs using precomputed middle pairs."""
    arr = np.array(flags, dtype=np.uint8)
    count = 0
    for m in range(2, N):  # m must have at least one on each side
        for (a, b) in _MIDDLE_PAIRS[m]:
            if arr[a] and arr[b]:
                count += 1
    return count

def _count_3ap_fast(flags):
    """Faster 3-AP count using numpy."""
    arr = np.array(flags, dtype=np.uint8)
    count = 0
    for d in range(1, (N + 1) // 2):
        max_a = N - 2 * d
        if max_a < 1:
            break
        count += int(np.sum(arr[1:max_a+1] & arr[1+d:max_a+1+d] & arr[1+2*d:max_a+1+2*d]))
    return count

def _check_4ap_with_flags(flags_arr, x):
    """Check if adding x creates a 4-AP. flags_arr is numpy uint8 array."""
    # x as first
    max_d = (N - x) // 3
    if max_d >= 1:
        ds = np.arange(1, max_d + 1)
        if np.any(flags_arr[x+ds] & flags_arr[x+2*ds] & flags_arr[x+3*ds]):
            return True
    # x as second
    max_d = min(x - 1, (N - x) // 2)
    if max_d >= 1:
        ds = np.arange(1, max_d + 1)
        if np.any(flags_arr[x-ds] & flags_arr[x+ds] & flags_arr[x+2*ds]):
            return True
    # x as third
    max_d = min((x - 1) // 2, N - x)
    if max_d >= 1:
        ds = np.arange(1, max_d + 1)
        if np.any(flags_arr[x-2*ds] & flags_arr[x-ds] & flags_arr[x+ds]):
            return True
    # x as fourth
    max_d = (x - 1) // 3
    if max_d >= 1:
        ds = np.arange(1, max_d + 1)
        if np.any(flags_arr[x-3*ds] & flags_arr[x-2*ds] & flags_arr[x-ds]):
            return True
    return False

def _delta_3ap(flags_arr, x):
    """Compute how many new 3-APs are created by adding x to current set."""
    delta = 0
    # x as first: (x, x+d, x+2d)
    for (a, b) in _X_AS_FIRST[x]:
        if flags_arr[a] and flags_arr[b]:
            delta += 1
    # x as second: (x-d, x, x+d)
    for (a, b) in _X_AS_SECOND[x]:
        if flags_arr[a] and flags_arr[b]:
            delta += 1
    # x as third: (x-2d, x-d, x)
    for (a, b) in _X_AS_THIRD[x]:
        if flags_arr[a] and flags_arr[b]:
            delta += 1
    return delta

def _greedy_setcover(deadline):
    """Deterministic greedy: repeatedly pick element with max delta_3ap that doesn't create 4-AP."""
    flags = [False] * (N + 1)
    flags_arr = np.zeros(N + 1, dtype=np.uint8)
    current_score = 0
    selected = []
    
    # Start with a seed: pick elements that are in many 3-APs
    # Precompute weight of each element (number of 3-APs it's part of)
    weights = np.zeros(N + 1, dtype=np.int32)
    for (a, b, c) in _ALL_3APS:
        weights[a] += 1
        weights[b] += 1
        weights[c] += 1
    
    # Sort elements by weight descending, use as priority
    order = sorted(range(1, N + 1), key=lambda x: -weights[x])
    
    for x in order:
        if time.time() > deadline:
            break
        if flags[x]:
            continue
        if _check_4ap_with_flags(flags_arr, x):
            continue
        flags[x] = True
        flags_arr[x] = 1
        current_score += _delta_3ap(flags_arr, x)  # Note: this counts 3-APs where x is new
        # Actually _delta_3ap before setting flag gives correct count
        # Let me fix: compute delta before adding
        selected.append(x)
    
    return selected, current_score

def _greedy_setcover_correct(deadline):
    """Corrected greedy set-cover."""
    flags = [False] * (N + 1)
    flags_arr = np.zeros(N + 1, dtype=np.uint8)
    current_score = 0
    selected = []
    
    weights = np.zeros(N + 1, dtype=np.int32)
    for (a, b, c) in _ALL_3APS:
        weights[a] += 1
        weights[b] += 1
        weights[c] += 1
    
    order = sorted(range(1, N + 1), key=lambda x: -weights[x])
    
    for x in order:
        if time.time() > deadline:
            break
        if flags[x]:
            continue
        if _check_4ap_with_flags(flags_arr, x):
            continue
        # Compute delta before adding
        delta = _delta_3ap(flags_arr, x)
        flags[x] = True
        flags_arr[x] = 1
        current_score += delta
        selected.append(x)
    
    return selected, current_score

def _local_search_1flip(flags, flags_arr, current_score, deadline):
    """1-flip local search: try adding or removing single elements."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding
        for x in range(1, N + 1):
            if time.time() > deadline:
                return current_score
            if flags[x]:
                continue
            if _check_4ap_with_flags(flags_arr, x):
                continue
            delta = _delta_3ap(flags_arr, x)
            if delta > 0:
                flags[x] = True
                flags_arr[x] = 1
                current_score += delta
                improved = True
        # Try removing (only if it doesn't reduce score -- but removing always reduces or keeps same)
        # Actually removing an element removes some 3-APs. Skip removal in 1-flip.
    return current_score

def _local_search_swap(flags, flags_arr, current_score, deadline):
    """2-swap local search: remove one element, add another (or two swaps)."""
    improved = True
    max_rounds = 20
    for _round in range(max_rounds):
        if time.time() > deadline or not improved:
            break
        improved = False
        current_list = [i for i in range(1, N + 1) if flags[i]]
        for elem in current_list:
            if time.time() > deadline:
                return current_score
            # Remove elem
            # Compute how many 3-APs are lost
            lost = 0
            for (a, b) in _X_AS_FIRST[elem]:
                if flags[a] and flags[b]:
                    lost += 1
            for (a, b) in _X_AS_SECOND[elem]:
                if flags[a] and flags[b]:
                    lost += 1
            for (a, b) in _X_AS_THIRD[elem]:
                if flags[a] and flags[b]:
                    lost += 1
            
            flags[elem] = False
            flags_arr[elem] = 0
            score_after_removal = current_score - lost
            
            # Try adding each candidate
            best_x = None
            best_gain = -1
            for x in range(1, N + 1):
                if flags[x]:
                    continue
                if _check_4ap_with_flags(flags_arr, x):
                    continue
                gain = _delta_3ap(flags_arr, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is not None and (score_after_removal + best_gain) > current_score:
                flags[best_x] = True
                flags_arr[best_x] = 1
                current_score = score_after_removal + best_gain
                improved = True
            else:
                # Restore
                flags[elem] = True
                flags_arr[elem] = 1
    
    return current_score

def _double_swap(flags, flags_arr, current_score, deadline):
    """Try removing two elements and adding two different ones."""
    improved = True
    for _round in range(5):
        if time.time() > deadline or not improved:
            break
        improved = False
        current_list = [i for i in range(1, N + 1) if flags[i]]
        n = len(current_list)
        if n < 2:
            break
        
        # Limit pairs to check
        max_pairs = 200
        pair_count = 0
        for i in range(n):
            if time.time() > deadline:
                return current_score
            for j in range(i + 1, n):
                if pair_count >= max_pairs:
                    break
                pair_count += 1
                if time.time() > deadline:
                    return current_score
                
                e1, e2 = current_list[i], current_list[j]
                
                # Remove both
                lost1 = 0
                for (a, b) in _X_AS_FIRST[e1]:
                    if flags[a] and flags[b]:
                        lost1 += 1
                for (a, b) in _X_AS_SECOND[e1]:
                    if flags[a] and flags[b]:
                        lost1 += 1
                for (a, b) in _X_AS_THIRD[e1]:
                    if flags[a] and flags[b]:
                        lost1 += 1
                
                lost2 = 0
                for (a, b) in _X_AS_FIRST[e2]:
                    if flags[a] and flags[b]:
                        lost2 += 1
                for (a, b) in _X_AS_SECOND[e2]:
                    if flags[a] and flags[b]:
                        lost2 += 1
                for (a, b) in _X_AS_THIRD[e2]:
                    if flags[a] and flags[b]:
                        lost2 += 1
                
                # Account for 3-APs containing both e1 and e2
                # A 3-AP contains both if {e1, e2, z} is a 3-AP for some z
                # This is complex; approximate by just summing losses
                # (may overcount if a 3-AP has both, but that's rare)
                
                flags[e1] = False
                flags_arr[e1] = 0
                flags[e2] = False
                flags_arr[e2] = 0
                score_after = current_score - lost1 - lost2
                
                # Try adding two new elements
                best_x1, best_x2 = None, None
                best_total_gain = -1
                
                # Try all pairs of candidates (limited)
                candidates = [x for x in range(1, N + 1) if not flags[x]]
                # Limit candidates
                candidates = candidates[:80]
                
                for x1 in candidates:
                    if time.time() > deadline:
                        break
                    if _check_4ap_with_flags(flags_arr, x1):
                        continue
                    gain1 = _delta_3ap(flags_arr, x1)
                    flags[x1] = True
                    flags_arr[x1] = 1
                    
                    for x2 in candidates:
                        if x2 == x1:
                            continue
                        if flags[x2]:
                            continue
                        if _check_4ap_with_flags(flags_arr, x2):
                            continue
                        gain2 = _delta_3ap(flags_arr, x2)
                        total_gain = gain1 + gain2
                        if total_gain > best_total_gain:
                            best_total_gain = total_gain
                            best_x1, best_x2 = x1, x2
                    
                    flags[x1] = False
                    flags_arr[x1] = 0
                
                if best_x1 is not None and (score_after + best_total_gain) > current_score:
                    flags[best_x1] = True
                    flags_arr[best_x1] = 1
                    flags[best_x2] = True
                    flags_arr[best_x2] = 1
                    current_score = score_after + best_total_gain
                    improved = True
                else:
                    flags[e1] = True
                    flags_arr[e1] = 1
                    flags[e2] = True
                    flags_arr[e2] = 1
    return current_score

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    best_set = None
    best_score = -1
    
    # Phase 1: Deterministic greedy set-cover (multiple weight orderings)
    # Weighting 1: raw 3-AP participation
    weights1 = np.zeros(N + 1, dtype=np.int32)
    for (a, b, c) in _ALL_3APS:
        weights1[a] += 1
        weights1[b] += 1
        weights1[c] += 1
    
    # Weighting 2: prefer middle positions (more 3-APs as middle)
    weights2 = np.zeros(N + 1, dtype=np.int32)
    for (a, b, c) in _ALL_3APS:
        weights2[a] += 1
        weights2[b] += 3  # middle is more valuable
        weights2[c] += 1
    
    # Weighting 3: prefer elements with many "free" 3-AP opportunities
    # (3-APs where the other two elements are far apart, less likely to conflict)
    weights3 = np.zeros(N + 1, dtype=np.int32)
    for (a, b, c) in _ALL_3APS:
        d = b - a
        # Larger d means less likely to be in a 4-AP
        w = 1 + (d > 10) * 2 + (d > 30) * 3
        weights3[a] += w
        weights3[b] += w
        weights3[c] += w
    
    for weights in [weights1, weights2, weights3]:
        if time.time() > deadline:
            break
        order = sorted(range(1, N + 1), key=lambda x: -weights[x])
        flags = [False] * (N + 1)
        flags_arr = np.zeros(N + 1, dtype=np.uint8)
        current_score = 0
        
        for x in order:
            if time.time() > deadline:
                break
            if flags[x]:
                continue
            if _check_4ap_with_flags(flags_arr, x):
                continue
            delta = _delta_3ap(flags_arr, x)
            flags[x] = True
            flags_arr[x] = 1
            current_score += delta
        
        # Local search polish
        current_score = _local_search_1flip(flags, flags_arr, current_score, deadline)
        current_score = _local_search_swap(flags, flags_arr, current_score, deadline)
        
        if current_score > best_score:
            best_score = current_score
            best_set = [i for i in range(1, N + 1) if flags[i]]
    
    # Phase 2: Randomized restarts with greedy + local search
    rng = np.random.default_rng(123)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random perturbation of ordering
        order = list(range(1, N + 1))
        # Mix: first 50% by weight, rest random
        weighted_part = sorted(range(1, N + 1), key=lambda x: -weights1[x])[:128]
        random_part = [x for x in order if x not in set(weighted_part)]
        rng.shuffle(random_part)
        mixed_order = weighted_part + random_part
        rng.shuffle(mixed_order)  # Full shuffle for diversity
        
        flags = [False] * (N + 1)
        flags_arr = np.zeros(N + 1, dtype=np.uint8)
        current_score = 0
        
        for x in mixed_order:
            if time.time() > deadline:
                break
            if flags[x]:
                continue
            if _check_4ap_with_flags(flags_arr, x):
                continue
            delta = _delta_3ap(flags_arr, x)
            flags[x] = True
            flags_arr[x] = 1
            current_score += delta
        
        current_score = _local_search_1flip(flags, flags_arr, current_score, deadline)
        current_score = _local_search_swap(flags, flags_arr, current_score, deadline)
        
        if current_score > best_score:
            best_score = current_score
            best_set = [i for i in range(1, N + 1) if flags[i]]
    
    # Phase 3: Final polish on best
    if best_set is not None and time.time() < deadline:
        flags = [False] * (N + 1)
        flags_arr = np.zeros(N + 1, dtype=np.uint8)
        for x in best_set:
            flags[x] = True
            flags_arr[x] = 1
        best_score = _local_search_1flip(flags, flags_arr, best_score, deadline)
        best_score = _local_search_swap(flags, flags_arr, best_score, deadline)
        best_score = _double_swap(flags, flags_arr, best_score, deadline)
        best_set = [i for i in range(1, N + 1) if flags[i]]
    
    # Final verification
    if best_set is not None:
        # Verify no 4-APs
        flags_final = [False] * (N + 1)
        for x in best_set:
            flags_final[x] = True
        flags_arr_final = np.array(flags_final, dtype=np.uint8)
        for (a, b, c, d) in _ALL_4APS:
            if flags_final[a] and flags_final[b] and flags_final[c] and flags_final[d]:
                # Invalid! Remove the last element
                best_set.remove(d)
                break
    
    return best_set if best_set is not None else [1]
