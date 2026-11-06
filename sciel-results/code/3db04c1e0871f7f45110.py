"""Find a dissociated reciprocal set from [1,22] maximizing cardinality."""
import time
from math import lcm

# EVOLVE-BLOCK-START

_LCM = 232792560  # LCM(1..22)
_VALS = {x: _LCM // x for x in range(1, 23)}

def _is_dissociated(elements):
    """Verify all 2^k subset sums of reciprocals are distinct."""
    if not elements:
        return True
    recips = [_LCM // e for e in elements]
    sums = {0}
    for r in recips:
        new_sums = {s + r for s in sums}
        if not sums.isdisjoint(new_sums):
            return False
        sums |= new_sums
    return True

def _spectral_gap_greedy(candidates, current_sums):
    """
    Greedily select elements maximizing minimum distance (spectral gap)
    to existing subset sums. Returns (selected_elements, final_sums).
    """
    selected = []
    sums = set(current_sums)
    remaining = list(candidates)
    
    while remaining:
        # Score each candidate by its minimum distance to current sums
        best_elem = None
        best_gap = -1
        for x in remaining:
            v = _VALS[x]
            shifted = {s + v for s in sums}
            # Check for collision
            if shifted & sums:
                continue
            # Compute minimum distance from shifted sums to original sums
            # (spectral gap: how far the new sums are from existing ones)
            min_dist = float('inf')
            for sv in shifted:
                for os in sums:
                    d = abs(sv - os)
                    if d < min_dist:
                        min_dist = d
                if min_dist == 0:
                    break
            if min_dist > best_gap:
                best_gap = min_dist
                best_elem = x
        
        if best_elem is None:
            break
        
        remaining.remove(best_elem)
        v = _VALS[best_elem]
        shifted = {s + v for s in sums}
        sums |= shifted
        selected.append(best_elem)
    
    return selected, sums

def _mitm_find_max(v_list, deadline):
    """
    MITM: find maximum dissociated subset from v_list.
    Returns (max_size, list_of_indices).
    """
    n = len(v_list)
    if n == 0:
        return 0, []
    
    mid = n // 2
    h1 = v_list[:mid]
    h2 = v_list[mid:]
    
    def gen_dissociated(vals):
        """Generate all dissociated subsets as (size, sums_set, indices)."""
        results = []
        
        def dfs(start, sums, indices):
            if time.time() > deadline:
                return
            results.append((len(indices), sums.copy(), indices.copy()))
            for i in range(start, len(vals)):
                v = vals[i]
                shifted = {s + v for s in sums}
                if not (shifted & sums):
                    dfs(i + 1, sums | shifted, indices + [i])
        
        dfs(0, {0}, [])
        return results
    
    subsets1 = gen_dissociated(h1)
    if time.time() > deadline:
        return 0, []
    subsets2 = gen_dissociated(h2)
    if time.time() > deadline:
        return 0, []
    
    # Precompute difference sets
    diffs1 = []
    for size, sums, inds in subsets1:
        if time.time() > deadline:
            break
        diffs1.append({a - b for a in sums for b in sums})
    
    diffs2 = []
    for size, sums, inds in subsets2:
        if time.time() > deadline:
            break
        diffs2.append({a - b for a in sums for b in sums})
    
    if len(diffs1) != len(subsets1) or len(diffs2) != len(subsets2):
        return 0, []
    
    # Sort by size descending
    order1 = sorted(range(len(subsets1)), key=lambda i: subsets1[i][0], reverse=True)
    order2 = sorted(range(len(subsets2)), key=lambda i: subsets2[i][0], reverse=True)
    
    best_size = 0
    best_pair = (0, 0)
    
    for i1 in order1:
        s1 = subsets1[i1][0]
        if s1 + subsets2[order2[0]][0] <= best_size:
            break
        d1 = diffs1[i1]
        for i2 in order2:
            s2 = subsets2[i2][0]
            if s1 + s2 <= best_size:
                break
            d2 = diffs2[i2]
            # Intersection should be exactly {0}
            if len(d1) < len(d2):
                inter = d1 & d2
            else:
                inter = d2 & d1
            if inter == {0}:
                if s1 + s2 > best_size:
                    best_size = s1 + s2
                    best_pair = (i1, i2)
    
    if best_size == 0:
        return 0, []
    
    i1, i2 = best_pair
    inds = subsets1[i1][2] + [mid + i for i in subsets2[i2][2]]
    return best_size, inds

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    universe = list(range(1, 23))
    
    # Phase 1: Spectral-gap greedy to get a good initial set
    greedy_set, greedy_sums = _spectral_gap_greedy(universe, {0})
    
    best_result = list(greedy_set)
    best_size = len(greedy_set)
    
    # Phase 2: Try to improve via MITM on the full set
    # Sort elements by value descending for MITM
    elements_sorted = sorted(universe, key=lambda x: -_VALS[x])
    sorted_vals = [_VALS[x] for x in elements_sorted]
    
    if time.time() < deadline - 10:
        mitm_size, mitm_inds = _mitm_find_max(sorted_vals, deadline - 5)
        if mitm_size > best_size:
            best_result = [elements_sorted[i] for i in mitm_inds]
            best_size = mitm_size
    
    # Phase 3: Local search - try to add elements not in best
    if time.time() < deadline - 5 and best_size < 22:
        current = list(best_result)
        current_sums = {0}
        for x in current:
            v = _VALS[x]
            current_sums |= {s + v for s in current_sums}
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Score candidates by spectral gap
            candidates = [x for x in universe if x not in current]
            best_candidate = None
            best_gap = -1
            for x in candidates:
                v = _VALS[x]
                shifted = {s + v for s in current_sums}
                if shifted & current_sums:
                    continue
                min_dist = float('inf')
                for sv in shifted:
                    for os in current_sums:
                        d = abs(sv - os)
                        if d < min_dist:
                            min_dist = d
                if min_dist > best_gap:
                    best_gap = min_dist
                    best_candidate = x
            
            if best_candidate is not None:
                v = _VALS[best_candidate]
                current_sums |= {s + v for s in current_sums}
                current.append(best_candidate)
                improved = True
                if len(current) > best_size:
                    best_size = len(current)
                    best_result = list(current)
    
    # Phase 4: Try removing one and adding two (swap search)
    if time.time() < deadline - 8 and best_size >= 3:
        current = list(best_result)
        for i in range(len(current)):
            if time.time() > deadline - 5:
                break
            without_i = current[:i] + current[i+1:]
            ws_sums = {0}
            for x in without_i:
                v = _VALS[x]
                ws_sums |= {s + v for s in ws_sums}
            
            for x in range(1, 23):
                if time.time() > deadline - 5:
                    break
                if x in without_i:
                    continue
                v1 = _VALS[x]
                s1 = {s + v1 for s in ws_sums}
                if s1 & ws_sums:
                    continue
                s1_combined = ws_sums | s1
                for y in range(x + 1, 23):
                    if y in without_i or y == x:
                        continue
                    v2 = _VALS[y]
                    s2 = {s + v2 for s in s1_combined}
                    if s2 & s1_combined:
                        continue
                    # Valid: without_i + [x, y]
                    candidate = without_i + [x, y]
                    if len(candidate) > best_size:
                        best_size = len(candidate)
                        best_result = candidate
                    break
    
    # Final verification
    if not best_result:
        best_result = [1]
    if not _is_dissociated(best_result):
        # Fallback to a known valid set
        for size in range(best_size, 0, -1):
            found = False
            # Just use the greedy result
            if _is_dissociated(greedy_set):
                best_result = list(greedy_set)
                found = True
                break
            if not found:
                best_result = [1]
    
    return sorted(best_result)

# EVOLVE-BLOCK-END
