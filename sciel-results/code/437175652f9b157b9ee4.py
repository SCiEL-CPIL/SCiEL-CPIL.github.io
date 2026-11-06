import bisect
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for the 90s hard limit

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves
    n1 = n // 2
    n2 = n - n1
    
    indices1 = list(range(n1))
    indices2 = list(range(n1, n))
    
    # Precompute float and Fraction values
    vals1 = [1.0 / denominators[i] for i in indices1]
    vals2 = [1.0 / denominators[i] for i in indices2]
    
    frac1 = [Fraction(1, denominators[i]) for i in indices1]
    frac2 = [Fraction(1, denominators[i]) for i in indices2]
    
    # Generate subset sums for half 1: (float_sum, mask)
    sums1 = [(0.0, 0)]
    for i in range(n1):
        v = vals1[i]
        new_sums = []
        for s, m in sums1:
            new_sums.append((s, m))
            new_sums.append((s + v, m | (1 << i)))
        sums1 = new_sums
        
    if time.time() > deadline - 20:
        return _greedy_fallback()

    # Generate subset sums for half 2: (float_sum, mask)
    sums2 = [(0.0, 0)]
    for i in range(n2):
        v = vals2[i]
        new_sums = []
        for s, m in sums2:
            new_sums.append((s, m))
            new_sums.append((s + v, m | (1 << i)))
        sums2 = new_sums
        
    # Sort half 2 by float sum
    sums2.sort(key=lambda x: x[0])
    
    # Extract sorted floats and masks for binary search
    s2_floats = [s for s, m in sums2]
    s2_masks = [m for s, m in sums2]
    
    best_deficit_float = float('inf')
    best_exact_deficit = None
    best_indices = None
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate through half 1 sums
    for s1, m1 in sums1:
        if time.time() > deadline - 5:
            break
            
        target = 1.0 - s1
        if target <= 0:
            continue
            
        # Find position in s2_floats
        pos = bisect.bisect_left(s2_floats, target)
        
        # Check neighbors around pos
        candidates = set()
        if pos > 0:
            candidates.add(pos - 1)
        if pos < len(s2_floats):
            candidates.add(pos)
        if pos + 1 < len(s2_floats):
            candidates.add(pos + 1)
            
        for j in candidates:
            s2 = s2_floats[j]
            m2 = s2_masks[j]
            
            s_float = s1 + s2
            if s_float >= 1.0:
                continue
                
            def_float = 1.0 - s_float
            if def_float >= best_deficit_float:
                continue
                
            # Compute exact sum
            sum_frac = Fraction(0)
            
            # Sum bits in m1
            mm = m1
            i = 0
            while mm:
                if mm & 1:
                    sum_frac += frac1[i]
                mm >>= 1
                i += 1
                
            # Sum bits in m2
            mm = m2
            i = 0
            while mm:
                if mm & 1:
                    sum_frac += frac2[i]
                mm >>= 1
                i += 1
                
            if sum_frac < 1:
                def_exact = 1 - sum_frac
                if best_exact_deficit is None or def_exact < best_exact_deficit:
                    best_exact_deficit = def_exact
                    best_deficit_float = float(def_exact)
                    best_mask1 = m1
                    best_mask2 = m2

    if best_indices is None:
        if best_mask1 is not None or best_mask2 is not None:
            # Construct indices from masks
            inds = []
            mm = best_mask1
            i = 0
            while mm:
                if mm & 1:
                    inds.append(indices1[i])
                mm >>= 1
                i += 1
            mm = best_mask2
            i = 0
            while mm:
                if mm & 1:
                    inds.append(indices2[i])
                mm >>= 1
                i += 1
            best_indices = inds
        else:
            return _greedy_fallback()
        
    # Local search polish: try swapping elements to improve the deficit
    if best_indices is not None and time.time() < deadline - 3:
        best_indices = _local_search_polish(best_indices, best_exact_deficit, denominators, deadline)
        
    return best_indices

def _local_search_polish(indices, current_deficit, denominators, deadline):
    """Try to improve the solution by swapping or adding/removing single elements."""
    if not indices:
        return indices
    
    # Convert to set for O(1) lookup
    index_set = set(indices)
    current_sum = Fraction(0)
    for i in indices:
        current_sum += Fraction(1, denominators[i])
    
    best_deficit = current_deficit
    best_indices = list(indices)
    improved = True
    
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try removing each element
        for i in list(best_indices):
            new_sum = current_sum - Fraction(1, denominators[i])
            if new_sum < 1:
                new_deficit = 1 - new_sum
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    best_indices.remove(i)
                    index_set.discard(i)
                    current_sum = new_sum
                    improved = True
                    break
        
        # Try adding each element not in the set
        if not improved and time.time() < deadline - 2:
            for i in range(len(denominators)):
                if i not in index_set:
                    new_sum = current_sum + Fraction(1, denominators[i])
                    if new_sum < 1:
                        new_deficit = 1 - new_sum
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            best_indices.append(i)
                            index_set.add(i)
                            current_sum = new_sum
                            improved = True
                            break
        
        # Try swapping: remove one, add another
        if not improved and time.time() < deadline - 2:
            for i in list(best_indices):
                for j in range(len(denominators)):
                    if j not in index_set:
                        new_sum = current_sum - Fraction(1, denominators[i]) + Fraction(1, denominators[j])
                        if new_sum < 1:
                            new_deficit = 1 - new_sum
                            if new_deficit < best_deficit:
                                best_deficit = new_deficit
                                best_indices.remove(i)
                                index_set.discard(i)
                                best_indices.append(j)
                                index_set.add(j)
                                current_sum = new_sum
                                improved = True
                                break
                if improved:
                    break
    
    return best_indices

def _greedy_fallback():
    selected, total = [], Fraction()
    for i, d in enumerate(range(2, 49)):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    if not selected:
        return [0]
    return selected
