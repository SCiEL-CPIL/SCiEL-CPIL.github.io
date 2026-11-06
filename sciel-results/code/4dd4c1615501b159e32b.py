"""Find a dissociated reciprocal set from [1,22] maximizing cardinality."""
import time
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    n = 22
    # Calculate LCM of 1..22 to convert to integer sums
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Map each x to its integer reciprocal value
    vals = {x: L // x for x in range(1, n + 1)}
    
    # Sort elements by value descending (smaller x -> larger reciprocal)
    elements = sorted(range(1, n + 1), key=lambda x: -vals[x])
    sorted_vals = [vals[x] for x in elements]
    
    best_result = [1]  # fallback
    best_size = 1
    
    # MITM Strategy
    mid = n // 2
    half1_vals = sorted_vals[:mid]
    half2_vals = sorted_vals[mid:]
    
    def generate_dissociated_subsets(v_list):
        """Generate all dissociated subsets of v_list.
        Returns list of (size, sums_set, indices_list)."""
        results = []
        
        def dfs(start, current_sums, current_indices):
            if time.time() > deadline:
                return
            results.append((len(current_indices), current_sums.copy(), current_indices.copy()))
            
            for i in range(start, len(v_list)):
                v = v_list[i]
                shifted = {s + v for s in current_sums}
                if not (shifted & current_sums):
                    new_sums = current_sums | shifted
                    dfs(i + 1, new_sums, current_indices + [i])
        
        dfs(0, {0}, [])
        return results

    # Generate subsets for both halves
    subsets1 = generate_dissociated_subsets(half1_vals)
    subsets2 = generate_dissociated_subsets(half2_vals)
    
    # Precompute difference sets for all subsets
    # diffs[s] = {a - b for a in sums, b in sums}
    diffs1 = [None] * len(subsets1)
    diffs2 = [None] * len(subsets2)
    
    for i, (size, sums, inds) in enumerate(subsets1):
        if time.time() > deadline:
            break
        diffs1[i] = {a - b for a in sums for b in sums}
        
    for i, (size, sums, inds) in enumerate(subsets2):
        if time.time() > deadline:
            break
        diffs2[i] = {a - b for a in sums for b in sums}
        
    if time.time() > deadline:
        # Fallback to greedy if precomputation takes too long
        return _greedy_fallback(elements, sorted_vals)
        
    # Sort subsets by size descending to find max quickly
    # We create a list of (size, index) for each half
    indices1 = sorted(range(len(subsets1)), key=lambda i: subsets1[i][0], reverse=True)
    indices2 = sorted(range(len(subsets2)), key=lambda i: subsets2[i][0], reverse=True)
    
    # Iterate through pairs
    # We want to maximize k1 + k2
    # We can break early if the max possible sum is <= best_size
    
    for i1 in indices1:
        size1 = subsets1[i1][0]
        if size1 + subsets2[indices2[0]][0] <= best_size:
            break # Even the largest in half2 can't beat best_size with this size1
        
        d1 = diffs1[i1]
        for i2 in indices2:
            size2 = subsets2[i2][0]
            if size1 + size2 <= best_size:
                break # Since indices2 is sorted descending, no further i2 can help
            
            d2 = diffs2[i2]
            
            # Check intersection
            # d1 & d2 should be {0}
            # Optimization: iterate over smaller set
            if len(d1) < len(d2):
                inter = d1 & d2
            else:
                inter = d2 & d1
                
            if inter == {0}:
                # Found a valid set
                if size1 + size2 > best_size:
                    best_size = size1 + size2
                    # Reconstruct elements
                    inds1 = subsets1[i1][2]
                    inds2 = subsets2[i2][2]
                    
                    final_elements = []
                    for idx in inds1:
                        final_elements.append(elements[idx])
                    for idx in inds2:
                        final_elements.append(elements[mid + idx])
                        
                    best_result = final_elements
                    
    return best_result

def _greedy_fallback(elements, sorted_vals):
    """Fallback greedy construction if MITM is too slow."""
    n = len(elements)
    result = []
    result_sums = {0}
    for i in range(n):
        v = sorted_vals[i]
        shifted = {s + v for s in result_sums}
        if not (shifted & result_sums):
            result.append(elements[i])
            result_sums |= shifted
    return result
# EVOLVE-BLOCK-END
