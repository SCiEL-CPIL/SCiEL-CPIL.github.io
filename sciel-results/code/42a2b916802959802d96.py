import time
import math
import bisect
import sys

def construct():
    start = time.time()
    deadline = start + 85  # 85 second budget, 5s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    L = 1
    for d in denominators:
        L = math.lcm(L, d)
        
    # Calculate weights w_i = L / d_i
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    weights_A = weights[:mid]
    weights_B = weights[mid:]
    
    def gen_sums(weights_subset):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights_subset):
            # Iterate over a copy of the current list to allow appending
            current_len = len(sums)
            for k in range(current_len):
                s = sums[k]
                m = masks[k]
                ns = s + w
                nm = m | (1 << i)
                if ns < L:
                    sums.append(ns)
                    masks.append(nm)
        return list(zip(sums, masks))
    
    # Generate sums for both halves
    sums_A = gen_sums(weights_A)
    sums_B = gen_sums(weights_B)
    
    # Sort sums_B by sum value
    sums_B.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sums_B_values = [s for s, m in sums_B]
    
    best_deficit = L  # Initial deficit is L (sum 0)
    best_mask_A = 0
    best_mask_B = 0
    
    # Iterate through sums_A and find best match in sums_B
    for s_A, m_A in sums_A:
        # We want s_B such that s_A + s_B < L
        # i.e., s_B < L - s_A
        target = L - s_A
        # Find the index of the first element >= target
        idx = bisect.bisect_left(sums_B_values, target)
        
        # The valid candidate is at idx - 1
        if idx > 0:
            s_B, m_B = sums_B[idx - 1]
            total_sum = s_A + s_B
            deficit = L - total_sum
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask_A = m_A
                best_mask_B = m_B
                
                # Early exit if deficit is 1 (exact integer 1 in scaled units)
                # But deficit is L - sum. Min positive deficit is 1 unit?
                # No, deficit is integer. Min positive is 1.
                if best_deficit == 1:
                    break
    
    # Reconstruct indices
    selected_indices = []
    
    # Check mask A
    for i in range(len(weights_A)):
        if best_mask_A & (1 << i):
            selected_indices.append(i)
            
    # Check mask B
    for i in range(len(weights_B)):
        if best_mask_B & (1 << i):
            selected_indices.append(mid + i)
            
    # The problem requires nonempty indices.
    # If best_deficit is L (sum 0), it means no valid non-empty subset was found?
    # But we know single elements are valid.
    # So selected_indices should not be empty.
    # However, if for some reason it is empty, return [0] as fallback.
    if not selected_indices:
        return [0]
        
    # Verify time
    elapsed = time.time() - start
    # No need to print, but good for debugging if allowed.
    
    return selected_indices

if __name__ == "__main__":
    # Just for local testing
    res = construct()
    print(res)
