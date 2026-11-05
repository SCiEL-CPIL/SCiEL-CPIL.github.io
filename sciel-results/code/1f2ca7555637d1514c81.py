import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # 5s margin

    n1 = 15
    n2 = 15
    
    # Generate S1
    # We want to generate all sums for i=1..15
    # Using iterative expansion with NumPy
    
    # Start with 0
    sums1 = np.array([0.0], dtype=np.float64)
    ids1 = np.array([0], dtype=np.int64)
    
    # We need to track the ID.
    # If we expand [s0, s1, s2] to [s0-1/i, s0, s0+1/i, s1-1/i, ...]
    # The ID mapping:
    # New ID = Old ID * 3 + digit
    # Where digit 0 -> -1, 1 -> 0, 2 -> 1?
    # Let's verify.
    # If Old ID is 0.
    # New IDs: 0*3+0=0, 0*3+1=1, 0*3+2=2.
    # Coeffs for ID 0: digit 0 is 0 -> -1. Sum = -1/1. Correct.
    # Coeffs for ID 1: digit 0 is 1 -> 0. Sum = 0. Correct.
    # Coeffs for ID 2: digit 0 is 2 -> 1. Sum = 1/1. Correct.
    
    for i in range(1, n1 + 1):
        inv_i = 1.0 / i
        # Expand sums
        # We need to interleave? No, just concatenate.
        # Order doesn't matter for sorting later.
        # But ID mapping must be consistent.
        # If we do:
        # new_sums = concat([sums1 - inv_i, sums1, sums1 + inv_i])
        # new_ids = concat([ids1*3 + 0, ids1*3 + 1, ids1*3 + 2])
        
        sums1 = np.concatenate([
            sums1 - inv_i,
            sums1,
            sums1 + inv_i
        ])
        ids1 = np.concatenate([
            ids1 * 3,
            ids1 * 3 + 1,
            ids1 * 3 + 2
        ])
        
        if time.time() > deadline:
            break

    # Sort S1
    sort_idx1 = np.argsort(sums1)
    sorted_sums1 = sums1[sort_idx1]
    sorted_ids1 = ids1[sort_idx1]
    
    # Generate S2
    sums2 = np.array([0.0], dtype=np.float64)
    ids2 = np.array([0], dtype=np.int64)
    
    for i in range(n1 + 1, n1 + n2 + 1):
        inv_i = 1.0 / i
        sums2 = np.concatenate([
            sums2 - inv_i,
            sums2,
            sums2 + inv_i
        ])
        ids2 = np.concatenate([
            ids2 * 3,
            ids2 * 3 + 1,
            ids2 * 3 + 2
        ])
        
        if time.time() > deadline:
            break

    # Search
    best_val = float('inf')
    best_cand = None
    candidates = [] # Store (val, id1, id2)
    
    # Iterate through S2
    # To speed up, we can vectorize the binary search?
    # np.searchsorted can handle arrays.
    # targets = -sums2
    # pos = np.searchsorted(sorted_sums1, targets)
    
    targets = -sums2
    pos = np.searchsorted(sorted_sums1, targets)
    
    # We need to check pos, pos-1, pos+1
    # Handle boundaries
    
    # Create arrays for checks
    # It's safer to loop in Python if we want to collect top K candidates easily,
    # but 14M iterations in Python is too slow.
    # We must vectorize the candidate collection.
    
    # Let's compute the sums for pos, pos-1, pos+1 for all elements in S2.
    
    # Mask for valid indices
    n1_size = len(sorted_sums1)
    
    # pos is in [0, n1_size]
    # valid_pos = (pos >= 0) & (pos < n1_size)
    
    # Candidate 1: pos
    p1 = np.clip(pos, 0, n1_size - 1)
    s1_c1 = sorted_sums1[p1]
    total_c1 = s1_c1 + sums2
    val_c1 = np.abs(total_c1)
    
    # Candidate 2: pos - 1
    p2 = np.clip(pos - 1, 0, n1_size - 1)
    s1_c2 = sorted_sums1[p2]
    total_c2 = s1_c2 + sums2
    val_c2 = np.abs(total_c2)
    
    # Candidate 3: pos + 1
    p3 = np.clip(pos + 1, 0, n1_size - 1)
    s1_c3 = sorted_sums1[p3]
    total_c3 = s1_c3 + sums2
    val_c3 = np.abs(total_c3)
    
    # Combine all candidates
    # We want the smallest non-zero values.
    # Create a flat array of (val, id1, id2)
    # id1 comes from sorted_ids1[p], id2 comes from ids2
    
    # This creates 3 * 14M = 42M candidates.
    # Sorting 42M tuples is heavy.
    # Instead, we can find the global minimum and top K using numpy.
    
    # Let's just find the minimum of the combined values.
    # But we need to exclude zero.
    # Float zero check: val == 0.0
    
    # Let's collect the top 100 smallest non-zero values.
    # We can use np.argpartition on the combined values.
    
    vals = np.concatenate([val_c1, val_c2, val_c3])
    # We need to keep track of which id1 and id2 correspond to these vals.
    # This is getting complex to vectorize fully with ID tracking.
    
    # Alternative:
    # Since we only need the best few, we can find the minimum val in each of the 3 arrays,
    # but that's not enough. We need the global minimum among all 3 arrays.
    
    # Let's just find the min of the concatenated vals, ignoring zeros for a moment.
    # If min is 0, we need the next smallest.
    
    # Let's filter out exact zeros first.
    # Note: Due to float precision, "exact zero" might be 1e-16.
    # We will treat val < 1e-12 as "potentially zero" and verify with Fraction later.
    
    # To be safe, let's just grab the smallest 1000 values from the combined set.
    # We need to reconstruct the IDs.
    
    # Let's do it in a loop but optimized? No, 14M is too big for Python loop.
    
    # Let's try to get the top 1000 from each of the 3 candidate sets and merge them.
    # Top 1000 from val_c1, Top 1000 from val_c2, Top 1000 from val_c3.
    # Total 3000 candidates.
    # This is safe because the true minimum must be among the closest neighbors.
    
    def get_top_k_ids(vals, ids_s1, ids_s2, k):
        # Filter out large values? No, just partition.
        if len(vals) == 0:
            return [], [], []
        
        # We want indices of smallest k values.
        # np.argpartition returns indices of k smallest values (unsorted).
        if k >= len(vals):
            idx = np.arange(len(vals))
        else:
            idx = np.argpartition(vals, k)[:k]
        
        # Filter out zeros?
        # We'll do it later.
        return idx
    
    idx1 = get_top_k_ids(val_c1, sorted_ids1, ids2, 1000)
    idx2 = get_top_k_ids(val_c2, sorted_ids1, ids2, 1000)
    idx3 = get_top_k_ids(val_c3, sorted_ids1, ids2, 1000)
    
    all_candidates = []
    
    # Process idx1
    for i in idx1:
        v = val_c1[i]
        if v < 1e-12: # Small enough to be suspicious
            id1 = sorted_ids1[p1[i]]
            id2 = ids2[i]
            all_candidates.append((v, id1, id2))
            
    for i in idx2:
        v = val_c2[i]
        if v < 1e-12:
            id1 = sorted_ids1[p2[i]]
            id2 = ids2[i]
            all_candidates.append((v, id1, id2))
            
    for i in idx3:
        v = val_c3[i]
        if v < 1e-12:
            id1 = sorted_ids1[p3[i]]
            id2 = ids2[i]
            all_candidates.append((v, id1, id2))
            
    if not all_candidates:
        # Fallback: take the smallest ones even if not tiny
        for i in idx1[:10]:
            all_candidates.append((val_c1[i], sorted_ids1[p1[i]], ids2[i]))
        for i in idx2[:10]:
            all_candidates.append((val_c2[i], sorted_ids1[p2[i]], ids2[i]))
        for i in idx3[:10]:
            all_candidates.append((val_c3[i], sorted_ids1[p3[i]], ids2[i]))

    # Verify candidates with Fraction
    best_exact_val = None
    best_coeffs = None
    
    # Sort candidates by float val to check best first
    all_candidates.sort(key=lambda x: x[0])
    
    for v, id1, id2 in all_candidates[:50]:
        # Decode id1 (15 coeffs)
        coeffs1 = [0] * 15
        m = id1
        for j in range(15):
            d = m % 3
            m //= 3
            if d == 0:
                coeffs1[j] = -1
            elif d == 1:
                coeffs1[j] = 0
            else:
                coeffs1[j] = 1
                
        # Decode id2 (15 coeffs)
        coeffs2 = [0] * 15
        m = id2
        for j in range(15):
            d = m % 3
            m //= 3
            if d == 0:
                coeffs2[j] = -1
            elif d == 1:
                coeffs2[j] = 0
            else:
                coeffs2[j] = 1
                
        all_coeffs = coeffs1 + coeffs2
        
        # Calculate exact sum
        s = Fraction(0, 1)
        for k, c in enumerate(all_coeffs):
            if c != 0:
                s += Fraction(c, k + 1)
                
        if s != 0:
            abs_s = abs(s)
            if best_exact_val is None or abs_s < best_exact_val:
                best_exact_val = abs_s
                best_coeffs = all_coeffs
                
        if time.time() > deadline:
            break
            
    if best_coeffs is None:
        # Fallback to seed-like solution
        return [0] * 28 + [1, -1]
        
    return best_coeffs
