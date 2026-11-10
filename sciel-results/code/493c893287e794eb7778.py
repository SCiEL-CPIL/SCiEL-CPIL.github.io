# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum squared norm is less than or equal to their minimum pairwise squared distance.
    
    Strategy:
    1. Focus on vectors with exactly four non-zero entries (±1). There are C(11,4) * 2^4 = 5280 such vectors.
       All have squared norm 4.
       Condition for validity: min pairwise squared distance >= max squared norm.
       If we only use these vectors, max_norm^2 = 4.
       We need min_dist^2 >= 4.
       For two such vectors u, v: dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 8 - 2 u.v.
       We require 8 - 2 u.v >= 4 => 2 u.v <= 4 => u.v <= 2.
       
    2. We want to find the largest subset S of these 5280 vectors such that for all u, v in S, u.v <= 2.
       This is a maximum independent set problem in a graph where edges connect vectors with dot product > 2.
       Exact solution is NP-hard. We use a randomized greedy heuristic.
       
    3. Algorithm:
       - Generate all 5280 candidate vectors.
       - Precompute the dot product matrix (5280x5280) or just check on the fly? 
         5280^2 * 11 ops ~ 300M ops. In numpy, this matrix multiplication is fast (~1-2s).
         Let's precompute D = V @ V.T.
       - Create a conflict mask: M[i, j] = True if D[i, j] > 2.
       - We want to select a set of indices I such that no two indices in I are connected in M.
       - Greedy: 
         - Shuffle the indices.
         - Iterate through shuffled indices.
         - If index i is not in conflict with any already selected index, add it.
         - "In conflict" means: for all j in selected, M[i, j] is False.
         - To speed this up, maintain a boolean array 'conflicted' of size n_candidates.
           Initially all False.
           When we select index k, we mark all i such that M[i, k] is True as 'conflicted'.
           Then for the next candidate i, if 'conflicted[i]' is True, we skip it.
           This reduces the check from O(|selected|) to O(1) per candidate after the O(n) update.
           Total complexity per run: O(n^2) for the updates (since each selected k updates n entries).
           With ~200-400 selected vectors, and n=5280, this is 200 * 5280 ~ 1M ops per run. Very fast.
       - Run this greedy procedure multiple times (e.g., 20-50 times) with different random shuffles.
       - Keep the result with the largest size.
       
    4. Can we add norm-2 vectors?
       Norm-2 vectors have squared norm 2.
       If we add any norm-2 vector, the max_norm^2 of the whole set becomes max(4, 2) = 4 (if we have norm-4 vectors).
       So the condition remains min_dist^2 >= 4.
       Check compatibility of a norm-2 vector u with a norm-4 vector v:
       dist^2 = 2 + 4 - 2 u.v = 6 - 2 u.v.
       We need 6 - 2 u.v >= 4 => 2 u.v <= 2 => u.v <= 1.
       Check compatibility of two norm-2 vectors u, w:
       dist^2 = 2 + 2 - 2 u.w = 4 - 2 u.w.
       We need 4 - 2 u.w >= 4 => u.w <= 0.
       
       So if we have a set S4 of norm-4 vectors, we can try to add norm-2 vectors S2.
       Conditions for adding a norm-2 vector u:
       1. For all v in S4: u.v <= 1.
       2. For all w in S2_selected: u.w <= 0.
       
       We can apply the same randomized greedy logic to add norm-2 vectors after fixing S4.
       Or we can interleave them. 
       Given that S4 is likely much larger (potentially > 200) than S2 (max 220, but constrained), 
       and the constraint on S2 is tighter (dot <= 0 with other S2s), it might be worth it.
       However, the inspiration 1 score 208 (k=416) suggests that just optimizing S4 might get us close to 400+.
       Let's first maximize S4. Then try to add S2.
       
    5. Time management:
       - Deadline: 600s.
       - Precomputation: ~2s.
       - Each greedy run: ~10ms.
       - We can run hundreds of greedy iterations.
       
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590
    
    # Generate all 4-norm vectors (four non-zeros, ±1)
    four_norm_vecs = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            four_norm_vecs.append(v)
    
    four_norm_arr = np.array(four_norm_vecs, dtype=np.int64)
    n_four = len(four_norm_arr)
    
    # Generate all 2-norm vectors (two non-zeros, ±1)
    two_norm_vecs = []
    for i in range(d):
        for j in range(i + 1, d):
            p1 = np.zeros(d, dtype=np.int64); p1[i] = 1; p1[j] = 1
            two_norm_vecs.append(p1)
            p2 = np.zeros(d, dtype=np.int64); p2[i] = -1; p2[j] = -1
            two_norm_vecs.append(p2)
            p3 = np.zeros(d, dtype=np.int64); p3[i] = 1; p3[j] = -1
            two_norm_vecs.append(p3)
            p4 = np.zeros(d, dtype=np.int64); p4[i] = -1; p4[j] = 1
            two_norm_vecs.append(p4)
            
    two_norm_arr = np.array(two_norm_vecs, dtype=np.int64)
    n_two = len(two_norm_arr)
    
    # Precompute dot products for 4-norm vectors
    # D4[i, j] = four_norm_arr[i] dot four_norm_arr[j]
    D4 = four_norm_arr @ four_norm_arr.T
    
    # Conflict mask for 4-norm vectors: True if dot > 2
    M4 = D4 > 2
    
    # Precompute dot products for 2-norm vectors
    D2 = two_norm_arr @ two_norm_arr.T
    # Conflict mask for 2-norm vectors: True if dot > 0 (since we need dot <= 0 for dist^2 >= 4)
    M2 = D2 > 0
    
    # Precompute cross dot products
    # D42[i, j] = four_norm_arr[i] dot two_norm_arr[j]
    D42 = four_norm_arr @ two_norm_arr.T
    # Conflict mask for 4-2 pairs: True if dot > 1 (since we need dot <= 1 for dist^2 >= 4)
    M42 = D42 > 1
    
    best_S4 = []
    best_S2 = []
    
    # Randomized greedy for S4
    # We run multiple iterations with different random orders
    num_iterations = 50 # We can increase this if time permits
    
    for it in range(num_iterations):
        if time.time() > deadline - 10:
            break
            
        # Shuffle indices
        order = np.random.permutation(n_four)
        
        selected_indices = []
        # conflicted[i] is True if i conflicts with any already selected index
        conflicted = np.zeros(n_four, dtype=bool)
        
        for idx in order:
            if conflicted[idx]:
                continue
                
            # Add idx to selected
            selected_indices.append(idx)
            
            # Update conflicted array
            # Any index i such that M4[i, idx] is True becomes conflicted
            # M4[idx, :] gives conflicts with idx. Since M4 is symmetric, M4[:, idx] is the same.
            # We can just use M4[idx, :] to mark conflicts.
            # Note: M4[idx, idx] is True (dot=4 > 2), but idx is already selected, so we don't care about re-selecting it.
            # We just need to mark future candidates.
            conflicted |= M4[idx, :]
            
        # If this selection is better, keep it
        if len(selected_indices) > len(best_S4):
            best_S4 = selected_indices[:]
            
        # If we have time, try to add 2-norm vectors to this specific S4
        # We can do this in the same loop or after. Let's do it after finding the best S4,
        # or try it for every good S4? 
        # Adding 2-norms depends on S4. So for each S4 candidate, we can try to add S2.
        # But that might be too slow if we do it 50 times.
        # Let's first find the best S4, then optimize S2 for that best S4.
        # Or, we can interleave: for each S4 found, quickly greedily add S2.
        
        # Let's try adding S2 for the current S4 (selected_indices)
        # We need to check compatibility with S4 and with already selected S2.
        
        # Pre-computed M42 is (n_four, n_two).
        # For a given S4 (list of 4-indices), a 2-vector j is compatible with S4 if:
        # for all i in S4, M42[i, j] is False.
        # This is equivalent to: (M42[S4, :].any(axis=0) == False)
        
        # Let's compute a mask 'incompatible_with_S4' for all 2-vectors
        if selected_indices:
            # M42[selected_indices, :] has shape (len(S4), n_two)
            # .any(axis=0) gives shape (n_two,)
            incompat_4 = M42[selected_indices, :].any(axis=0)
        else:
            incompat_4 = np.zeros(n_two, dtype=bool)
            
        # Now greedily select S2
        # Condition for 2-vectors: dot <= 0 with other selected 2-vectors.
        # M2[i, j] is True if dot > 0.
        
        order2 = np.random.permutation(n_two)
        selected_indices_2 = []
        conflicted_2 = np.zeros(n_two, dtype=bool)
        
        for idx2 in order2:
            if incompat_4[idx2]:
                continue
            if conflicted_2[idx2]:
                continue
                
            # Add idx2
            selected_indices_2.append(idx2)
            
            # Update conflicted_2
            conflicted_2 |= M2[idx2, :]
            
        # Check if total size is better
        total_size = len(selected_indices) + len(selected_indices_2)
        best_total_size = len(best_S4) + len(best_S2)
        
        if total_size > best_total_size:
            best_S4 = selected_indices[:]
            best_S2 = selected_indices_2[:]
            
    # Construct final array
    if best_S4:
        final_4 = four_norm_arr[best_S4]
    else:
        final_4 = np.empty((0, d), dtype=np.int64)
        
    if best_S2:
        final_2 = two_norm_arr[best_S2]
    else:
        final_2 = np.empty((0, d), dtype=np.int64)
        
    if len(final_4) > 0 and len(final_2) > 0:
        final_points = np.vstack([final_4, final_2])
    elif len(final_4) > 0:
        final_points = final_4
    elif len(final_2) > 0:
        final_points = final_2
    else:
        # Fallback: return at least 2 valid points if something went wrong
        # e.g., e1, e2
        p1 = np.zeros(d, dtype=np.int64); p1[0] = 1; p1[1] = 1
        p2 = np.zeros(d, dtype=np.int64); p2[0] = 1; p2[1] = -1
        final_points = np.vstack([p1, p2])
        
    return final_points.astype(np.int64)

# EVOLVE-BLOCK-END
