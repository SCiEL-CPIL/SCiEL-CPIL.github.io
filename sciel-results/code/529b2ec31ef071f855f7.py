import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We use 145s.
    deadline = start_time + 145.0
    
    N = 256
    
    # 1. Precompute all 4-term arithmetic progressions (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    # d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # 2. Precompute incidence matrix M (N x num_aps)
    # M[i, j] = 1 if index i is in AP j, else 0
    # Since each AP has 4 elements, we can build this efficiently.
    # Instead of a full boolean matrix which might be sparse and large, 
    # we can exploit the structure.
    # However, for N=256 and ~10k APs, a boolean matrix is 256 * 10000 = 2.56M booleans.
    # This is small enough to keep in memory and use for vectorized ops.
    
    # Actually, we can compute the delta for ALL flips at once more efficiently 
    # using the property:
    # Delta(i) = sum_{j in APs containing i} (mono_after(i,j) - mono_before(i,j))
    
    # Let's precompute for each AP j, the 4 indices.
    # aps is already (num_aps, 4).
    
    # We need to calculate, for the current coloring c:
    # 1. Which APs are currently mono?
    # 2. For each index i, if we flip it, which APs become mono or stop being mono?
    
    # Vectorized approach for "Best Flip":
    # Let c be the current coloring (0/1).
    # An AP is mono if all 4 bits are equal.
    # mono_mask = (c[a]==c[b]) & (c[b]==c[c]) & (c[c]==c[d])
    
    # If we flip index i:
    # The APs affected are those containing i.
    # For an AP j containing i, let the other 3 indices be o1, o2, o3.
    # Before flip: mono if c[i]==c[o1]==c[o2]==c[o3].
    # After flip: mono if (1-c[i])==c[o1]==c[o2]==c[o3].
    
    # This is hard to vectorize across ALL i simultaneously without heavy memory usage 
    # if we create a 4D tensor.
    
    # Alternative: Iterative Greedy with Vectorized "Score Improvement" calculation.
    # We can calculate the "potential" of each flip by looking at the neighbors.
    
    # Let's use a different vectorized trick.
    # For each AP j, define its state based on the counts of 0s and 1s.
    # mono if count_0==4 or count_1==4.
    
    # Let's try a "Block Coordinate Descent" or "Exhaustive Greedy" where we 
    # evaluate the delta for all i in O(N * avg_degree) but vectorized per AP?
    
    # Actually, the most efficient vectorized greedy step for small N:
    # 1. Compute current mono_mask (num_aps,)
    # 2. For each i, we want delta_i.
    #    delta_i = sum_{j: i in j} ( new_mono_j - old_mono_j )
    
    # We can precompute for each i, the list of APs it touches.
    # Then, for each i, we check the state of those APs.
    # This is what the SA did, but sequentially.
    
    # To make it "Exhaustive Greedy", we do:
    # Loop:
    #   Find i* that minimizes (maximizes improvement) score.
    #   Flip i*.
    #   Repeat until no improvement.
    
    # To speed up "Find i*", we can vectorize the delta calculation for ALL i.
    # Let's build a structure that allows this.
    
    # For each AP j, we have 4 indices.
    # Let's create arrays:
    # ap_indices: (num_aps, 4)
    
    # We want to compute, for each i, the change in mono status for APs containing i.
    
    # Let's precompute:
    # For each AP j, the 4 indices are idxs[j, 0..3].
    # Let c be the current coloring.
    
    # We can compute the "mono-ness" contribution.
    # Let's try to compute the delta for all i using broadcasting, but it's O(N * num_aps) 
    # which is 256 * 10000 = 2.5M operations. This is fast in NumPy!
    
    # Strategy:
    # 1. Current coloring c (N,)
    # 2. Compute mono_mask (num_aps,)
    # 3. For each i in 0..N-1:
    #    We need to know which APs contain i.
    #    This is sparse.
    
    # Better:
    # Create a matrix A (N, num_aps) where A[i, j] = 1 if i in AP j.
    # This is 256 * 10000 = 2.56M elements. Manageable.
    
    # Let's build A.
    A = np.zeros((N, num_aps), dtype=np.int8)
    for j in range(num_aps):
        for k in range(4):
            A[aps[j, k], j] = 1
    
    # Now, for a given coloring c:
    # mono_mask = (c[aps[:,0]] == c[aps[:,1]]) & ...
    
    # To find the best flip:
    # We simulate flipping each i.
    # c_flip_i = c.copy(); c_flip_i[i] = 1 - c_flip_i[i]
    # This is O(N^2) if done naively.
    
    # Vectorized Delta Calculation:
    # For AP j, let the 4 values be v0, v1, v2, v3.
    # mono_j = 1 if v0==v1==v2==v3 else 0.
    
    # If we flip index i, only APs where A[i, j]=1 are affected.
    # For an affected AP j, let the value at i be v_i.
    # The other 3 values are o1, o2, o3.
    # old_mono = 1 if v_i == o1 == o2 == o3
    # new_mono = 1 if (1-v_i) == o1 == oo2 == o3
    
    # Note: new_mono is 1 ONLY IF o1==o2==o3 AND (1-v_i) == o1.
    # old_mono is 1 ONLY IF o1==o2==o3 AND v_i == o1.
    
    # So, for a flip to change the mono status, the other 3 MUST be equal.
    # If the other 3 are NOT equal, the AP was not mono and won't be mono after flip.
    # Delta = 0.
    
    # If the other 3 ARE equal (let val = o1):
    #   If v_i == val: old_mono=1, new_mono=0 => delta = -1
    #   If v_i != val: old_mono=0, new_mono=1 => delta = +1
    
    # So, delta_i = sum_{j in APs(i)} ( -1 if c[i]==c[other3] and other3 equal, else +1 if c[i]!=c[other3] and other3 equal, else 0 )
    
    # We can precompute for each i, the list of APs it belongs to, and the indices of the other 3 elements.
    # But we want to vectorize.
    
    # Let's precompute:
    # For each AP j, we have 4 slots.
    # Let's create 4 arrays: 
    # pos0, pos1, pos2, pos3 are the indices of the 4 elements.
    
    # We can compute a matrix "others_equal" (num_aps, 4) where 
    # others_equal[j, k] = 1 if the 3 elements in AP j excluding position k are equal.
    
    # This is expensive to recompute every step? 
    # c changes, so "others_equal" changes.
    
    # However, "others_equal" depends on c.
    
    # Let's compute "others_equal" for the current c.
    # For AP j, elements are e0, e1, e2, e3.
    # others_equal[j, 0] = 1 if c[e1]==c[e2]==c[e3]
    # others_equal[j, 1] = 1 if c[e0]==c[e2]==c[e3]
    # others_equal[j, 2] = 1 if c[e0]==c[e1]==c[e3]
    # others_equal[j, 3] = 1 if c[e0]==c[e1]==c[e2]
    
    # We can compute this vectorized:
    # c_e0 = c[aps[:, 0]]
    # c_e1 = c[aps[:, 1]]
    # c_e2 = c[aps[:, 2]]
    # c_e3 = c[aps[:, 3]]
    
    # oe0 = (c_e1 == c_e2) & (c_e2 == c_e3)
    # oe1 = (c_e0 == c_e2) & (c_e2 == c_e3)
    # oe2 = (c_e0 == c_e1) & (c_e1 == c_e3)
    # oe3 = (c_e0 == c_e1) & (c_e1 == c_e2)
    
    # Now, for a flip at index i:
    # It affects APs j where A[i, j] == 1.
    # In those APs, i is at some position k.
    # The delta is:
    # If oe_k == 1:
    #    delta += 1 if c[i] != c[other_representative] else -1
    #    Note: if oe_k==1, all others are equal to each other. 
    #    Let val_other = c[other_index].
    #    If c[i] == val_other: delta -= 1
    #    If c[i] != val_other: delta += 1
    
    # We need to map i to its position k in AP j.
    # This is tricky to vectorize fully without a lookup table for positions.
    
    # Let's precompute:
    # pos_matrix (num_aps, 4) is just range(4) repeated? No, aps has the indices.
    # We need to know, for a given i, which APs it is in and at what position.
    
    # Let's create arrays:
    # ap_of_index[i] = list of (ap_idx, pos)
    # This is what we had before.
    
    # To vectorize the delta calculation for ALL i at once:
    # We can't easily broadcast because the "other" values depend on the specific AP.
    
    # However, N=256 is small. 
    # A loop over i (256 iterations) where each iteration does vectorized ops on the APs containing i 
    # is O(N * avg_degree). Avg degree is ~40. 256 * 40 = 10k ops. Very fast.
    
    # So, "Exhaustive Greedy" step:
    # 1. Compute current mono_mask.
    # 2. For each i in 0..N-1:
    #      Calculate delta_i.
    # 3. Find i* with min delta (max improvement).
    # 4. If delta_i* < 0, flip i*. Else, break (local minimum).
    
    # To escape local minima, we can do a "kick":
    # Flip a random number of bits (e.g., 5-10) and restart greedy.
    
    # Let's implement this.
    
    # Precompute neighbor lists for fast delta calculation
    # index_to_aps[i] = list of (ap_idx, pos, o1, o2, o3)
    index_to_aps = [[] for _ in range(N)]
    for j in range(num_aps):
        a, b, c_idx, d_idx = aps[j]
        index_to_aps[a].append((j, 0, b, c_idx, d_idx))
        index_to_aps[b].append((j, 1, a, c_idx, d_idx))
        index_to_aps[c_idx].append((j, 2, a, b, d_idx))
        index_to_aps[d_idx].append((j, 3, a, b, c_idx))
        
    # Convert to numpy arrays for faster access in the inner loop?
    # Or just use lists. Lists of tuples are fast for small sizes.
    
    # Helper to calculate score
    def calc_score(c):
        ca = c[aps[:, 0]]
        cb = c[aps[:, 1]]
        cc = c[aps[:, 2]]
        cd = c[aps[:, 3]]
        is_mono = (ca == cb) & (cb == cc) & (cc == cd)
        return np.sum(is_mono)
        
    # Helper to calculate deltas for all i
    def calc_deltas(c):
        deltas = np.zeros(N, dtype=np.int32)
        for i in range(N):
            if not index_to_aps[i]:
                continue
            val_i = c[i]
            new_val_i = 1 - val_i
            delta = 0
            for ap_idx, pos, o1, o2, o3 in index_to_aps[i]:
                v1 = c[o1]
                v2 = c[o2]
                v3 = c[o3]
                
                # Check if others are equal
                if v1 == v2 and v2 == v3:
                    # If others are equal, the AP is mono iff val_i matches them
                    if val_i == v1:
                        # Was mono, will not be mono
                        delta -= 1
                    else:
                        # Was not mono, will be mono
                        delta += 1
            deltas[i] = delta
        return deltas

    rng = np.random.default_rng()
    
    best_score = np.inf
    best_coloring = None
    
    # Multi-start Greedy
    num_restarts = 20
    max_greedy_steps = 100 # Usually converges in < 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        # Random start
        c = rng.integers(0, 2, size=N, dtype=np.int8)
        
        # Optional: Start with a heuristic? e.g. alternating?
        # c = np.arange(N) % 2
        
        current_score = calc_score(c)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = c.copy()
            
        # Greedy Loop
        for step in range(max_greedy_steps):
            if time.time() > deadline - 2:
                break
                
            deltas = calc_deltas(c)
            min_delta = np.min(deltas)
            
            if min_delta >= 0:
                # Local minimum reached
                break
                
            # Find all i with min_delta
            # If multiple, pick one randomly? Or the first?
            # Picking randomly among best can help escape symmetry?
            candidates = np.where(deltas == min_delta)[0]
            i_star = candidates[rng.integers(0, len(candidates))]
            
            c[i_star] = 1 - c[i_star]
            current_score += min_delta
            
            if current_score < best_score:
                best_score = current_score
                best_coloring = c.copy()
                
        # Kick to escape local minimum
        # Flip 5 random bits
        for _ in range(5):
            k = rng.integers(0, N)
            c[k] = 1 - c[k]
            
        # Run greedy again on the kicked coloring
        for step in range(max_greedy_steps):
            if time.time() > deadline - 2:
                break
                
            deltas = calc_deltas(c)
            min_delta = np.min(deltas)
            
            if min_delta >= 0:
                break
                
            candidates = np.where(deltas == min_delta)[0]
            i_star = candidates[rng.integers(0, len(candidates))]
            
            c[i_star] = 1 - c[i_star]
            current_score += min_delta
            
            if current_score < best_score:
                best_score = current_score
                best_coloring = c.copy()

    # Final check
    if best_coloring is None:
        return [0] * N
        
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
