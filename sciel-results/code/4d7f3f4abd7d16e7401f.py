import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0 -> 2*11 = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0 -> C(11,4) * 2^4 = 330 * 16 = 5280
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute conflict masks
    # M = candidates @ candidates.T
    # conflict if M[i,j] > 2
    # We can compute dot products in chunks to save memory if N was larger, 
    # but 5302x5302 is ~28M entries, fits in memory.
    
    M = candidates @ candidates.T
    
    conflict_masks = [0] * N
    for i in range(N):
        # Get indices where M[i, j] > 2
        # M[i, :] is a numpy array
        # We need to convert to bitmask
        # This loop might be slow in pure Python. 
        # Let's vectorize the mask creation.
        pass
        
    # Faster mask creation:
    # Create a boolean matrix of conflicts
    # conflicts = (M > 2)
    # Then for each i, convert row to int bitmask.
    
    # Numpy doesn't have direct to-bitmask.
    # We can use:
    # for i in range(N):
    #     row = (M[i] > 2)
    #     # Convert boolean array to int bitmask
    #     # This is still O(N^2) in Python loops if done naively.
    
    # Let's try to be efficient.
    # Since N is ~5300, N^2 is 28M.
    # Python loop over 5300 rows is fine.
    # Converting a 5300-bit array to int is fast if done right.
    
    # Alternative: Just use the M matrix directly in the SA?
    # No, bitmask is much faster for intersection checks.
    
    # Let's build the masks.
    # We can use a trick:
    # np.packbits? No, we need integer bitmask.
    
    # Let's just do it. 5300 iterations is nothing.
    for i in range(N):
        # M[i, :] is int32 array
        # We want bits set where M[i, j] > 2
        # We can iterate over non-zeros?
        # Or just use a loop?
        # Actually, we can pre-compute the list of conflicting indices for each i.
        # But for SA, we need fast "is j in conflict with any in S?".
        # Bitmask AND is best.
        
        # How to convert np array of bools to int bitmask quickly?
        # We can sum: sum((M[i] > 2) << np.arange(N))? 
        # np.arange(N) creates a large array.
        
        # Let's stick to the list of indices for conflicts?
        # If we use list of indices, checking compatibility is:
        # for j in conflicts[i]: if j in S: return False
        # This is O(deg(i)). Deg is likely small.
        # Let's check average degree.
        # Total edges?
        # If graph is sparse, list-based is faster than bitmask construction.
        
        # Let's estimate edges.
        # Type 1 (22 nodes) connect to Type 2 if dot > 2.
        # Type 1 dot Type 2 is always <= 2. So no edges between Type 1 and Type 2.
        # Type 1 nodes are independent of each other?
        # v1=(2,0..), v2=(-2,0..). Dot -4. OK.
        # v1=(2,0..), v3=(0,2..). Dot 0. OK.
        # So Type 1 is an independent set.
        
        # Type 2 nodes:
        # Conflict if dot > 2.
        # Max dot 4 (self).
        # Dot 3: share 3 indices with same sign.
        # For a given Type 2 vector, how many others have dot 3?
        # It has 4 indices.
        # Pick 3 of them. There are C(4,3)=4 ways.
        # For each set of 3, the 4th index of the other vector must be different from the current one's 4th index.
        # And the signs on the 3 shared indices must be the same.
        # The sign on the new 4th index can be +/-1.
        # So for each choice of 3 indices (4 ways), there are 2 choices for the new index position (11-4=7 choices? No, the new index must be one of the other 7 dimensions).
        # Wait, the other vector must have exactly 4 non-zeros.
        # If it shares 3, it has 1 unique non-zero.
        # The unique non-zero can be in any of the other 7 dimensions.
        # And it can be +1 or -1.
        # So for each of the 4 subsets of 3 indices, there are 7 * 2 = 14 vectors that conflict?
        # Let's verify.
        # v = (1,1,1,1, 0,0,0,0,0,0,0)
        # Conflict if w has 1s at 1,2,3 and a non-zero at k (k!=4).
        # w = (1,1,1, 0, 1, 0,0,0,0,0,0). Dot 3.
        # w = (1,1,1, 0, -1, 0,0,0,0,0,0). Dot 3.
        # Yes.
        # So each Type 2 node conflicts with 4 * 7 * 2 = 56 other Type 2 nodes?
        # Plus maybe some with dot 4? No, distinct vectors.
        # So degree is ~56.
        # Total edges ~ 5280 * 56 / 2 ~ 147k.
        # Very sparse!
        
        # So list-based adjacency is better.
        
        adj = [[] for _ in range(N)]
        # Compute dot products and fill adj
        # We can do this efficiently with numpy
        # M = candidates @ candidates.T
        # for i in range(N):
        #     js = np.where(M[i] > 2)[0]
        #     for j in js:
        #         if i != j:
        #             adj[i].append(j)
    #             # adj[j].append(i) # Not needed if we only check one way?
    #             # For SA, we need to check if j conflicts with any in S.
    #             # If we have adj[i], we can check if any element of adj[i] is in S.
    
    # Let's use the bitmask approach but build it carefully.
    # Or just use the list approach with a set for S?
    # Set membership is O(1).
    # Check compatibility of i:
    # for j in adj[i]:
    #     if j in S_set: return False
    # This is O(deg(i)).
    
    # Given deg is ~56, this is very fast.
    # And we don't need to build huge bitmasks.
    
    # Let's re-evaluate.
    # S is a set of integers.
    # Add i:
    #   for j in adj[i]:
    #       if j in S: return False
    #   S.add(i)
    
    # Remove i:
    #   S.remove(i)
    
    # This is O(deg) for add, O(1) for remove.
    # Very simple and likely fast enough for Python.
    
    # Let's implement this.
    
    # 1. Build adj list
    M = candidates @ candidates.T
    adj = [[] for _ in range(N)]
    # Vectorized finding of indices
    # We can iterate over i
    for i in range(N):
        # M[i, :] > 2
        # Exclude i itself
        js = np.where(M[i] > 2)[0]
        for j in js:
            if i != j:
                adj[i].append(int(j))
                
    # 2. Greedy + SA
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_S = set()
    
    # Greedy Initializations
    # Order by degree (ascending) to pick low-degree nodes first?
    # Or descending?
    # Low degree nodes are less likely to conflict.
    degrees = [len(a) for a in adj]
    
    orderings = [
        np.argsort(degrees),
        np.argsort(-np.array(degrees)),
        np.arange(N)
    ]
    for _ in range(20):
        orderings.append(rng.permutation(N))
        
    for order in orderings:
        if time.time() > deadline - 60:
            break
        S = set()
        for i in order:
            # Check if i is compatible with all in S
            compatible = True
            for j in adj[i]:
                if j in S:
                    compatible = False
                    break
            if compatible:
                S.add(i)
                
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    # Simulated Annealing
    if time.time() < deadline - 5 and best_S:
        S = best_S.copy()
        current_size = len(S)
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        while time.time() < deadline - 2 and T > T_min:
            move_type = rng.random()
            
            if move_type < 0.5:
                # Add
                # Pick random i not in S
                # To avoid bias, pick from all N
                i = rng.integers(N)
                if i in S:
                    continue
                # Check compatibility
                compatible = True
                for j in adj[i]:
                    if j in S:
                        compatible = False
                        break
                if compatible:
                    S.add(i)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                        
            elif move_type < 0.8:
                # Swap
                if not S:
                    continue
                i = list(S)[rng.integers(len(S))]
                j = rng.integers(N)
                if j in S:
                    continue
                # Check if j is compatible with S \ {i}
                # i.e., j is not compatible with any k in S where k != i
                compatible = True
                for k in adj[j]:
                    if k in S and k != i:
                        compatible = False
                        break
                if compatible:
                    S.remove(i)
                    S.add(j)
                    # Size unchanged
                    
            else:
                # Remove
                if not S:
                    continue
                i = list(S)[rng.integers(len(S))]
                S.remove(i)
                current_size -= 1
                # Accept with prob exp(-1/T)
                if rng.random() > np.exp(-1.0 / T):
                    # Revert
                    S.add(i)
                    current_size += 1
            
            T *= cooling
            
    # Final Greedy Expansion
    if time.time() < deadline - 1 and best_S:
        S = best_S.copy()
        # Try to add any compatible node
        changed = True
        while changed:
            changed = False
            for i in range(N):
                if i in S:
                    continue
                compatible = True
                for j in adj[i]:
                    if j in S:
                        compatible = False
                        break
                if compatible:
                    S.add(i)
                    changed = True
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    if not best_S:
        # Fallback
        result = candidates[:22].astype(np.int64)
        return result
        
    result = candidates[list(best_S)].astype(np.int64)
    
    # Verify
    n = len(result)
    if n < 2:
        # Fallback to a known valid small set if we somehow ended up with <2
        # The 22 Type 1 vectors are valid.
        return candidates[:22].astype(np.int64)
        
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    diffs = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    diff_norms_sq = np.sum(diffs**2, axis=2)
    min_dist_sq = np.min(diff_norms_sq[np.triu_indices(n, k=1)])
    if min_dist_sq < max_norm_sq:
        return candidates[:22].astype(np.int64)
        
    return result
