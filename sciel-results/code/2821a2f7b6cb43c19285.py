import time
import random
import numpy as np

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 88.0  # Safety margin against 90s limit
    
    # Precompute pair sums for all pairs (i, j) with i < j
    # We only care about sums <= n, as elements are in 1..n
    # Actually, sums can be up to 2n, but if sum > n, it can't be in the set.
    # So for constraint checking, we only care if sum is in S.
    # If sum > n, it's automatically not in S.
    
    # Generate all pairs (i, j) with 0 <= i < j < n (indices for 0..n-1, values 1..n)
    # Let's use 0-indexed values internally: v = idx + 1
    # Pair (i, j) -> sum = (i+1) + (j+1) = i + j + 2
    
    # We need to check triples (a, b, c) such that a+b, a+c, b+c are in S.
    # This is equivalent to: for any triple (i, j, k), 
    # NOT ( (i+j+2 in S) AND (i+k+2 in S) AND (j+k+2 in S) )
    # where "in S" means the corresponding index (sum-1) is active, provided sum <= n.
    # If sum > n, the condition "sum in S" is False, so the triple is safe.
    
    # Vectorized Check Function
    # Input: mask (bool array of length n)
    # Output: bool (True if valid)
    
    # Precompute indices for pairs
    # We can vectorize the check for a single mask by checking all triples?
    # N=128, triples = 128^3/6 ~ 350k. 
    # Doing this in NumPy for each candidate is fast.
    # But we can do better: 
    # A set is invalid if there exists a triple (a,b,c) in S such that all three sums are in S.
    # Let's precompute a list of all "bad" triples? No, that's 350k triples.
    # For each triple, we can check if a,b,c are in S AND sums are in S.
    
    # Alternative: 
    # For a given mask S, we want to know if there is ANY triple (a,b,c) in S 
    # such that s_ab, s_ac, s_bc are also in S.
    
    # Let's create a 3D boolean array? No, too big.
    
    # Let's iterate through all pairs (a,b) in S.
    # For each pair, check if a+b in S. If so, we need to find if there is a c in S 
    # (c != a, c != b) such that a+c in S and b+c in S.
    
    # This is still potentially slow if done in Python loops.
    
    # Let's try a different vectorized approach for the validity check:
    # 1. Get indices of active elements: S_idx = np.where(mask)[0]
    # 2. If len(S_idx) < 3, valid.
    # 3. For each pair (i, j) in S_idx:
    #    sum_ij = S_idx[i] + S_idx[j] + 2
    #    if sum_ij <= n and mask[sum_ij-1]:
    #       # Now we need to check if there exists k in S_idx (k != i, k != j)
    #       # such that mask[S_idx[k] + S_idx[i] + 2 - 1] and mask[S_idx[k] + S_idx[j] + 2 - 1]
    #       # This is getting complicated to fully vectorize per-pair.
    
    # Let's use a simpler, robust vectorized check that handles the whole set at once.
    # We can precompute all triples (a,b,c) as indices.
    # There are C(128,3) = 341,376 triples.
    # We can store these as a 3x341376 array of indices (0-based).
    # Then for a mask, we check:
    # mask[A] & mask[B] & mask[C] & mask[sumAB-1] & mask[sumAC-1] & mask[sumBC-1]
    # where sumAB = A+B+2, etc.
    # We must handle sum > n by masking those out.
    
    # Precomputation
    # Generate all triples
    # This might take a second or two to generate, but it's one-time.
    # Let's do it efficiently.
    
    # Actually, generating 341k triples in Python might take ~1-2 seconds.
    # Is it worth it?
    # If we can check validity in ~1ms, we can do 80,000 checks in 80s.
    # Random search with 80k restarts is very powerful.
    
    # Let's precompute the triples and their sum indices.
    # If sum > n, we can set the sum index to a "dummy" index that is always False in the mask, 
    # or simply mask it out in the boolean logic.
    
    # Let's create arrays:
    # A, B, C: indices of the triple elements
    # SA, SB, SC: indices of the sums (A+B+2-1, etc.), or -1 if sum > n
    
    # To avoid generating 341k rows in Python (slow), we can use a more clever check?
    # Or just accept the 2s precomputation.
    
    # Let's try to generate triples using numpy meshgrid?
    # i, j, k ranges 0..127 with i<j<k.
    # This is hard to vectorize directly without loops.
    
    # Alternative:
    # Check validity by iterating pairs (a,b) in S.
    # If a+b in S, then we need to check if there is c in S such that a+c in S and b+c in S.
    # This can be vectorized for a fixed pair (a,b) against the rest of S.
    # But iterating over pairs in S is O(K^2) where K is size of S.
    # If K ~ 70, K^2 ~ 5000.
    # 5000 vectorized checks per validity check.
    # Each vectorized check is O(K).
    # Total O(K^3) but with small constants (NumPy).
    # 5000 * 70 = 350k ops. 
    # This is similar to the Python loop but 100x faster.
    
    # Let's implement this "Pair Iteration with Vectorized Inner Check".
    
    def is_valid_vectorized(mask):
        # mask: bool array of length n
        S_idx = np.where(mask)[0]
        k = len(S_idx)
        if k < 3:
            return True
        
        # We need to check if there exist i < j < l in S_idx such that:
        # mask[S_idx[i] + S_idx[j] + 2 - 1]
        # mask[S_idx[i] + S_idx[l] + 2 - 1]
        # mask[S_idx[j] + S_idx[l] + 2 - 1]
        
        # Iterate over pairs (i, j)
        # For each pair, compute sum_ij_idx = S_idx[i] + S_idx[j] + 1
        # If sum_ij_idx < n and mask[sum_ij_idx]:
        #    We need to find l in S_idx (l != i, l != j) such that:
        #    mask[S_idx[l] + S_idx[i] + 1] and mask[S_idx[l] + S_idx[j] + 1]
        
        # Precompute for all l in S_idx:
        # sum_li = S_idx[l] + S_idx[i] + 1
        # sum_lj = S_idx[l] + S_idx[j] + 1
        
        # We can vectorize over l for each pair (i,j).
        
        for i in range(k):
            for j in range(i+1, k):
                sum_ij = S_idx[i] + S_idx[j] + 1 # 0-based index for sum
                if sum_ij < n and mask[sum_ij]:
                    # Check all l != i, j
                    # We can just check all l in S_idx, and mask out i and j later?
                    # Or create a list of valid l's.
                    
                    # Vectorized check for all l in S_idx:
                    # sum_li = S_idx + S_idx[i] + 1
                    # sum_lj = S_idx + S_idx[j] + 1
                    
                    # We need mask[sum_li] and mask[sum_lj] to be True.
                    # And sum_li, sum_lj < n.
                    
                    # Create boolean arrays for validity of sums
                    # Note: S_idx contains 0-based indices.
                    # sum_li = S_idx + S_idx[i] + 1
                    # If sum_li >= n, it's invalid.
                    
                    sum_li = S_idx + S_idx[i] + 1
                    sum_lj = S_idx + S_idx[j] + 1
                    
                    valid_li = (sum_li < n) & mask[sum_li]
                    valid_lj = (sum_lj < n) & mask[sum_lj]
                    
                    # We need an l such that valid_li[l] and valid_lj[l]
                    # And l != i and l != j.
                    
                    # Since i and j are indices in S_idx array, not values.
                    # Wait, S_idx is an array of values (0-based indices of original set).
                    # i and j are indices into S_idx.
                    # So l is an index into S_idx.
                    
                    # We need to exclude l=i and l=j from the candidates.
                    
                    # Create a mask for l indices
                    l_mask = np.ones(k, dtype=bool)
                    l_mask[i] = False
                    l_mask[j] = False
                    
                    # Check if any l satisfies conditions
                    if np.any(valid_li & valid_lj & l_mask):
                        return False
        
        return True

    # This is O(K^3) but vectorized inner loop.
    # K ~ 70. K^2 = 4900 pairs.
    # Inner loop is O(K) vectorized.
    # Total ops ~ 4900 * 70 = 343k NumPy ops.
    # This should be ~1ms per check.
    
    # Strategy:
    # 1. Start with random masks.
    # 2. Use local search: flip bits to improve validity/size.
    #    Actually, we want to MAXIMIZE size subject to validity.
    #    So we can start with a random set, make it valid by removing elements,
    #    then try to add elements.
    
    # Better Strategy: Randomized Greedy Construction + Local Search
    # 1. Generate many random initial sets (e.g., random size, or random mask).
    # 2. For each, "repair" to valid state by removing elements that cause conflicts.
    #    How to repair? 
    #    If invalid, find a conflicting triple and remove one of its elements.
    #    Repeat until valid.
    # 3. Then try to add elements (greedy or random) to increase size.
    
    # Finding a conflicting triple quickly:
    # We can modify is_valid to return the first conflict found.
    
    def find_conflict(mask):
        S_idx = np.where(mask)[0]
        k = len(S_idx)
        if k < 3:
            return None
        
        for i in range(k):
            for j in range(i+1, k):
                sum_ij = S_idx[i] + S_idx[j] + 1
                if sum_ij < n and mask[sum_ij]:
                    sum_li = S_idx + S_idx[i] + 1
                    sum_lj = S_idx + S_idx[j] + 1
                    valid_li = (sum_li < n) & mask[sum_li]
                    valid_lj = (sum_lj < n) & mask[sum_lj]
                    
                    l_mask = np.ones(k, dtype=bool)
                    l_mask[i] = False
                    l_mask[j] = False
                    
                    valid_l_indices = np.where(valid_li & valid_lj & l_mask)[0]
                    if len(valid_l_indices) > 0:
                        l = valid_l_indices[0]
                        # Conflict involves S_idx[i], S_idx[j], S_idx[l]
                        return (S_idx[i], S_idx[j], S_idx[l])
        return None

    # Repair function:
    def repair(mask):
        # Remove elements until valid
        # To maximize size, we should remove the element that "causes" the most conflicts?
        # Or just remove one randomly?
        # Removing one from the conflict triple is safe.
        # Let's try removing the smallest element? Or random?
        # Random might be better for diversity.
        
        while True:
            conflict = find_conflict(mask)
            if conflict is None:
                break
            a, b, c = conflict
            # Remove one of them. 
            # Heuristic: Remove the one with smallest value? 
            # Or the one that appears in most potential conflicts?
            # Let's just remove the smallest one to keep larger numbers (which are safer?)
            # Actually, larger numbers have sums > n more often, so they are "safer" to keep.
            # So remove the smallest one.
            to_remove = min(a, b, c)
            mask[to_remove] = False
        
        return mask

    # Local Search:
    def local_search(mask):
        # Try to add elements
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Try adding each element not in mask
            # To speed up, try random order? Or sorted?
            # Sorted might be deterministic. Random is better for exploration.
            candidates = np.where(~mask)[0]
            random.shuffle(candidates)
            
            for c in candidates:
                if time.time() > deadline - 1:
                    break
                # Try adding c
                mask[c] = True
                if find_conflict(mask) is None:
                    # Valid, keep it
                    improved = True
                else:
                    # Revert
                    mask[c] = False
        
        return mask

    best_size = 0
    best_mask = None
    
    # Initial good mask: Upper half
    init_mask = np.zeros(n, dtype=bool)
    init_mask[63:] = True # 64..128 -> indices 63..127
    # Repair it (should be valid already)
    init_mask = repair(init_mask)
    best_mask = init_mask.copy()
    best_size = np.sum(best_mask)
    
    # Randomized Search
    num_restarts = 50000 # We will break based on time
    restart_count = 0
    
    while time.time() < deadline - 2:
        restart_count += 1
        
        # Generate random mask
        # Bias towards larger sets?
        # Or start with random density?
        # Let's try random density between 0.3 and 0.8
        density = random.uniform(0.3, 0.8)
        mask = np.random.rand(n) < density
        
        # Repair
        mask = repair(mask)
        
        # Local Search
        mask = local_search(mask)
        
        size = np.sum(mask)
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
            
            # If we found a very large set, maybe we can try to improve it further?
            # The local search already tried to add elements.
    
    # Convert best_mask to list of ints
    # Indices are 0-based, values are 1-based
    result_indices = np.where(best_mask)[0]
    result = [int(i) + 1 for i in result_indices]
    
    # Ensure types are correct
    return result

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    S = set(res)
    valid = True
    for a in S:
        for b in S:
            if a >= b: continue
            for c in S:
                if b >= c: continue
                if (a+b in S) and (a+c in S) and (b+c in S):
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
