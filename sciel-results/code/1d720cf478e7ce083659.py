import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s, task says 60s. Use 55s to be safe.
    deadline = start_time + 55.0
    
    n = 4096
    max_sum = 3 * n
    
    best_set = []
    best_size = 0
    
    def check_and_update(S, x, pair_sums, triple_sums):
        """
        Checks if x can be added to S.
        S: list of ints
        x: int
        pair_sums: np.array bool of size 2*n+1
        triple_sums: np.array bool of size 3*n+1
        Returns: (valid, new_pair_sums, new_triple_sums)
        """
        S_np = np.array(S)
        
        # New sums involving x:
        # 1. 3*x
        # 2. 2*x + y for y in S
        # 3. x + y + z for y, z in S
        
        s3 = 3 * x
        s2 = 2 * x + S_np
        # Get indices of existing pair sums
        pair_indices = np.nonzero(pair_sums)[0]
        s1 = x + pair_indices
        
        # Combine all new sums
        # Note: s2 and s1 are arrays, s3 is scalar
        # We need to check collisions with existing triple_sums AND internal collisions
        
        # 1. Check collision with existing triple_sums
        # s3
        if triple_sums[s3]:
            return False, pair_sums, triple_sums
            
        # s2
        if np.any(triple_sums[s2]):
            return False, pair_sums, triple_sums
            
        # s1
        if np.any(triple_sums[s1]):
            return False, pair_sums, triple_sums
            
        # 2. Check internal collisions among new sums
        # We can combine them into one array and check uniqueness
        # However, s1 can be large (size of pair_sums set).
        # For k ~ 40, pair_sums size is ~ 800. s1 size ~ 800.
        # s2 size ~ 40.
        # Total ~ 841 elements. Unique check is fast.
        
        all_new = np.concatenate(([s3], s2, s1))
        
        # Check for duplicates
        # Using sort + diff is often faster than set() for numpy arrays of this size
        # But set() in python might be okay for 1000 elements?
        # Let's use numpy unique
        if len(np.unique(all_new)) < len(all_new):
            return False, pair_sums, triple_sums
            
        # 3. Also need to check if new pair sums collide with existing pair sums?
        # No, the condition is only on TRIPLE sums.
        # "Every unordered triple... must have a different sum."
        # So we only care about triple sums.
        
        # Valid! Update structures
        new_pair_sums = pair_sums.copy()
        new_triple_sums = triple_sums.copy()
        
        # Update pair_sums:
        # New pairs: x+y for y in S, and x+x (2x)
        new_pairs = 2 * x + S_np
        new_pairs = np.append(new_pairs, 2*x)
        new_pair_sums[new_pairs] = True
        
        # Update triple_sums:
        # Mark all new triple sums
        new_triple_sums[all_new] = True
        
        return True, new_pair_sums, new_triple_sums

    def greedy_fill(S, pair_sums, triple_sums, candidates):
        """
        Greedily adds elements from candidates to S.
        """
        S = list(S)
        for x in candidates:
            if time.time() > deadline:
                break
                
            valid, pair_sums, triple_sums = check_and_update(S, x, pair_sums, triple_sums)
            if valid:
                S.append(x)
                if len(S) >= 80:
                    break
                    
        return S, pair_sums, triple_sums

    # Strategy 1: Sparse Skeleton (Powers of 4)
    # Start with a known valid set that grows fast to avoid early tight constraints
    # 1, 4, 16, 64, 256, 1024, 4096 (too big? 4096 is max)
    # Let's try 1, 4, 16, 64, 256, 1024
    # Max sum for these: 3*1024 = 3072. Fits in 3*4096.
    
    base_S = [1, 4, 16, 64, 256, 1024]
    # Initialize pair and triple sums for base_S
    pair_sums = np.zeros(2 * n + 1, dtype=bool)
    triple_sums = np.zeros(3 * n + 1, dtype=bool)
    
    S_np_base = np.array(base_S)
    # Compute all pair sums
    for i in range(len(base_S)):
        for j in range(i, len(base_S)):
            pair_sums[base_S[i] + base_S[j]] = True
            
    # Compute all triple sums
    for i in range(len(base_S)):
        for j in range(i, len(base_S)):
            for k in range(j, len(base_S)):
                triple_sums[base_S[i] + base_S[j] + base_S[k]] = True
                
    candidates = list(range(1, n + 1))
    S1, _, _ = greedy_fill(base_S, pair_sums, triple_sums, candidates)
    
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1

    # Strategy 2: Randomized Greedy with different orders
    # We will try a few random permutations, but use the fast check.
    # Since the check is now O(k^2) in numpy (very fast), we can afford more attempts.
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
            
        random.seed(42 + attempt)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # Start with empty set
        S = []
        pair_sums = np.zeros(2 * n + 1, dtype=bool)
        triple_sums = np.zeros(3 * n + 1, dtype=bool)
        
        S, _, _ = greedy_fill(S, pair_sums, triple_sums, candidates)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return best_set
