import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Strategy: Focus on Weight 4 vectors (norm^2 = 16).
    # Why Weight 4?
    # For weight w, norm^2 = w^2? No, vectors have entries 0, 1, -1.
    # Weight w means w non-zero entries. Norm^2 = w.
    # Wait, the previous code used weights 2..6 with entries 0, 1, -1.
    # So Norm^2 = w.
    # Let's re-evaluate.
    # If entries are 0, 1, -1:
    # Weight 2: Norm^2 = 2. Dist^2 >= 2.
    # Weight 4: Norm^2 = 4. Dist^2 >= 4.
    # Weight 8: Norm^2 = 8. Dist^2 >= 8.
    
    # Let's check the condition for Weight 4 (Norm^2 = 4).
    # Vectors have four 1s/-1s and seven 0s.
    # Two vectors u, v.
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # We need ||u-v||^2 >= 4 (since max norm is 2, dist >= 2, dist^2 >= 4).
    # 8 - 2 u.v >= 4 => 4 >= 2 u.v => u.v <= 2.
    # u.v is the sum of products. Since entries are 0, 1, -1:
    # u.v = (number of positions where both are 1) - (number of positions where both are -1).
    # Let I be the set of indices where u is non-zero (size 4).
    # Let J be the set of indices where v is non-zero (size 4).
    # Overlap size |I \cap J| can be 0, 1, 2, 3, 4.
    # If overlap is 0: u.v = 0. 0 <= 2. Valid.
    # If overlap is 1: u.v is 1 or -1. Valid.
    # If overlap is 2: u.v is 2, 0, or -2.
    #   If u.v = 2 (both same sign on 2 overlaps): 2 <= 2. Valid.
    #   If u.v = 0: Valid.
    #   If u.v = -2: Valid.
    # If overlap is 3: u.v is 3, 1, -1, -3.
    #   If u.v = 3: 3 > 2. INVALID.
    # If overlap is 4: u.v is 4, 2, 0, -2, -4.
    #   If u.v = 4: INVALID.
    #   If u.v = 2: Valid.
    
    # So, conflict exists if |I \cap J| >= 3 AND the signs align such that u.v > 2.
    # Specifically:
    # Overlap 3: Invalid only if u.v = 3 (all 3 overlapping signs are same).
    # Overlap 4: Invalid if u.v = 4 (all 4 same) or u.v = 2 (3 same, 1 diff? No. 3*1 + 1*(-1) = 2. 2<=2 Valid. Wait.
    # Let's re-calc u.v for overlap 4.
    # Signs on 4 positions.
    # 4 matches: 4 or -4. (4>2 Invalid, -4 Valid)
    # 3 matches, 1 mismatch: 3*1 + 1*(-1) = 2. (2<=2 Valid). Or 3*(-1)+1*(1) = -2. Valid.
    # 2 matches, 2 mismatch: 0. Valid.
    # So for Overlap 4, only all-same-sign is invalid.
    
    # So the conflict graph is quite sparse!
    # Number of weight-4 vectors: C(11,4) * 2^4 = 330 * 16 = 5280.
    # This is a very manageable size for bitset operations.
    
    # Generate all weight-4 vectors
    pool = []
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool.append(v)
            
    pool = np.array(pool, dtype=np.int8) # int8 is enough for -1, 0, 1
    n_pool = len(pool)
    
    # Precompute dot products?
    # 5280 x 5280 matrix of dots.
    # 5280^2 ~ 27.8 million entries.
    # int8 dot product fits in int16? Max dot is 4. Min -4.
    # We can use numpy to compute the full Gram matrix.
    # Gram = pool @ pool.T
    # This takes some time but is a one-time cost.
    # 5280x11 @ 11x5280 -> 5280x5280.
    # Should be fast (< 1s).
    
    if time.time() > deadline - 10:
        return pool[:10] # Fallback
        
    Gram = pool @ pool.T
    
    # Conflict condition: u.v > 2
    # Note: u.v is integer.
    # So conflict if Gram[i,j] > 2.
    # We want Max Independent Set in graph G where edge (i,j) exists if Gram[i,j] > 2.
    
    # Represent conflicts as bitsets.
    # n_pool = 5280.
    # 5280 / 64 = 82.5 words.
    # We can use Python integers as bitsets (arbitrary precision).
    # conflict_bits[i] is an integer where bit j is 1 if i and j conflict.
    
    conflict_bits = [0] * n_pool
    
    # Efficiently build conflict_bits
    # Gram is symmetric.
    # For each i, we need to find j where Gram[i,j] > 2.
    # We can vectorize the "find indices" part, but converting to bitset requires care.
    # Alternatively, since the graph is sparse, we can iterate.
    # But 5280^2 / 2 = 14M checks. In Python loop, this is too slow (~10s+).
    # Let's use NumPy to find indices and then build bitsets.
    
    # Actually, we can just store the conflict sets as sets of integers for the initial phase,
    # or use a list of lists.
    # Given n=5280, a list of lists of neighbors might be memory efficient and fast enough for ILS.
    
    # Let's try to build adjacency lists.
    # adj[i] = set of j such that Gram[i,j] > 2
    
    # Optimization:
    # Gram > 2 is a boolean mask.
    # We can process row by row.
    
    adj = [set() for _ in range(n_pool)]
    
    # This loop might be slow in pure Python if done naively.
    # Let's use np.nonzero on the upper triangle?
    # Or just loop i and use np.where(Gram[i] > 2).
    
    for i in range(n_pool):
        if time.time() > deadline - 15:
            break
        # Find neighbors
        # Gram[i] is a row.
        # We want j != i where Gram[i,j] > 2.
        # Since Gram is symmetric, we can just check all j.
        # np.where is fast.
        neighbors = np.where(Gram[i] > 2)[0]
        # Exclude self? Gram[i,i] = 4. 4 > 2. So self is in neighbors.
        # We don't want self in conflict set for IS logic usually, or we handle it.
        # For IS, we just don't pick i if i is in current set.
        # The conflict set for "can I add i?" is neighbors of i.
        # If i is in S, we can't add i.
        # If j in S and j in neighbors(i), we can't add i.
        
        # Remove i from neighbors
        neighbors = neighbors[neighbors != i]
        adj[i] = set(neighbors.tolist())
        
    # Now we have adj[i] as a set of conflicting indices.
    # We want to find a large Independent Set.
    # Algorithm: Iterative Local Search (ILS)
    # 1. Randomly initialize a set S (e.g., greedy).
    # 2. Repeat:
    #    a. Pick a random vector v not in S.
    #    b. If v conflicts with any in S, remove one of the conflicting vectors from S (randomly chosen from conflicts).
    #    c. Add v to S.
    #    d. If size(S) > best_size, update best.
    
    # To make this efficient, we need fast "remove from S" and "check conflict".
    # Using a boolean array `in_set` of size n_pool.
    # And a list/set `current_set` of indices.
    
    best_size = 0
    best_set_indices = []
    
    # Greedy Initialization
    # Random order, add if no conflict.
    indices_list = list(range(n_pool))
    random.shuffle(indices_list)
    
    current_set = set()
    in_set = [False] * n_pool
    
    for i in indices_list:
        # Check if i conflicts with any in current_set
        # Iterate through current_set? No, iterate through adj[i] and check in_set.
        # adj[i] is a set.
        conflict_found = False
        for j in adj[i]:
            if in_set[j]:
                conflict_found = True
                break
        if not conflict_found:
            current_set.add(i)
            in_set[i] = True
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set_indices = list(current_set)
        
    # ILS Loop
    # Budget: ~130 seconds.
    # Each step involves:
    # 1. Pick random candidate v.
    # 2. Find conflicts C = {j in S | j in adj[v]}
    # 3. If C empty, add v.
    # 4. Else, pick random c in C, remove c from S, add v.
    
    # To speed up "Find conflicts", we can maintain a list of current_set.
    # But checking `j in adj[v]` for all j in S is O(|S|).
    # |S| is likely ~100-200.
    # 130s / (200 ops) = 650k steps.
    # 650k steps is decent.
    
    # Optimization:
    # Instead of picking random v from all n_pool, pick from "promising" candidates?
    # Or just random.
    
    # Let's try to optimize the conflict check.
    # We can maintain a list `active_indices` of the current set.
    
    current_list = list(current_set)
    
    # We need to map index -> position in list for O(1) removal?
    # Or just use a set for membership and a list for iteration.
    # Removal from list is O(N). N is small.
    
    steps = 0
    last_improve = start_time
    
    while time.time() < deadline - 2:
        steps += 1
        
        # Pick random candidate
        # To avoid re-picking recently failed candidates, we could use a queue, but random is simple.
        v = random.randint(0, n_pool - 1)
        
        if in_set[v]:
            continue
            
        # Find conflicts
        conflicts = []
        # Iterate through current_set
        # Using list is faster for iteration than set
        for j in current_list:
            if j in adj[v]:
                conflicts.append(j)
                
        if not conflicts:
            # Add v
            current_list.append(v)
            current_set.add(v)
            in_set[v] = True
            
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_set_indices = list(current_list)
                last_improve = time.time()
        else:
            # Remove a random conflict
            # This is the "kick"
            c = random.choice(conflicts)
            
            # Remove c from current_list
            # Find index of c in current_list
            # Since N is small, linear search is fine.
            try:
                idx_c = current_list.index(c)
                current_list[idx_c] = current_list[-1]
                current_list.pop()
            except ValueError:
                pass # Should not happen
            
            current_set.discard(c)
            in_set[c] = False
            
            # Add v
            current_list.append(v)
            current_set.add(v)
            in_set[v] = True
            
            # Note: We don't update best here immediately unless size increased?
            # Size stayed same (removed 1, added 1).
            # But structure changed.
            
        # Periodic restart if stuck?
        # If no improvement for 5 seconds, restart with random greedy.
        if time.time() - last_improve > 5.0 and steps > 1000:
            # Restart
            current_set.clear()
            in_set = [False] * n_pool
            current_list = []
            
            indices_list = list(range(n_pool))
            random.shuffle(indices_list)
            
            for i in indices_list:
                conflict_found = False
                for j in adj[i]:
                    if in_set[j]:
                        conflict_found = True
                        break
                if not conflict_found:
                    current_set.add(i)
                    in_set[i] = True
                    current_list.append(i)
                    
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_set_indices = list(current_list)
                last_improve = time.time()
                
            steps = 0

    # Reconstruct result
    if not best_set_indices:
        # Fallback: return first few
        return pool[:10]
        
    result = pool[best_set_indices]
    return result
# EVOLVE-BLOCK-END
