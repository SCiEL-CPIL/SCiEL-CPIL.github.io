import time
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    # 1. Precompute 3-AP neighbor lists
    # ap_neighbors[i] is a list of (j, k) such that {i, j, k} is a 3-AP.
    # We store this as a list of arrays for faster iteration if needed, 
    # but for counting, we just need to check if both are in S.
    ap_neighbors = [[] for _ in range(n)]
    for i in range(n):
        # i is 1st: i, i+d, i+2d
        for d in range(1, (n - 1 - i) // 2 + 1):
            ap_neighbors[i].append((i + d, i + 2 * d))
        # i is 2nd: i-d, i, i+d
        for d in range(1, min(i, n - 1 - i) + 1):
            ap_neighbors[i].append((i - d, i + d))
        # i is 3rd: i-2d, i-d, i
        for d in range(1, i // 2 + 1):
            ap_neighbors[i].append((i - 2 * d, i - d))
            
    # 2. Precompute 4-AP conflict lists
    # conflict[i] is a list of pairs (j, k) such that {i, j, k} + i forms a 4-AP? 
    # No, conflict[i] should be: if i is added, it creates a 4-AP if any of the 3 other elements are in S.
    # Let's define conflict_mask[i] as a set of pairs (j, k) such that {i, j, k} is a subset of a 4-AP.
    # Actually, it's easier to check: for each i, list all (j, k) such that adding i to a set containing j and k creates a 4-AP.
    # The 4-AP is {a, a+d, a+2d, a+3d}.
    # If i is one of them, the other three must be in S.
    # So for each i, we list all sets of 3 elements {u, v, w} such that {i, u, v, w} is a 4-AP.
    # This is equivalent to: for each d, check the 4 positions relative to i.
    
    # Let's precompute for each i, a list of "required pairs" (j, k).
    # If S contains j and k, and we add i, does it form a 4-AP?
    # It forms a 4-AP if there exists a d such that {i, j, k} are 3 of the 4 terms.
    # This is getting complex to precompute as simple pairs.
    # Alternative: Precompute for each i, a list of "triples" (j, k, l) is too big.
    
    # Better approach for conflict check:
    # When considering adding i to S, we check if i completes any 4-AP.
    # A 4-AP is {x, x+d, x+2d, x+3d}.
    # i can be at position 1, 2, 3, or 4.
    # If i is pos 1: need i+d, i+2d, i+3d in S.
    # If i is pos 2: need i-d, i+d, i+2d in S.
    # If i is pos 3: need i-2d, i-d, i+d in S.
    # If i is pos 4: need i-3d, i-2d, i-d in S.
    
    # We can precompute for each i, a list of masks (bitmasks) of the other 3 elements.
    # But with n=256, we can use a 256-bit integer or just a set of tuples.
    # Let's use a list of tuples of 3 indices for each i.
    
    conflict_triples = [[] for _ in range(n)]
    for i in range(n):
        # i is 1st
        for d in range(1, (n - 1 - i) // 3 + 1):
            conflict_triples[i].append((i + d, i + 2 * d, i + 3 * d))
        # i is 2nd
        for d in range(1, min(i, n - 1 - i) // 2 + 1):
            # Need i-d >= 0 and i+2d < n
            if i - d >= 0 and i + 2 * d < n:
                conflict_triples[i].append((i - d, i + d, i + 2 * d))
        # i is 3rd
        for d in range(1, min(i, n - 1 - i) // 1 + 1):
            if i - 2 * d >= 0 and i + d < n:
                conflict_triples[i].append((i - 2 * d, i - d, i + d))
        # i is 4th
        for d in range(1, i // 3 + 1):
            if i - 3 * d >= 0:
                conflict_triples[i].append((i - 3 * d, i - 2 * d, i - d))

    # Helper: Count 3-APs in S (boolean array)
    def count_3aps(S):
        count = 0
        for i in range(n):
            if S[i]:
                for j, k in ap_neighbors[i]:
                    if S[j] and S[k]:
                        count += 1
        return count // 1 # Each AP is counted 3 times? 
        # Wait. 
        # If {a, b, c} is an AP.
        # For i=a, (b, c) is in list. Count 1.
        # For i=b, (a, c) is in list. Count 1.
        # For i=c, (a, b) is in list. Count 1.
        # Total count = 3 * (number of distinct APs).
        # So return count // 3.
        
    def count_3aps_correct(S):
        count = 0
        for i in range(n):
            if S[i]:
                for j, k in ap_neighbors[i]:
                    if S[j] and S[k]:
                        count += 1
        return count // 3

    # Helper: Check if i can be added to S
    def can_add(S, i):
        for j, k, l in conflict_triples[i]:
            if S[j] and S[k] and S[l]:
                return False
        return True

    # Helper: Calculate delta for swapping u (out) and v (in)
    def calc_delta(S, u, v):
        # u is currently in S. v is not.
        # We want to remove u and add v.
        # First, check if v can be added to S \ {u}.
        # Temporarily set S[u] = False
        S[u] = False
        valid = True
        for j, k, l in conflict_triples[v]:
            if S[j] and S[k] and S[l]:
                valid = False
                break
        if not valid:
            S[u] = True
            return -1000000 # Invalid swap
            
        # Calculate gain from adding v to S (which is S_original \ {u})
        gain = 0
        for j, k in ap_neighbors[v]:
            if S[j] and S[k]:
                gain += 1
                
        # Calculate loss from removing u from S_original
        # We need to count APs involving u that are fully in S_original.
        # Note: S is currently S_original \ {u}.
        # So if we check S[j] and S[k], we are checking if j,k are in S_original \ {u}.
        # This is correct for APs that do not involve u.
        # But we are removing u, so we lose all APs that contain u.
        # An AP containing u is {u, j, k} where j,k in S_original.
        # Since S currently does not have u, S[j] and S[k] checks if j,k are in S_original.
        # So the count of APs lost is exactly the number of (j,k) in ap_neighbors[u] such that S[j] and S[k].
        
        loss = 0
        for j, k in ap_neighbors[u]:
            if S[j] and S[k]:
                loss += 1
                
        S[u] = True
        return gain - loss

    # Multi-start Local Search
    best_S = np.zeros(n, dtype=bool)
    best_count = 0
    
    num_restarts = 50
    steps_per_restart = 20000
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Initialize S
        # Strategy: Random subset with probability p, then repair?
        # Or Greedy: add elements in random order if possible.
        S = np.zeros(n, dtype=bool)
        
        # Greedy initialization
        order = np.random.permutation(n)
        for i in order:
            if can_add(S, i):
                S[i] = True
                
        current_count = count_3aps_correct(S)
        
        # Local Search
        improved = True
        step_count = 0
        while improved and step_count < steps_per_restart:
            if time.time() - start_time > time_limit - 5:
                break
            improved = False
            step_count += 1
            
            # Try random swaps
            # Pick random u in S
            in_S = np.where(S)[0]
            if len(in_S) == 0:
                break
            not_in_S = np.where(~S)[0]
            if len(not_in_S) == 0:
                break
                
            u = in_S[np.random.randint(len(in_S))]
            v = not_in_S[np.random.randint(len(not_in_S))]
            
            delta = calc_delta(S, u, v)
            
            if delta > 0:
                S[u] = False
                S[v] = True
                current_count += delta
                improved = True
                
                if current_count > best_count:
                    best_count = current_count
                    best_S = S.copy()
            elif delta == 0:
                # Accept neutral moves to escape plateaus?
                if np.random.random() < 0.1:
                    S[u] = False
                    S[v] = True
                    improved = True

    # Return best_S as sorted list of 1-based indices
    result_indices = np.where(best_S)[0] + 1
    return sorted(result_indices)

if __name__ == "__main__":
    res = construct()
    print(len(res))
