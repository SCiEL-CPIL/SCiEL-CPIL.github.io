import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Strategy 1: Try to find a valid coloring for n=26 (known possible)
    # Then try to extend to n=27 (unknown/likely impossible but worth trying)
    
    # We use a robust Simulated Annealing approach.
    # It is crucial to handle the case where no solution is found.
    
    # First, try n=26.
    # If successful, we have a valid n=26.
    # Then try n=27. If successful, return n=27.
    # If n=27 fails, return the n=26 solution.
    # If n=26 fails, try n=25, etc.
    # If all fail, return a hardcoded valid small solution.

    result_26 = None
    if time.time() < deadline - 20:
        result_26 = robust_sa(26, time_limit=40.0)
        
    if result_26 is not None:
        # Try to extend to 27? 
        # Extending is hard. Instead, run SA for 27 from scratch with less time.
        if time.time() < deadline - 10:
            result_27 = robust_sa(27, time_limit=15.0)
            if result_27 is not None:
                return result_27
        return result_26

    # Fallbacks for smaller n
    for n in [25, 24, 23, 22, 21, 20, 15, 10]:
        if time.time() > deadline - 5:
            break
        res = robust_sa(n, time_limit=5.0)
        if res is not None:
            return res

    # Absolute Fallback: Hardcoded valid coloring for n=10
    # A simple cyclic coloring or random check.
    # For n=10, a valid 5-coloring is easy to find.
    # Let's generate one greedily or just return a known valid one.
    # Actually, random restart SA for n=10 is instant.
    res = robust_sa(10, time_limit=1.0)
    if res is not None:
        return res
        
    # Last resort: Return a valid n=1 coloring (trivial)
    # K_1 has no edges, so no C4.
    return {'n': 1, 'colors': []}

def robust_sa(n, time_limit=40.0):
    """
    Simulated Annealing to find a C4-free 5-coloring of K_n.
    Returns dict with 'n' and 'colors' or None.
    """
    num_edges = n * (n - 1) // 2
    if num_edges == 0:
        return {'n': n, 'colors': []}
        
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initialize randomly
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Bitmask adjacency for each color
    # adj[c][u] is a bitmask of neighbors of u in color c
    adj = [[0] * n for _ in range(5)]
    
    def build_adj():
        for c in range(5):
            for u in range(n):
                adj[c][u] = 0
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)

    build_adj()

    # Calculate initial violations
    # Violation for a color c is the number of pairs (u, v) with >= 2 common neighbors.
    # Sum over all colors.
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                # Only check v > u to avoid double counting
                # But we need to iterate all pairs.
                # Optimization: 
                # For each u, for each v > u, count common neighbors.
                for v in range(u + 1, n):
                    common = adj[c][u] & adj[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    current_violations = count_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}

    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling = 0.9995
    min_temp = 0.01
    
    while time.time() - start_time < time_limit:
        if current_violations == 0:
            return {'n': n, 'colors': colors}
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Pick a random edge to change
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_c = colors[e_idx]
        
        # Try changing to a different color
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
            
        # Calculate delta in violations
        # Removing edge (i,j) from old_c
        # Adding edge (i,j) to new_c
        
        delta = 0
        
        # Helper to calculate change in violations for a specific color when toggling edge (i,j)
        def get_delta(c, adding):
            # adding=True means we are adding edge (i,j) to color c
            # adding=False means we are removing edge (i,j) from color c
            
            # The change in violations only affects pairs (u, v) where u or v is i or j.
            # Specifically, pairs (i, v) and (j, v) for v != i, j.
            # And the pair (i, j) itself?
            # The violation count for pair (i, j) is C(|N_c(i) ∩ N_c(j)|, 2).
            # N_c(i) and N_c(j) do not include each other in the intersection calculation 
            # because the edge (i,j) is not considered a "common neighbor".
            # Wait, the definition of common neighbors of i and j is {k | (i,k) in c and (j,k) in c}.
            # The edge (i,j) itself does not contribute to the common neighbors of i and j.
            # So changing the color of (i,j) does NOT change the violation count for the pair (i,j).
            # It ONLY changes the violation counts for pairs (i, v) and (j, v).
            
            # For a fixed color c, let S_i = N_c(i) and S_j = N_c(j).
            # If we add (i,j) to c:
            #   j is added to S_i. i is added to S_j.
            #   For any v != i, j:
            #     The common neighbors of (i, v) change.
            #     Old: S_i ∩ S_v. New: (S_i ∪ {j}) ∩ S_v = (S_i ∩ S_v) ∪ ({j} ∩ S_v).
            #     So if j is in S_v (i.e., (j,v) is c), the count increases by 1.
            #     Similarly for (j, v), if i is in S_v (i.e., (i,v) is c), count increases by 1.
            
            #     Let k_old = |S_i ∩ S_v|.
            #     If j in S_v: k_new = k_old + 1. Delta = C(k_old+1, 2) - C(k_old, 2) = k_old.
            #     If j not in S_v: k_new = k_old. Delta = 0.
            
            #     So, delta_add = sum_{v in S_v where (j,v) is c} |S_i ∩ S_v|  +  sum_{v in S_v where (i,v) is c} |S_j ∩ S_v|
            #     Note: S_i does not contain j yet. S_v is N_c(v).
            
            # If we remove (i,j) from c:
            #   j is removed from S_i. i is removed from S_j.
            #   For any v != i, j:
            #     If j in S_v: k_old = |S_i ∩ S_v| (which includes j? No, j is in S_i, is j in S_v? Yes).
            #     Wait, S_i contains j. S_v contains j. So j is in the intersection.
            #     k_old includes j. k_new = k_old - 1.
            #     Delta = C(k_old-1, 2) - C(k_old, 2) = -(k_old - 1).
            
            # Let's implement this carefully.
            
            d = 0
            mask_i = adj[c][i]
            mask_j = adj[c][j]
            
            if adding:
                # We are adding (i,j) to c.
                # Update pairs (i, v) for v in N_c(j) (excluding i)
                # For each v in N_c(j), if (j,v) is c, then j becomes a common neighbor of i and v.
                # The number of common neighbors increases by 1.
                # The increase in violation count is k_old, where k_old = |N_c(i) ∩ N_c(v)|.
                temp = mask_j & ~(1 << i)
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    k_old = (mask_i & adj[c][v]).bit_count()
                    d += k_old
                
                # Update pairs (j, v) for v in N_c(i) (excluding j)
                temp = mask_i & ~(1 << j)
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    k_old = (mask_j & adj[c][v]).bit_count()
                    d += k_old
                    
            else:
                # We are removing (i,j) from c.
                # j is in N_c(i). i is in N_c(j).
                # For each v in N_c(j) (excluding i):
                #   j is a common neighbor of i and v.
                #   k_old = |N_c(i) ∩ N_c(v)|. Since j is in both, k_old >= 1.
                #   k_new = k_old - 1.
                #   Delta = C(k_old-1, 2) - C(k_old, 2) = -(k_old - 1).
                temp = mask_j & ~(1 << i)
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    k_old = (mask_i & adj[c][v]).bit_count()
                    d -= (k_old - 1)
                
                # For each v in N_c(i) (excluding j):
                temp = mask_i & ~(1 << j)
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    k_old = (mask_j & adj[c][v]).bit_count()
                    d -= (k_old - 1)
                    
            return d

        delta_remove = get_delta(old_c, False)
        delta_add = get_delta(new_c, True)
        total_delta = delta_remove + delta_add
        
        # Metropolis acceptance
        if total_delta < 0 or (temperature > min_temp and random.random() < math.exp(-total_delta / temperature)):
            # Apply change
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            colors[e_idx] = new_c
            current_violations += total_delta
            
        temperature *= cooling

    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    return None
