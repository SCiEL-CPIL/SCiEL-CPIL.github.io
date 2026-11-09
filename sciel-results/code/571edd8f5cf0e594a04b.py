import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for return/eval overhead
    
    N = 16
    E = N * (N - 1) // 2  # 120
    
    # 1. Precompute structure: K4s and their edge indices
    def edge_idx(u, v):
        if u > v:
            u, v = v, u
        return u * (2 * N - 1 - u) // 2 + (v - u - 1)

    k4s = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    idxs = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
                    ]
                    k4s.append(idxs)
    
    num_k4s = len(k4s)
    # K4s is a list of lists of 6 ints. 
    # Convert to a 2D numpy array for vectorization
    # k4s_np shape: (1820, 6)
    k4s_np = np.array(k4s, dtype=np.int32)
    
    # 2. Vectorized Objective Function
    # We want to minimize the number of colors.
    # Objective: Sum over all K4s of max(0, 5 - distinct_colors)
    # This is 0 if valid.
    
    def compute_objective(colors_np, num_colors):
        # colors_np: shape (120,)
        # We need distinct colors for each of the 1820 K4s.
        # For each K4 (6 edges), count distinct colors.
        # Vectorized distinct count for 6 elements:
        # d = 6 - sum(pairs that are equal)
        # Pairs in a K4: (0,1), (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)
        
        # Extract the 6 colors for each K4
        # c0 = colors_np[k4s_np[:, 0]]
        # ...
        # This creates 6 arrays of shape (1820,)
        
        c0 = colors_np[k4s_np[:, 0]]
        c1 = colors_np[k4s_np[:, 1]]
        c2 = colors_np[k4s_np[:, 2]]
        c3 = colors_np[k4s_np[:, 3]]
        c4 = colors_np[k4s_np[:, 4]]
        c5 = colors_np[k4s_np[:, 5]]
        
        # Count equalities
        # We can sum the boolean arrays
        # eq = (c0==c1) + (c0==c2) + ...
        # distinct = 6 - eq
        # penalty = np.maximum(0, 5 - distinct)
        
        # To save memory/time, we can accumulate the sum of equalities
        # Note: (c0==c1) is a boolean array. Summing them gives the number of equal pairs.
        # If all 6 are same, sum=15, distinct=1, penalty=4.
        # If 5 distinct, sum=1, distinct=5, penalty=0.
        # If 4 distinct (e.g. 2,1,1,1,1,1), sum=2? No.
        # Let's check: 2,1,1,1,1,1.
        # Pairs:
        # 2!=1 (5 pairs)
        # 1==1 (4 pairs among the 1s? No, 5 ones. C(5,2)=10 pairs. Plus 2!=1 is 5. Total 15 pairs? No, total pairs is 15.
        # If 2,1,1,1,1,1:
        # Pairs involving 2: 5 (all unequal)
        # Pairs involving 1s: C(5,2) = 10 (all equal)
        # Total equal pairs = 10.
        # Distinct = 6 - 10? No.
        # The formula d = 6 - sum(equal pairs) works ONLY if we count the number of pairs (i,j) with i<j such that c_i == c_j.
        # Yes, that is the standard identity.
        # So sum_equal = 10. d = 6 - 10 = -4? No.
        # Wait. The number of distinct elements in a multiset of size 6 is NOT 6 - (number of equal pairs).
        # Example: 1,1,1,1,1,1. Pairs=15. Distinct=1. 6-15 = -9. Wrong.
        # The formula is: Distinct = 6 - (sum over colors k of (count_k - 1))? No.
        # Distinct = 6 - (sum over pairs (i,j) [c_i == c_j]) is INCORRECT for >2 duplicates.
        
        # Correct vectorized distinct count for small N (6):
        # We can use a "one-hot" or sorting approach, or just explicit pair counting if we realize the flaw.
        # Actually, the number of distinct elements is:
        # 6 - (number of pairs (i,j) with i<j and c_i == c_j) is WRONG.
        # Let's re-verify.
        # 1,1,1. Pairs: (1,1), (1,1), (1,1) -> 3 pairs. Distinct=1. 3-3=0.
        # 1,1,2. Pairs: (1,1) eq, (1,2) neq, (1,2) neq. Sum=1. Distinct=2. 3-1=2. Correct.
        # 1,1,1,1. Pairs: 6. Distinct=1. 4-6=-2. Wrong.
        
        # So we cannot just sum pairs.
        # We need a different vectorized distinct count.
        # Option 1: Sorting.
        # Option 2: Use the fact that colors are small (0..14).
        # For each K4, we can check if a color is present.
        # But that's 15 checks per K4. 1820 * 15 = 27300 ops.
        
        # Alternative:
        # Distinct = 6 - (sum_{c} max(0, count_c - 1))
        # This is equivalent to: 6 - (sum_{c} count_c) + (number of distinct c with count_c > 0)
        # 6 - 6 + distinct = distinct.
        # So Distinct = number of distinct colors.
        
        # How to compute "number of distinct colors" vectorized?
        # We can use a trick with bitsets if colors fit in an int.
        # Or simply:
        # For each K4, sort the 6 colors. Then count changes.
        # np.sort on a (1820, 6) array is fast.
        
        sorted_k4s = np.sort(np.vstack([c0, c1, c2, c3, c4, c5]), axis=1)
        
        # Count distinct: 1 + number of i where sorted[i] != sorted[i+1]
        diffs = np.diff(sorted_k4s, axis=1)
        distinct_counts = 1 + np.sum(diffs != 0, axis=1)
        
        # Penalty: max(0, 5 - distinct_counts)
        penalty = np.maximum(0, 5 - distinct_counts)
        
        return np.sum(penalty)

    def annealing_search(num_colors, time_budget):
        # Initialize
        # Start with a random assignment
        colors = np.random.randint(0, num_colors, size=E, dtype=np.int32)
        
        # Initial objective
        current_obj = compute_objective(colors, num_colors)
        
        if current_obj == 0:
            return colors.tolist()
            
        best_colors = colors.copy()
        best_obj = current_obj
        
        # Annealing parameters
        # T starts high, decays
        T_start = 5.0  # Penalty is at most 4 per K4. Max total ~ 1820*4 = 7280.
        # Actually, if we use sum of penalties, T should be comparable to typical delta.
        # Typical delta is small.
        T = T_start
        T_end = 0.01
        alpha = 0.999  # Decay rate
        
        start = time.time()
        
        while time.time() - start < time_budget:
            # Pick a random edge
            e_idx = np.random.randint(0, E)
            old_c = colors[e_idx]
            
            # Pick a new color
            # Randomly choose from 0..num_colors-1
            new_c = np.random.randint(0, num_colors)
            
            if new_c == old_c:
                continue
            
            # Calculate new objective
            # We can do this incrementally or recompute.
            # Recomputing is ~ 1820 * 6 ops + sort.
            # Numpy sort on (1820, 6) is very fast (C optimized).
            # Let's estimate: 1820*6 = 10920 elements. Sort is O(N log N).
            # This might be ~ 10-50 microseconds.
            # 100k iterations * 50us = 5 seconds.
            # This is feasible within 115s.
            
            colors[e_idx] = new_c
            new_obj = compute_objective(colors, num_colors)
            
            # Acceptance probability
            delta = new_obj - current_obj
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_colors = colors.copy()
                    if best_obj == 0:
                        return best_colors.tolist()
            else:
                # Reject
                colors[e_idx] = old_c
                
            T *= alpha
            if T < T_end:
                T = T_end
                
        return None

    # Strategy:
    # Try 13, 14, 15 colors.
    # Allocate time to each.
    # If 13 is found, return it.
    # Else if 14, return it.
    # Else if 15, return it.
    # Else fallback to 17.
    
    # Time allocation:
    # Total 115s.
    # Try 13 for 30s.
    # Try 14 for 30s.
    # Try 15 for 40s.
    
    # Note: Random seed should be fixed for reproducibility?
    # The task allows standard library.
    # Let's use a fixed seed for the "best" case, but maybe vary it?
    # For now, fixed seed 42.
    
    np.random.seed(42)
    random.seed(42)
    
    # Attempt 1: 13 colors
    sol13 = annealing_search(13, 30)
    if sol13 is not None:
        return sol13
        
    # Attempt 2: 14 colors
    sol14 = annealing_search(14, 30)
    if sol14 is not None:
        return sol14
        
    # Attempt 3: 15 colors
    sol15 = annealing_search(15, 40)
    if sol15 is not None:
        return sol15
        
    # Fallback: Seed (17 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
