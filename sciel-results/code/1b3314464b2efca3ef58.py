import numpy as np
import time
import random

def construct():
    n = 192
    max_time = 55.0
    start_time = time.time()
    
    # Phase 1: Conflict-Based Deletion
    # Start with all numbers
    S = set(range(1, n + 1))
    
    # Precompute nothing, just check violations on the fly or incrementally?
    # With 192 elements, checking all violations is O(N^4) which is too slow if done naively.
    # N^4 = 192^4 = 1.3 * 10^9. Too slow.
    # We need to find violations efficiently.
    
    # A violation is (x, a, b, c) such that x | (a+b+c).
    # We can iterate over sums?
    # Sums of 3 distinct elements range from 1+2+3=6 to 190+191+192=573.
    # For each sum S_val, find all triples (a,b,c) that sum to S_val.
    # Then check if any x in S divides S_val.
    
    # This is feasible.
    # Max sum is 573.
    # We can precompute for each sum value, the list of triples?
    # Or just iterate x in S, and check if x divides any triple sum.
    
    # Let's optimize the violation detection.
    # For a given set S, we want to find if there is any x in S and a,b,c in S\{x} such that x | (a+b+c).
    
    # Algorithm for Conflict Detection:
    # 1. Compute all pair sums of S.
    # 2. For each x in S:
    #    a. Check if x divides sum of any 3 in S\{x}.
    #       This is equivalent to: Does there exist a,b,c in S\{x} such that a+b+c = k*x?
    #       Since max sum is ~573, k is small (1 to 573/x).
    #       We can check if any triple sum is a multiple of x.
    
    # Let's use the Numpy approach for violation detection, but batched.
    
    def find_conflicts(S_list):
        """
        Returns a list of conflicts. Each conflict is a tuple (x, a, b, c) 
        where x divides a+b+c.
        Also returns a dictionary counting conflicts per element.
        """
        S_arr = np.array(S_list)
        n_s = len(S_arr)
        if n_s < 4:
            return [], {}
        
        # We need to check for each x in S, if x divides sum of any 3 others.
        # This is O(N^4) if done naively.
        # N=192. 192^4 is too big.
        
        # Alternative:
        # For each x, we want to know if there are a,b,c in S\{x} such that a+b+c == k*x.
        # Possible values for k*x: x, 2x, ..., floor(573/x)*x.
        # For each target T = k*x, we need to check if T can be formed by 3 distinct elements from S\{x}.
        
        # To check if T can be formed by 3 distinct elements from S\{x}:
        # We can use a set of pair sums?
        # Pair sums of S\{x}.
        
        # This is still heavy to do for every x from scratch.
        
        # Let's try a different heuristic for Phase 1.
        # Just remove random elements involved in a conflict found by sampling.
        
        pass

    # Given the time limit and complexity, a simpler "New" approach is 
    # **Greedy with Lookahead**.
    
    # Let's stick to a robust **Local Search** but with a **Better Initialization** 
    # and **More Efficient Moves**.
    
    # The seed's weakness is likely that `is_valid_add` is called too often 
    # with large sets, and the random moves are inefficient.
    
    # New Strategy:
    # 1. Initialize with a large set using a fast heuristic (e.g., keep only numbers > 100).
    # 2. Use Simulated Annealing to optimize.
    
    # Let's implement Simulated Annealing.
    
    def is_valid_set(S_list):
        """Check if the whole set is valid. Returns True if valid."""
        S_arr = np.array(S_list)
        n_s = len(S_arr)
        if n_s < 4:
            return True
            
        # Check 1: x does not divide sum of any 3 in S\{x}
        # Vectorized check for all x?
        # For each x, compute triple sums of S\{x} and check mod x.
        
        # To speed up, we can precompute all triple sums of S?
        # Triple sums of S: S^3 size. 192^3 = 7M.
        # We need to exclude cases where x is in the triple.
        
        # Let's generate all triple sums with indices.
        # This is memory intensive: 192^3 * 3 indices.
        # 7M * 3 * 4 bytes = 84 MB. Acceptable.
        
        # But we only need to do this once per evaluation? No, set changes.
        
        # Let's go back to the seed's `is_valid_add` but optimize it.
        # The seed's check 2 is a loop over `a in S`.
        
        pass

    # Let's try a **Greedy Construction** that is **Deterministic** and **Fast**.
    # Order: Descending.
    # At each step, add x if valid.
    # This is what the seed does in initialization.
    # To improve, we can try **Multiple Descending Orders** (e.g., skip some numbers).
    
    # Let's try a **Backtracking Search** with a depth limit.
    
    best_size = 0
    best_set = set()
    
    # Helper for fast validity check of adding x to S_list
    def can_add(x, S_list):
        # S_list is a list of ints
        if not S_list:
            return True
        
        S_arr = np.array(S_list)
        
        # Check 1: x does not divide sum of any 3 in S_list
        if len(S_arr) >= 3:
            # Triple sums of S_list
            # Use itertools for small lists, numpy for larger?
            # Let's use a set of triple sums for faster lookup?
            # No, we need to check if ANY triple sum is divisible by x.
            
            # Efficient check:
            # Compute all triple sums.
            # If len(S) is large, this is slow.
            
            # Optimization:
            # If x > max_sum_possible, it can't divide any sum.
            max_sum = np.sum(np.sort(S_arr)[-3:])
            if x > max_sum:
                # Check 1 passes automatically
                pass
            else:
                # Check if any triple sum is divisible by x
                # Generate triple sums
                triple_sums = set()
                # This is O(N^3) in Python loop, which is slow.
                # Use numpy broadcasting?
                # S_arr[:, None, None] + S_arr[None, :, None] + S_arr[None, None, :]
                # Mask distinct indices.
                
                # For N=40, N^3=64k. Fast.
                # For N=100, N^3=1M. Slower.
                
                S_3d = S_arr[:, None, None] + S_arr[None, :, None] + S_arr[None, None, :]
                i_idx, j_idx, k_idx = np.indices((len(S_arr), len(S_arr), len(S_arr)))
                mask = (i_idx != j_idx) & (j_idx != k_idx) & (i_idx != k_idx)
                triple_sums_arr = S_3d[mask]
                
                if np.any(triple_sums_arr % x == 0):
                    return False
        
        # Check 2: For each a in S_list, a does not divide x + u + v
        # where u, v in S_list\{a}
        
        for a in S_arr:
            T = S_arr[S_arr != a]
            if len(T) < 2:
                continue
            
            # Check if x + u + v is divisible by a for any u, v in T, u!=v
            # u + v == k*a - x
            # Pair sums of T
            
            T_2d = T[:, None] + T[None, :]
            i_idx, j_idx = np.indices((len(T), len(T)))
            mask = (i_idx != j_idx)
            pair_sums = T_2d[mask]
            
            if np.any((pair_sums + x) % a == 0):
                return False
                
        return True

    # Phase 1: Greedy Initialization with multiple strategies
    strategies = [
        list(range(n, 0, -1)),
        list(range(1, n + 1)),
        [x for x in range(n, 0, -1) if x % 2 == 1], # Odd first
        [x for x in range(n, 0, -1) if x % 2 == 0], # Even first
    ]
    
    # Add random strategies
    for _ in range(5):
        random.shuffle(strategies[-1]) if len(strategies) > 4 else strategies.append(list(range(1, n+1)))
        random.shuffle(strategies[-1])
        
    for strategy in strategies:
        S = []
        for x in strategy:
            if can_add(x, S):
                S.append(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    # Phase 2: Local Search (Simulated Annealing)
    current_set = list(best_set)
    current_size = len(current_set)
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.99
    
    while time.time() - start_time < max_time and T > T_min:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            candidates = [x for x in range(1, n + 1) if x not in best_set]
            if candidates:
                x = random.choice(candidates)
                if can_add(x, current_set):
                    new_set = current_set + [x]
                    new_size = current_size + 1
                    # Accept if better or with probability
                    if new_size > current_size or random.random() < np.exp((new_size - current_size) / T):
                        current_set = new_set
                        current_size = new_size
                        best_set = set(current_set)
                        best_size = current_size
        elif move_type < 0.6:
            # Remove
            if current_set:
                x = random.choice(current_set)
                new_set = [y for y in current_set if y != x]
                new_size = current_size - 1
                if random.random() < np.exp((new_size - current_size) / T):
                    current_set = new_set
                    current_size = new_size
        else:
            # Swap
            if current_set:
                x = random.choice(current_set)
                candidates = [y for y in range(1, n + 1) if y not in best_set]
                if candidates:
                    y = random.choice(candidates)
                    # Check if y can replace x
                    # Remove x, add y
                    temp_set = [z for z in current_set if z != x]
                    if can_add(y, temp_set):
                        new_set = temp_set + [y]
                        new_size = current_size
                        # Accept if same size (explore)
                        current_set = new_set
                        # best_set unchanged size, but might be better? 
                        # We only update best_set if size increases.
                        
        T *= cooling_rate

    return list(best_set)
