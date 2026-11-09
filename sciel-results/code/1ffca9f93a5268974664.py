import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Target sums
    min_sum = 128
    max_sum = 256
    n = 128
    
    # R[S] stores the number of representations for sum S
    # We only care about S in [128, 256]
    # Index S in R corresponds to sum S
    R = np.zeros(max_sum + 1, dtype=np.int32)
    
    def get_cost(A_indices, R_curr):
        # A_indices: set of indices
        # R_curr: numpy array of multiplicities
        uncovered = np.sum(R_curr[min_sum:max_sum+1] == 0)
        max_mult = np.max(R_curr[min_sum:max_sum+1])
        # Penalty for uncovered sums
        # We want to minimize max_mult, but must have 0 uncovered
        # Cost = 1000 * uncovered + max_mult
        return 1000 * uncovered + max_mult

    def apply_move(v, x, R_curr, A_set):
        """
        Flip v[x]. Returns new R, new A_set, delta_cost?
        Actually, we compute cost from scratch or update incrementally.
        Given the small size, recomputing cost from R is O(129).
        Updating R is O(|A|).
        """
        if v[x]:
            # Remove x
            # Pairs involving x and y in A_set \ {x}
            # If x in A_set, we remove pairs (x, y) for y in A_set.
            # Note: pair (x, x) is only one pair.
            # If we remove x, we remove:
            # 1. Pair (x, x) if x was in A.
            # 2. Pairs (x, y) for y in A, y != x.
            
            # Let's just iterate over current A_set
            # But A_set includes x.
            # We need to subtract contributions of x with all other elements in A.
            # And x with itself.
            
            # Efficient way:
            # Current A contains x.
            # Sums contributed by x are x + y for y in A.
            # We subtract 1 from R[x+y] for all y in A.
            
            Ss = x + np.array(list(A_set))
            mask = (Ss >= min_sum) & (Ss <= max_sum)
            R_curr[Ss[mask]] -= 1
            
            A_set.remove(x)
            v[x] = False
        else:
            # Add x
            # Sums contributed by x are x + y for y in A (current A, which does not contain x yet)
            # Plus x + x?
            # If we add x, we form pairs (x, y) for all y in A.
            # And pair (x, x).
            # So we add 1 to R[x+y] for all y in A, AND add 1 to R[2x].
            
            if len(A_set) > 0:
                Ss = x + np.array(list(A_set))
                mask = (Ss >= min_sum) & (Ss <= max_sum)
                R_curr[Ss[mask]] += 1
            
            # Add pair (x, x)
            Sxx = 2 * x
            if min_sum <= Sxx <= max_sum:
                R_curr[Sxx] += 1
                
            A_set.add(x)
            v[x] = True

    # Initial Best
    best_cost = float('inf')
    best_A = set()
    
    # Try several random initializations
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random init size
        # Sizes 20-40 seem reasonable
        k = random.randint(20, 40)
        A_init = set(random.sample(range(n+1), k))
        
        v = np.zeros(n+1, dtype=bool)
        for x in A_init:
            v[x] = True
            
        R_curr = np.zeros(max_sum + 1, dtype=np.int32)
        A_set = set(A_init)
        
        # Initialize R_curr
        # O(k^2)
        A_list = list(A_set)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if min_sum <= s <= max_sum:
                    R_curr[s] += 1
        
        current_cost = get_cost(A_set, R_curr)
        
        # Simulated Annealing
        T = 50.0
        T_min = 0.1
        cooling = 0.99
        
        # Steps per restart
        max_steps = 50000
        steps_done = 0
        
        while steps_done < max_steps:
            if time.time() > deadline:
                break
                
            # Pick random x
            x = random.randint(0, n)
            
            # Calculate cost if we flip x
            # We need to simulate the flip to get new R, then cost, then revert?
            # Or calculate delta?
            # Given O(129) cost calc and O(k) R update, simulation is fine.
            
            old_R = R_curr.copy()
            old_A_set = set(A_set)
            old_v = v.copy()
            
            apply_move(v, x, R_curr, A_set)
            
            new_cost = get_cost(A_set, R_curr)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(A_set)
            else:
                # Revert
                R_curr[:] = old_R
                A_set = old_A_set
                v[:] = old_v
            
            T *= cooling
            if T < T_min:
                T = T_min
            steps_done += 1
            
        # If we broke out of while due to time, check outer loop
        if time.time() > deadline:
            break

    # Construct result from best_A
    # Ensure it's valid?
    # If best_cost has uncovered sums, it's invalid.
    # The task requires valid A.
    # If we didn't find a valid one, we should return a known valid one (e.g. seed)
    # But seed is valid.
    
    # Check if best_A is valid
    # We can re-verify best_A
    R_check = np.zeros(max_sum + 1, dtype=np.int32)
    A_list = list(best_A)
    for i in range(len(A_list)):
        for j in range(i, len(A_list)):
            s = A_list[i] + A_list[j]
            if min_sum <= s <= max_sum:
                R_check[s] += 1
    
    if np.any(R_check[min_sum:max_sum+1] == 0):
        # Invalid, fallback to seed
        return list(range(64, 129))
        
    return list(best_A)
