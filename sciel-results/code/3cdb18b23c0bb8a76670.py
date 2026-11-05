"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import random

def construct():
    n = 128
    target_low = 128
    target_high = 256
    size = n + 1  # 129
    
    # Precompute indices for fast count updates
    # For each element x, which sums s does it contribute to?
    # x contributes to sum s if s-x is in A.
    # We will maintain a boolean array `in_A` and an integer array `counts`.
    
    def objective(x):
        # x is a numpy array of shape (129,)
        b = (x > 0.5).astype(np.int8)
        
        # Convolution for ordered pairs
        c = np.convolve(b, b)
        
        # Extract relevant sums 128 to 256
        c_relevant = c[128:257]
        
        # Compute d_s: 1 if s is even and b[s//2] == 1, else 0
        s_vals = np.arange(128, 257)
        is_even = (s_vals % 2 == 0)
        mid_indices = s_vals // 2
        d_vals = np.zeros_like(c_relevant, dtype=np.int8)
        d_vals[is_even] = b[mid_indices[is_even]]
        
        # Count unordered pairs
        counts = (c_relevant + d_vals) / 2.0
        
        # Check coverage
        uncovered = np.sum(counts == 0)
        
        if uncovered > 0:
            return 1000.0 + uncovered * 100.0 # Stronger penalty
            
        max_count = np.max(counts)
        return max_count

    start_time = time.time()
    
    # Initial guess: A = [64, 128]
    init_b = np.zeros(129, dtype=np.int8)
    init_b[64:129] = 1
    
    try:
        res = differential_evolution(
            objective, 
            bounds=[(0, 1)] * 129, 
            popsize=30, 
            maxiter=200, 
            tol=1e-4, 
            seed=42,
            polish=False,
            init='latinhypercube'
        )
    except Exception:
        res = None

    best_x = res.x if res is not None else init_b.astype(float)
    
    # Refine with local search
    current_A = set(np.where(best_x > 0.5)[0].tolist())
    
    # Ensure validity
    def is_valid(A_set):
        for s in range(target_low, target_high + 1):
            found = False
            a_min = max(0, s - 128)
            a_max = min(128, s // 2)
            for a in range(a_min, a_max + 1):
                if a in A_set and (s - a) in A_set:
                    found = True
                    break
            if not found:
                return False
        return True

    if not is_valid(current_A):
        current_A = set(range(64, 129))

    # Initialize counts vector
    # counts[s] for s in 0..256
    counts = np.zeros(257, dtype=np.int32)
    in_A = np.zeros(129, dtype=bool)
    
    def update_counts_from_set(A_set):
        counts[:] = 0
        in_A[:] = False
        for a in A_set:
            in_A[a] = True
        # Calculate counts
        # For each a in A, for each b in A with a <= b:
        # sum = a + b
        # count[sum] += 1
        A_list = list(A_set)
        for i, a in enumerate(A_list):
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if s <= 256:
                    counts[s] += 1
        return counts, in_A

    counts, in_A = update_counts_from_set(current_A)
    
    current_max = np.max(counts[target_low:target_high+1])
    best_max = current_max
    best_A = set(current_A)
    
    time_limit = 55
    
    # Local Search
    # Strategy: Try to remove elements. If a removal keeps coverage and reduces max, do it.
    # If no removal works, try swaps.
    
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # 1. Try Removals
        # Iterate through elements in A
        A_list = list(current_A)
        random.shuffle(A_list) # Random order to avoid bias
        
        for x in A_list:
            if x not in current_A: continue
            
            # Simulate removal of x
            # x contributes to sum s if in_A[s-x] is True
            # We need to check if any sum s becomes 0.
            
            # Identify sums affected by x
            # s ranges from x+0 to x+128
            # But we only care about 128..256 for coverage, and all for max?
            # Max is over 128..256.
            # However, removing x affects counts for s where s-x in A.
            
            affected_sums = []
            min_s = max(0, x) # x+0
            max_s = min(256, x + 128)
            
            # We need to know if count[s] becomes 0.
            # count[s] decreases by 1 if in_A[s-x] is True.
            
            # Check coverage for s in 128..256
            # If count[s] == 1 and in_A[s-x] is True, then removing x makes count[s] 0 -> Invalid.
            
            valid = True
            new_max = current_max
            
            # We can't easily compute new_max without updating all affected counts.
            # Let's just update a temporary copy or track changes.
            
            # Optimization: 
            # 1. Check if any critical sum (count==1) loses its only representation.
            # 2. Check if max decreases.
            
            # Let's do the update on a copy of counts? 
            # Copying 257 ints is cheap.
            temp_counts = counts.copy()
            
            for s in range(128, 257):
                y = s - x
                if 0 <= y <= 128 and in_A[y]:
                    temp_counts[s] -= 1
                    if temp_counts[s] == 0:
                        valid = False
                        break
            if not valid:
                continue
            
            # If valid, check new max
            # We only changed counts for s where in_A[s-x] is True.
            # The new max is max( max(counts[s] for s not affected), max(temp_counts[s] for s affected) )
            # Actually, just taking max of temp_counts[128:257] is O(129).
            new_max_val = np.max(temp_counts[128:257])
            
            if new_max_val < current_max:
                # Accept
                counts = temp_counts
                in_A[x] = False
                current_A.remove(x)
                current_max = new_max_val
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_A = set(current_A)
                break # Restart loop with new state
            
        if improved:
            continue
            
        # 2. Try Swaps
        # If no removal improved, try swapping x in A with y not in A.
        # This is O(|A| * (129-|A|)) which can be large.
        # We sample random swaps.
        
        if len(current_A) > 0:
            x = random.choice(list(current_A))
            candidates = [i for i in range(129) if i not in current_A]
            if candidates:
                y = random.choice(candidates)
                
                # Simulate swap: remove x, add y
                temp_counts = counts.copy()
                
                # Remove x
                for s in range(128, 257):
                    z = s - x
                    if 0 <= z <= 128 and in_A[z]:
                        temp_counts[s] -= 1
                        if temp_counts[s] == 0:
                            # Can't remove x yet? Or does adding y fix it?
                            # We must check coverage after adding y.
                            pass # Don't break immediately, add y might fix it.
                
                # Add y
                for s in range(128, 257):
                    z = s - y
                    if 0 <= z <= 128 and in_A[z]:
                        temp_counts[s] += 1
                
                # Check validity
                valid = True
                for s in range(128, 257):
                    if temp_counts[s] == 0:
                        valid = False
                        break
                
                if valid:
                    new_max_val = np.max(temp_counts[128:257])
                    if new_max_val < current_max:
                        # Accept
                        counts = temp_counts
                        in_A[x] = False
                        in_A[y] = True
                        current_A.remove(x)
                        current_A.add(y)
                        current_max = new_max_val
                        improved = True
                        if current_max < best_max:
                            best_max = current_max
                            best_A = set(current_A)
                        break

    return list(best_A)
# EVOLVE-BLOCK-END
