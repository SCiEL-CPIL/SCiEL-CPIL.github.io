import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin (evaluator kills at 90s)
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    target_len = max_sum - min_sum + 1  # 129
    
    # Precompute pair indices for fast count updates
    # For a set A of size k, the number of unordered pairs (with repetition) is k*(k+1)/2
    # We need to efficiently compute the sum counts
    
    def compute_counts(A_list):
        """Compute counts of all sums from pairs in A_list using vectorized approach."""
        k = len(A_list)
        if k == 0:
            return np.zeros(max_sum + 1, dtype=np.int32)
        
        A_arr = np.array(A_list, dtype=np.int32)
        # Use outer sum approach
        # For unordered pairs with repetition: (i,j) where i <= j
        # We can compute all sums and count them
        
        # Efficient: use np.add.outer for all pairs, then count
        # But that's k x k which is fine for k ~ 20-30
        sums = np.add.outer(A_arr, A_arr)
        
        # For unordered pairs: sum[i,j] = sum[j,i], so we need to be careful
        # The count for a sum s should be:
        #   number of unordered pairs {a,b} (a<=b) such that a+b=s
        # Using the full outer product, each unordered pair {a,b} with a!=b appears twice
        # and each {a,a} appears once.
        
        # Better approach: count from the full matrix
        # For sum s: total occurrences in sums matrix = 2*(# unordered pairs with a!=b) + 1*(# pairs with a==b)
        # We want: # unordered pairs = (# with a!=b) + (# with a==b)
        # Let f(s) = total in matrix, g(s) = number of a in A with 2a=s
        # Then # unordered pairs with a!=b contributing to s = (f(s) - g(s)) / 2
        # Total unordered pairs for s = (f(s) - g(s)) / 2 + g(s) = (f(s) + g(s)) / 2
        
        # Count occurrences of each sum in the full matrix
        # Use bincount for efficiency
        flat_sums = sums.ravel()
        # Only count sums in range [0, max_sum]
        mask = (flat_sums >= 0) & (flat_sums <= max_sum)
        flat_sums_valid = flat_sums[mask]
        
        if len(flat_sums_valid) == 0:
            return np.zeros(max_sum + 1, dtype=np.int32)
        
        f = np.bincount(flat_sums_valid, minlength=max_sum + 1).astype(np.int32)
        
        # g(s) = number of a in A such that 2a = s, i.e., a = s/2 and s is even and a in A
        # g is nonzero only for even s
        A_set = set(A_list)
        g = np.zeros(max_sum + 1, dtype=np.int32)
        for a in A_list:
            s = 2 * a
            if s <= max_sum:
                g[s] += 1
        
        # Total unordered pairs for each sum
        # (f + g) / 2
        counts = (f + g) // 2
        
        return counts
    
    def get_target_counts(A_list):
        """Get counts only for target sums [128, 256]."""
        counts = compute_counts(A_list)
        return counts[min_sum:max_sum + 1]
    
    def get_cost(A_list):
        """Compute the cost: max representation multiplicity in target range, with penalty for uncovered."""
        tc = get_target_counts(A_list)
        if np.any(tc == 0):
            uncovered = np.sum(tc == 0)
            return 1000 + uncovered
        return int(np.max(tc))
    
    def smooth_cost(A_list, beta=1.0):
        """Smooth surrogate for max multiplicity using log-sum-exp."""
        tc = get_target_counts(A_list).astype(np.float64)
        # Add small epsilon to avoid log(0)
        tc_safe = np.maximum(tc, 1e-6)
        # Log-sum-exp
        max_tc = np.max(tc_safe)
        lse = max_tc + np.log(np.sum(np.exp(tc_safe - max_tc)))
        # Coverage penalty: large penalty for uncovered sums
        uncovered = np.sum(tc == 0)
        return lse + 100.0 * uncovered
    
    # Initial set: start with a structured set
    # A good starting point: include 0, 128, and some spread
    random.seed(42)
    
    # Start with a reasonable initial set
    # We need k such that k*(k+1)/2 >= 129, so k >= 16
    # Start with k=20
    A_list = list(random.sample(range(0, N + 1), 20))
    A_set = set(A_list)
    
    best_cost = get_cost(A_list)
    best_A = list(A_list)
    
    # Phase 1: Ensure coverage by adding elements
    while time.time() < deadline:
        tc = get_target_counts(A_list)
        uncovered = [s for s in target_sums if tc[s - min_sum] == 0]
        
        if not uncovered:
            break
        
        # Find best element to add
        best_x = None
        best_gain = -1
        
        # Try adding each candidate
        for x in range(0, N + 1):
            if x in A_set:
                continue
            # Gain: how many uncovered sums get covered
            gain = 0
            for y in A_list:
                s = x + y
                if min_sum <= s <= max_sum and tc[s - min_sum] == 0:
                    gain += 1
            # Also x+x
            s = 2 * x
            if min_sum <= s <= max_sum and tc[s - min_sum] == 0:
                gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None or best_gain == 0:
            # Stuck, try random
            candidates = [x for x in range(0, N + 1) if x not in A_set]
            if candidates:
                best_x = random.choice(candidates)
            else:
                break
        
        A_list.append(best_x)
        A_set.add(best_x)
        
        # If set is getting too large, try to prune
        if len(A_list) > 25:
            # Try removing elements that don't break coverage
            for _ in range(3):
                if time.time() >= deadline:
                    break
                for idx in range(len(A_list)):
                    x = A_list[idx]
                    A_list.remove(x)
                    A_set.remove(x)
                    tc = get_target_counts(A_list)
                    if np.any(tc == 0):
                        A_list.append(x)
                        A_set.add(x)
                    else:
                        break
    
    # Phase 2: Local search with simulated annealing
    current_cost = get_cost(A_list)
    current_smooth = smooth_cost(A_list)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_A = list(A_list)
    
    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    
    # Precompute all possible elements outside A for faster swaps
    all_elements = set(range(0, N + 1))
    
    iteration = 0
    stuck_count = 0
    max_stuck = 500
    
    while time.time() < deadline:
        iteration += 1
        T = T_start * (T_end / T_start) ** (iteration / 10000.0)
        if T < T_end:
            T = T_end
        
        # Choose move type
        r = random.random()
        
        if r < 0.7 and len(A_list) > 0:
            # Swap: remove one, add one
            idx = random.randint(0, len(A_list) - 1)
            x_in = A_list[idx]
            x_out = random.choice([x for x in range(0, N + 1) if x not in A_set])
            
            A_list[idx] = x_out
            A_set.remove(x_in)
            A_set.add(x_out)
            
            new_cost = get_cost(A_list)
            new_smooth = smooth_cost(A_list)
            
            # Accept if better or with probability exp(-delta/T)
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_cost = new_cost
                current_smooth = new_smooth
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(A_list)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                # Revert
                A_list[idx] = x_in
                A_set.remove(x_out)
                A_set.add(x_in)
                stuck_count += 1
                
        elif r < 0.85 and len(A_list) > 15:
            # Remove an element
            idx = random.randint(0, len(A_list) - 1)
            x = A_list[idx]
            A_list.pop(idx)
            A_set.remove(x)
            
            new_cost = get_cost(A_list)
            new_smooth = smooth_cost(A_list)
            
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_cost = new_cost
                current_smooth = new_smooth
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(A_list)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                A_list.append(x)
                A_set.add(x)
                stuck_count += 1
                
        elif r < 0.95 and len(A_list) < 30:
            # Add an element
            x = random.choice([x for x in range(0, N + 1) if x not in A_set])
            A_list.append(x)
            A_set.add(x)
            
            new_cost = get_cost(A_list)
            new_smooth = smooth_cost(A_list)
            
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_cost = new_cost
                current_smooth = new_smooth
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(A_list)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                A_list.remove(x)
                A_set.remove(x)
                stuck_count += 1
                
        else:
            # Double swap: remove two, add two
            if len(A_list) >= 2:
                idx1, idx2 = random.sample(range(len(A_list)), 2)
                x1 = A_list[idx1]
                x2 = A_list[idx2]
                x_out1 = random.choice([x for x in range(0, N + 1) if x not in A_set])
                x_out2 = random.choice([x for x in range(0, N + 1) if x not in A_set and x != x_out1])
                
                A_list[idx1] = x_out1
                A_list[idx2] = x_out2
                A_set.remove(x1)
                A_set.remove(x2)
                A_set.add(x_out1)
                A_set.add(x_out2)
                
                new_cost = get_cost(A_list)
                new_smooth = smooth_cost(A_list)
                
                delta = new_smooth - current_smooth
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                    current_cost = new_cost
                    current_smooth = new_smooth
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = list(A_list)
                        stuck_count = 0
                    else:
                        stuck_count += 1
                else:
                    A_list[idx1] = x1
                    A_list[idx2] = x2
                    A_set.remove(x_out1)
                    A_set.remove(x_out2)
                    A_set.add(x1)
                    A_set.add(x2)
                    stuck_count += 1
            else:
                stuck_count += 1
        
        # If stuck too long, restart from best
        if stuck_count > max_stuck:
            A_list = list(best_A)
            A_set = set(A_list)
            current_cost = get_cost(A_list)
            current_smooth = smooth_cost(A_list)
            stuck_count = 0
            T = T_start  # Reset temperature
            
        # Early termination if we found a very good solution
        if best_cost <= 2:
            break
    
    # Final polish: try to reduce cost with greedy local search
    if time.time() < deadline - 5:
        polish_A = list(best_A)
        polish_set = set(polish_A)
        polish_cost = get_cost(polish_A)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try removing each element
            for idx in range(len(polish_A)):
                x = polish_A[idx]
                polish_A.remove(x)
                polish_set.remove(x)
                new_cost = get_cost(polish_A)
                if new_cost < polish_cost:
                    polish_cost = new_cost
                    improved = True
                    break
                else:
                    polish_A.append(x)
                    polish_set.add(x)
            
            if not improved:
                # Try swapping
                for idx in range(len(polish_A)):
                    x_in = polish_A[idx]
                    for x_out in range(0, N + 1):
                        if x_out in polish_set:
                            continue
                        polish_A[idx] = x_out
                        polish_set.remove(x_in)
                        polish_set.add(x_out)
                        new_cost = get_cost(polish_A)
                        if new_cost < polish_cost:
                            polish_cost = new_cost
                            improved = True
                            break
                        else:
                            polish_A[idx] = x_in
                            polish_set.remove(x_out)
                            polish_set.add(x_in)
                    if improved:
                        break
            
            if polish_cost < best_cost:
                best_cost = polish_cost
                best_A = list(polish_A)
    
    return list(best_A)
