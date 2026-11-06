import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    n = 128
    target_lo = 128
    target_hi = 256
    target_range = range(target_lo, target_hi + 1)
    num_targets = len(target_range)

    # Precompute for speed
    # A is a set of indices in [0, n].
    # We represent A as a boolean array of size n+1 for fast membership checks.
    # We also maintain a list of active elements for iteration.

    def get_score(counts):
        """
        Returns (uncovered, peak, variance)
        We prioritize:
        1. Minimize uncovered (0 is best)
        2. Minimize peak (if uncovered == 0)
        3. Minimize variance (to flatten distribution)
        """
        uncovered = 0
        peak = 0
        total_sq = 0
        total = 0
        # Use numpy for faster reduction on counts
        # But counts is a list. Converting to numpy every time is slow.
        # We can keep counts as numpy array.
        
        # Let's assume counts is a numpy array for speed in the main loop.
        c_arr = counts
        # Counts in target range
        c_target = c_arr[target_lo:target_hi+1]
        
        # Uncovered
        uncovered = np.sum(c_target == 0)
        
        if uncovered == 0:
             peak = np.max(c_target)
             # Variance
             mean = np.mean(c_target)
             var = np.var(c_target)
        else:
             peak = 0
             var = 0.0
            
        return int(uncovered), int(peak), float(var)

    def calculate_initial_counts_numpy(A_set):
        """
        Calculate counts for a given A_set (set of ints).
        Returns numpy array of size 257.
        """
        counts = np.zeros(257, dtype=np.int32)
        # Convert to list for iteration
        A_list = list(A_set)
        # If A is small, O(N^2) is fine.
        # To be faster, we can use convolution?
        # Indicator vector v of size 129.
        # v * v (convolution) gives counts of ordered pairs.
        # We want unordered pairs with replacement.
        # Let C be the convolution of v with v.
        # C[s] = sum_{i+j=s} v[i]v[j]
        # This counts (i,j) and (j,i) if i!=j, and (i,i) once.
        # We want unordered pairs:
        # If i != j: count 1 for (i,j) and (j,i) in C, but we want 1. So C[s]/2?
        # If i == j: count 1 in C, we want 1.
        
        # Formula:
        # Let U be the set of unordered pairs {a,b} with a,b in A.
        # Count of s in U is the number of pairs.
        # Convolution K[s] = sum_{i+j=s} v[i]v[j].
        # K[s] = 2 * (number of pairs {i,j} with i!=j, i+j=s) + (number of pairs {i,i} with 2i=s).
        # Let N_neq[s] be number of distinct pairs i<j with i+j=s.
        # Let N_eq[s] be 1 if s is even and s/2 in A, else 0.
        # Then K[s] = 2*N_neq[s] + N_eq[s].
        # We want Total[s] = N_neq[s] + N_eq[s].
        # So Total[s] = (K[s] + N_eq[s]) / 2.
        
        v = np.zeros(n+1, dtype=np.int8)
        for x in A_set:
            v[x] = 1
            
        # Convolution
        K = np.convolve(v, v, mode='full')
        # K size is 2*n+1 = 257.
        
        # Calculate N_eq
        # N_eq[s] = 1 if s is even and s/2 in A.
        # s ranges 0..2n.
        # s/2 ranges 0..n.
        # Check if s is even.
        # eq_counts = np.zeros(257, dtype=np.int8)
        # for s in range(0, 257, 2):
        #    if s/2 < n+1:
        #       eq_counts[s] = v[s//2]
        
        # Vectorized:
        # indices 0, 2, 4, ... 256
        even_indices = np.arange(0, 257, 2)
        half_indices = even_indices // 2
        # Filter half_indices that are within [0, n]
        valid_mask = half_indices <= n
        valid_even_indices = even_indices[valid_mask]
        valid_half_indices = half_indices[valid_mask]
        
        N_eq = np.zeros(257, dtype=np.int32)
        N_eq[valid_even_indices] = v[valid_half_indices]
        
        counts = (K + N_eq) // 2
        return counts

    def get_delta_counts(x, y, A_set):
        """
        Calculate delta in counts when removing x and adding y.
        Returns (delta_arr, new_A_set)
        """
        # This is complex to vectorize easily for a single step.
        # We'll stick to incremental updates in the main loop for speed.
        pass

    # --- Optimized Local Search ---
    
    def local_search(A_init, time_limit):
        """
        Starts with A_init (set), runs local search to minimize peak.
        Uses incremental count updates.
        """
        A = list(A_init)
        A_set = set(A)
        counts = calculate_initial_counts_numpy(A_set)
        
        curr_unc, curr_peak, curr_var = get_score(counts)
        best_A = A[:]
        best_counts = counts.copy()
        best_score = (curr_unc, curr_peak, curr_var)
        best_val = curr_unc * 10000 + curr_peak * 100 + curr_var
        
        # SA Parameters
        T = 50.0
        cooling = 0.995
        min_T = 0.1
        
        # Precompute possible moves? No, too many.
        # Move: Remove x, Add y.
        
        while time.time() < time_limit:
            if T < min_T:
                # Reset T to continue searching? Or stop.
                # Let's stop if we've been at min_T for a while.
                break
                
            # Select x to remove
            if not A:
                break
            x = random.choice(A)
            
            # Select y to add
            # To ensure progress, we can bias y towards covering gaps.
            # But for now, random.
            y = random.randint(0, n)
            
            # Avoid x == y
            if y == x:
                continue
            # Avoid y already in A
            if y in A_set:
                # Try a few times
                for _ in range(5):
                    y = random.randint(0, n)
                    if y != x and y not in A_set:
                        break
                else:
                    continue
            
            # Apply Move: Remove x, Add y
            # 1. Remove x
            # Update counts: subtract contributions of x with all z in A (z != x)
            # And subtract contribution of x+x.
            
            # Contributions of x:
            # Pairs (x, z) for z in A, z != x. Sum s = x+z.
            # Pair (x, x). Sum s = 2x.
            
            # Note: A currently contains x.
            # We iterate over z in A.
            # If z == x, it's the (x,x) pair.
            # If z != x, it's (x,z) pair.
            
            # Since we are modifying counts in place, we must be careful.
            # Let's store the changes.
            
            # We need the list of z in A BEFORE removing x.
            A_old = A[:]
            
            for z in A_old:
                if z == x:
                    s = 2 * x
                    if target_lo <= s <= target_hi:
                        counts[s] -= 1
                else:
                    s = x + z
                    if target_lo <= s <= target_hi:
                        counts[s] -= 1
            
            # Remove x from structures
            A.remove(x)
            A_set.remove(x)
            
            # 2. Add y
            # Contributions of y:
            # Pairs (y, z) for z in A (current, x is removed). Sum s = y+z.
            # Pair (y, y). Sum s = 2y.
            
            A_new = A[:] # A does not contain x, but contains other elements.
            
            for z in A_new:
                s = y + z
                if target_lo <= s <= target_hi:
                    counts[s] += 1
            
            s2 = 2 * y
            if target_lo <= s2 <= target_hi:
                counts[s2] += 1
                
            A.append(y)
            A_set.add(y)
            
            # Evaluate
            new_unc, new_peak, new_var = get_score(counts)
            new_val = new_unc * 10000 + new_peak * 100 + new_var
            
            delta = new_val - best_val # Compare to best? No, compare to current.
            # We need current_val
            curr_val = curr_unc * 10000 + curr_peak * 100 + curr_var
            delta = new_val - curr_val
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                # Accept
                curr_unc, curr_peak, curr_var = new_unc, new_peak, new_var
                curr_val = new_val
                
                # Check for new best
                if new_val < best_val:
                    best_val = new_val
                    best_A = A[:]
                    best_counts = counts.copy()
                    best_score = (new_unc, new_peak, new_var)
            else:
                # Reject: Rollback
                # Remove y
                for z in A:
                    if z == y:
                        s = 2 * y
                        if target_lo <= s <= target_hi:
                            counts[s] -= 1
                    else:
                        s = y + z
                        if target_lo <= s <= target_hi:
                            counts[s] -= 1
                
                A.remove(y)
                A_set.remove(y)
                
                # Add x back
                for z in A:
                    s = x + z
                    if target_lo <= s <= target_hi:
                        counts[s] += 1
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    counts[s2] += 1
                    
                A.append(x)
                A_set.add(x)
                
                # curr state is unchanged, so no need to update curr_unc etc.
            
            T *= cooling
            
        return best_A, best_score

    # --- Initialization ---
    
    def greedy_build():
        """
        Builds a valid set A using greedy coverage.
        """
        A = []
        A_set = set()
        counts = np.zeros(257, dtype=np.int32)
        
        # Start with some random elements
        for _ in range(20):
            x = random.randint(0, n)
            if x not in A_set:
                A.append(x)
                A_set.add(x)
        
        counts = calculate_initial_counts_numpy(A_set)
        
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Check if all covered
            c_target = counts[target_lo:target_hi+1]
            if np.all(c_target > 0):
                break
                
            # Find best x to add
            best_x = -1
            best_gain = -1
            
            # Candidates: all x in [0, n] not in A
            # We can sample if too slow, but n=128 is small.
            for x in range(n + 1):
                if x in A_set:
                    continue
                
                # Calculate gain
                # Gain = number of newly covered sums
                # And penalty for increasing peak
                
                # Simulate adding x
                new_zeros = 0
                max_count_after = 0
                
                # Contributions of x
                for z in A:
                    s = x + z
                    if target_lo <= s <= target_hi:
                        c = counts[s]
                        if c == 0:
                            new_zeros += 1
                        if c + 1 > max_count_after:
                            max_count_after = c + 1
                
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    c = counts[s2]
                    if c == 0:
                        new_zeros += 1
                    if c + 1 > max_count_after:
                        max_count_after = c + 1
                
                # Score: Prioritize coverage
                # Penalty if max_count_after is high
                # We want to keep counts low.
                # Let's say if max_count_after > 15, penalize.
                penalty = 0
                if max_count_after > 15:
                    penalty = (max_count_after - 15) * 10
                
                score = new_zeros * 100 - penalty
                
                if score > best_gain:
                    best_gain = score
                    best_x = x
            
            if best_x == -1:
                break
                
            # Add best_x
            A.append(best_x)
            A_set.add(best_x)
            
            # Update counts
            for z in A:
                if z == best_x:
                    s = 2 * best_x
                    if target_lo <= s <= target_hi:
                        counts[s] += 1
                else:
                    s = best_x + z
                    if target_lo <= s <= target_hi:
                        counts[s] += 1
            
        return A, counts

    # --- Main Loop ---
    
    best_overall_A = None
    best_overall_val = float('inf')
    
    # Try different initial sizes
    # Smaller sets are easier to cover but might have higher peaks?
    # Actually, larger sets give more flexibility to balance peaks.
    # But too large increases total counts.
    # Optimal size is likely around 30-50.
    
    for i in range(50):
        if time.time() > deadline - 10:
            break
            
        # Greedy build
        A_init, _ = greedy_build()
        
        # Refine size?
        # Maybe remove redundant elements to reduce peak?
        # Or just let local search handle it.
        
        # Local Search
        remaining = deadline - time.time()
        ls_time = max(1.0, remaining * 0.5)
        
        A_res, score_res = local_search(A_init, time.time() + ls_time)
        
        unc, peak, var = score_res
        val = unc * 10000 + peak * 100 + var
        
        if val < best_overall_val:
            best_overall_val = val
            best_overall_A = A_res
            
        # Early stop if perfect
        if unc == 0 and peak <= 4:
            break
            
    if best_overall_A is None:
        # Fallback: Random set
        return list(range(64, 129))
        
    return best_overall_A

# Keep the required interface
# The task says: "Keep the function names and signatures of the seed program."
# Seed program had:
# def construct():
# ...
# def construct():  # (duplicate in seed, but effectively one)
#
# My code has one construct().

if __name__ == "__main__":
    A = construct()
    print(A)
