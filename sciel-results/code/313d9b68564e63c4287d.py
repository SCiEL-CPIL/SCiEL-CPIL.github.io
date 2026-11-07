import numpy as np

def construct():
        import time
        start = time.time()
        deadline = start + 85.0
        
        N = 10000000
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Sieve for sigma
        # sigma[n] = sum of divisors of n
        for i in range(1, N + 1):
            if time.time() > deadline - 10.0:
                break
            sigma[i::i] += i
        
        # Prepare data for sorting
        # We need to sort indices 1..N by sigma value
        # argsort on sigma[1:N+1]
        vals = sigma[1:N+1]
        order = np.argsort(vals, kind='stable')
        sorted_vals = vals[order]
        sorted_nums = order + 1  # actual numbers
        
        # Find groups
        # changes = np.concatenate(([True], sorted_vals[1:] != sorted_vals[:-1]))
        # This creates a boolean array of size N.
        
        best_err = float('inf')
        best_a = 1
        best_b = 1
        
        # Iterate through groups
        # To avoid creating huge temporary arrays for every group, we can process chunks
        # But groups are usually small.
        
        # Efficient way:
        # Use np.diff to find boundaries
        diff = np.diff(sorted_vals)
        # boundaries are where diff != 0
        # start indices: 0 and indices where diff != 0
        # end indices: indices where diff != 0 and N-1
        
        # Let's just loop through groups.
        # Number of groups is roughly N / log(N) ? No, sigma values are dense.
        # Actually, sigma(n) grows roughly as n. Distinct values are many.
        
        # We can iterate over `order` and keep track of current group.
        # But we need to access the sorted numbers in the group.
        
        # Alternative:
        # For each unique sigma value, extract the numbers.
        # This might be slow if done in Python loop over unique values.
        # How many unique sigma values?
        # sigma(n) <= ~ 3n. Range is 1..30M.
        # Most values have small fibers.
        
        # Let's try the two-pointer approach on the global sorted array?
        # No, sigma must be equal.
        
        # Let's use the group iteration.
        # To speed up, we can filter out groups of size 1 immediately.
        
        # Create a mask for group starts
        # is_start = np.concatenate(([True], sorted_vals[1:] != sorted_vals[:-1]))
        # group_starts = np.where(is_start)[0]
        
        # This creates arrays of size N.
        
        # Let's do it.
        
        if time.time() > deadline - 5.0:
             # Fallback
             return [23, 15]
             
        is_start = np.empty(N, dtype=bool)
        is_start[0] = True
        is_start[1:] = sorted_vals[1:] != sorted_vals[:-1]
        
        group_starts = np.where(is_start)[0]
        
        # For each group start, find the end
        # group_ends = np.concatenate((group_starts[1:], [N]))
        
        # Iterate
        for i in range(len(group_starts)):
            if time.time() > deadline - 2.0:
                break
                
            gs = group_starts[i]
            ge = group_starts[i+1] if i + 1 < len(group_starts) else N
            
            if ge - gs < 2:
                continue
                
            group_nums = sorted_nums[gs:ge]
            
            # Find best pair in group_nums
            # We want min |2a - 3b| for a,b in group_nums, a!=b
            
            # Two pointer:
            # a_idx, b_idx start at 0?
            # We want a ~ 1.5 b.
            # Let's iterate b and find best a.
            # Since group is sorted, we can use binary search or two pointers.
            # Two pointers:
            # j = 0
            # for k in range(len(group_nums)):
            #    b = group_nums[k]
            #    target = 1.5 * b
            #    while j < len(group_nums) and group_nums[j] < target:
            #         j += 1
            #    # check j and j-1
            
            # This is O(M) per group where M is group size.
            # Total O(N).
            
            L = len(group_nums)
            j = 0
            for k in range(L):
                b = int(group_nums[k])
                target = 1.5 * b
                # Advance j
                while j < L and group_nums[j] < target:
                    j += 1
                    if time.time() > deadline - 1.0: break
                if time.time() > deadline - 1.0: break
                
                # Candidates: j, j-1
                # Ensure j < L
                if j < L:
                    a = int(group_nums[j])
                    if a != b:
                        err = abs(2*a - 3*b) / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                            
                if j > 0:
                    a = int(group_nums[j-1])
                    if a != b:
                        err = abs(2*a - 3*b) / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                            
            if time.time() > deadline - 1.0:
                break
                
        return [int(best_a), int(best_b)]
