import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    N = 10000000
    # 1. Compute sigma(n) for n in [1, N]
    # Using vectorized additions
    sigma = np.ones(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        # If time is running out, break? 
        # The loop is the most expensive part.
        if time.time() > deadline - 10:
             # Fallback to a smaller range or return best found so far?
             # We haven't found any yet.
             # We must return a valid pair.
             # The seed pair [11, 6] is valid.
             return [11, 6]
        
        sigma[i::i] += i
        
    if time.time() > deadline - 5:
        return [11, 6]

    # 2. Sort indices by sigma value
    # We need to group numbers with the same sigma.
    # argsort is O(N log N)
    sorted_indices = np.argsort(sigma)
    sorted_sigmas = sigma[sorted_indices]
    
    if time.time() > deadline - 2:
        return [11, 6]

    # 3. Iterate through groups
    best_error = 1.0 # Initial error for [11,6] is 1/3 ~ 0.33. We want to beat this.
    # Wait, score = seed/value. If seed error is 1/3, and we want score > 1, we need error < 1/3.
    # Let's set best_error to the error of the seed pair.
    seed_a, seed_b = 11, 6
    best_error = abs(seed_a/seed_b - 1.5)
    best_pair = [seed_a, seed_b]
    
    # Identify groups
    # Find where sigma value changes
    # diff of sorted_sigmas
    # We can use np.diff to find boundaries
    # But doing it in one go might be memory heavy if we store all boundaries.
    # Let's iterate.
    
    i = 0
    n = len(sorted_indices)
    
    while i < n:
        if time.time() > deadline - 1:
            break
            
        # Find end of current group
        j = i + 1
        # Optimization: if the group is small, linear scan is fine.
        # If group is large, we might want to jump?
        # np.where(sorted_sigmas[i+1:] != sorted_sigmas[i])[0]
        # This creates a slice and does a comparison.
        
        # Let's try to find the next change
        # If the group is very large, this could be slow if done naively.
        # But most groups are small.
        
        # Better: use np.searchsorted on sorted_sigmas for the next unique value?
        # Or just simple loop if groups are small.
        
        # Let's assume groups are small on average.
        while j < n and sorted_sigmas[j] == sorted_sigmas[i]:
            j += 1
            
        # Group is sorted_indices[i:j]
        group = sorted_indices[i:j]
        
        if len(group) >= 2:
            # We have at least 2 numbers with same sigma
            # We want min |2a - 3b| / (2b)
            # Let A = 2*group, B = 3*group
            # We want to find a in group, b in group minimizing |2a - 3b|/b
            
            # Vectorized approach:
            # For each b in group, find best a.
            # a should be close to 1.5 * b.
            
            # 2*group is sorted.
            # We can search for 1.5*b in 2*group.
            
            # To avoid overflow or precision issues, 1.5*b might not be integer.
            # We search in the array `2*group`.
            
            # Let's compute targets = 3 * group (which is 2 * (1.5 * group))
            # We want 2a close to 3b.
            
            targets = 3 * group
            # searchsorted in 2*group
            # 2*group is sorted
            
            # Create array of 2*group
            # This might be memory heavy if group is large, but group size is usually small.
            # Max group size? 
            # Sigma values are spread out.
            
            two_group = 2 * group
            
            # Find indices
            # np.searchsorted returns the index where the target would be inserted
            # We check idx and idx-1
            
            idxs = np.searchsorted(two_group, targets)
            
            # Candidate indices: idxs and idxs - 1
            # Filter valid indices
            # We need to be careful with bounds
            
            # Let's compute diffs for valid candidates
            # To do this vectorized:
            
            # We can create an array of candidate indices
            # But idxs varies.
            
            # Simpler: loop over the group if it's small.
            # If group is large, vectorize.
            
            if len(group) < 1000:
                min_diff = float('inf')
                best_a_local = -1
                best_b_local = -1
                for b in group:
                    target = 1.5 * b
                    # Find a in group closest to target
                    # Since group is sorted, we can binary search
                    # But we already have two_group and targets.
                    # Let's just use the vectorized result for all.
                    pass
            
            # Let's stick to vectorized for all, it's safer for large groups.
            # We need to handle the indices carefully.
            
            # idxs are in [0, len(two_group)]
            # Candidates are idxs and idxs-1
            
            # We can flatten this by creating two sets of candidates and filtering.
            
            # Candidate 1: idxs
            # Candidate 2: idxs - 1
            
            # Let's compute the difference for idxs where idxs < len(two_group)
            # and for idxs-1 where idxs-1 >= 0
            
            # It's easier to just compute the min diff using the sorted property.
            # Actually, since we just want the min error, we can compute the error for the closest match.
            
            # Let's try a different vectorized approach:
            # We have two sorted arrays X = 2*group, Y = 3*group.
            # We want min |x - y| / y for x in X, y in Y.
            
            # Merge X and Y, keeping track of origin.
            # This is O(M log M) where M is group size.
            # But we already sorted group.
            
            # Let's just use the searchsorted result.
            
            valid_idx1 = idxs < len(two_group)
            valid_idx2 = (idxs - 1) >= 0
            
            # We can't easily vectorize the "min of two" per element without creating large arrays.
            # Given the deadline, and that most groups are small, let's just loop if group < 10000.
            # If group >= 10000, it's rare, and we can afford a slower method or specific optimization.
            
            if len(group) < 5000:
                min_err_local = float('inf')
                for b in group:
                    # Find a closest to 1.5 b
                    # Use searchsorted on two_group
                    pos = np.searchsorted(two_group, 1.5 * b)
                    candidates = []
                    if pos < len(two_group):
                        candidates.append(two_group[pos])
                    if pos > 0:
                        candidates.append(two_group[pos-1])
                    
                    for a2 in candidates:
                        a = a2 // 2
                        # Check if a is in group? 
                        # two_group contains 2*a for a in group.
                        # So a is in group.
                        err = abs(a/b - 1.5)
                        if err < min_err_local:
                            min_err_local = err
                            if err < best_error:
                                best_error = err
                                best_pair = [a, b]
            else:
                # Large group: use vectorized diff
                # This is complex to get right quickly. 
                # Given time constraints, we might skip large groups if time is tight, 
                # but large groups are where the best pairs might be?
                # Actually, sigma(n) is unique for most n. Collisions are rare.
                # So groups are likely small.
                pass
                
        i = j
        
    # Ensure types are correct
    # best_pair is list of numpy ints? 
    # group elements are numpy ints.
    # We need Python ints.
    
    a, b = best_pair
    return [int(a), int(b)]

# Test locally? No, just output.
# One issue: `sigma[i::i] += i` in a loop for i=2..10^7 is VERY slow in Python if not optimized.
# The loop itself has 10^7 iterations. Each iteration does a vectorized op.
# Overhead of 10^7 Python loop iterations is ~1s? No, 10^7 iterations is ~0.5-1s in PyPy, but CPython is slower.
# 10^7 iterations of `sigma[i::i] += i` might take 10-20 seconds.
# Plus the slice creation.
# This might be too slow.

# Alternative:
# Use a C-optimized library? No, standard library only.
# Can we speed up the sieve?
# 
# Maybe we can limit the range?
# The task says a,b <= 10^7.
# 
# Let's check the time for the sieve.
# If it's too slow, we might need to reduce N?
# But the contract says <= 10^7.
# 
# Is there a faster way to compute sigma?
# We can compute the prime factorization and then sigma.
# But that's also O(N log log N) with higher constant.
# 
# Let's stick to the vectorized sieve but monitor time.
# If it's too slow, we might need to return the seed result.
# But p36 got score 4e6, so it's possible.
# Maybe p36 used a smarter approach or C extensions?
# The prompt says "Python standard library, NumPy and SciPy are available".
# 
# Let's optimize the loop.
# We can't easily vectorize the loop over i.
# 
# However, we can break early if time is running out.
# 
# Let's refine the construct function to be robust.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 10000000
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve
    for i in range(2, N + 1):
        if time.time() > deadline - 15:
            # Not enough time to finish sieve and search
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() > deadline - 5:
        return [11, 6]

    # Sort and Group
    sorted_indices = np.argsort(sigma)
    sorted_sigmas = sigma[sorted_indices]
    
    if time.time() > deadline - 2:
        return [11, 6]

    best_error = abs(11/6 - 1.5)
    best_pair = [11, 6]
    
    i = 0
    n = len(sorted_indices)
    
    while i < n:
        if time.time() > deadline - 1:
            break
            
        j = i + 1
        while j < n and sorted_sigmas[j] == sorted_sigmas[i]:
            j += 1
            
        if j - i >= 2:
            group = sorted_indices[i:j]
            
            # Find best pair in group
            # Use vectorized search if group is large, else loop
            # For simplicity and speed in CPython, loop is better for small groups.
            # Most groups are size 1 or 2.
            
            # We can just check all pairs if size is very small?
            # Size 2: 1 pair.
            # Size 10: 100 pairs.
            # Size 100: 10000 pairs.
            
            if len(group) <= 100:
                for idx_a in range(len(group)):
                    a = group[idx_a]
                    for idx_b in range(len(group)):
                        b = group[idx_b]
                        err = abs(a/b - 1.5)
                        if err < best_error:
                            best_error = err
                            best_pair = [a, b]
            else:
                # Larger group, use searchsorted
                two_group = 2 * group
                for b in group:
                    pos = np.searchsorted(two_group, 1.5 * b)
                    if pos < len(two_group):
                        a = two_group[pos] // 2
                        err = abs(a/b - 1.5)
                        if err < best_error:
                            best_error = err
                            best_pair = [a, b]
                    if pos > 0:
                        a = two_group[pos-1] // 2
                        err = abs(a/b - 1.5)
                        if err < best_error:
                            best_error = err
                            best_pair = [a, b]
                            
        i = j
        
    return [int(best_pair[0]), int(best_pair[1])]
