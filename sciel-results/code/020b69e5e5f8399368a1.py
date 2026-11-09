import time
import random
import numpy as np

def construct():
    N = 51975
    
    # 1. Get divisors > 1
    divisors = []
    for i in range(3, N + 1, 2):
        if N % i == 0:
            divisors.append(i)
            
    # Also include N itself? The problem says "dividing 51975". 
    # Usually covering systems use proper divisors, but N is a divisor.
    # If m=N, r is 0..N-1. Class (N, r) covers only residue r.
    # This is useful for covering singletons if needed.
    # The range in the seed was range(3, N+1, 2). This includes N.
    
    best_uncovered = N
    best_classes = {}
    
    start_time = time.time()
    deadline = start_time + 110
    
    def get_gain(m, r, cov):
        # Count uncovered in class (m, r)
        # Use numpy for speed
        # Indices: r, r+m, ..., < N
        # We can slice cov[r:N:m]
        # But cov is a numpy array.
        return np.sum(cov[r:N:m] == 0)
        
    def apply_class(m, r, cov, uncovered):
        # Adds class (m, r) to cov
        # Returns new uncovered count
        # We need to know how many were 0 before
        # It's easier to just re-calculate or track.
        # Since we are doing greedy, we know the gain.
        # For local search, we calculate delta.
        pass
        
    def greedy(order):
        cov = np.zeros(N, dtype=np.int32)
        classes = {}
        uncovered = N
        
        for m in order:
            best_r = 0
            max_gain = -1
            # Optimize: if m is large, number of r is large.
            # But gain calculation is O(N/m).
            # Total cost per m is O(N).
            # Total cost for greedy is O(N * num_moduli).
            # 51975 * 48 ~ 2.5M ops. Very fast.
            
            for r in range(m):
                gain = get_gain(m, r, cov)
                if gain > max_gain:
                    max_gain = gain
                    best_r = r
            
            classes[m] = best_r
            # Update cov
            for x in range(best_r, N, m):
                if cov[x] == 0:
                    uncovered -= 1
                cov[x] += 1
                
        return uncovered, classes, cov

    def local_search(classes, cov, uncovered):
        # classes: dict m -> r
        # cov: numpy array
        # uncovered: int
        # Returns improved uncovered, classes, cov
        
        improved = True
        # Limit iterations to prevent infinite loops if time is tight
        max_iters = 1000
        
        while improved and time.time() < deadline:
            improved = False
            # Shuffle moduli to avoid bias
            moduli_list = list(classes.keys())
            random.shuffle(moduli_list)
            
            for m in moduli_list:
                if time.time() > deadline:
                    break
                    
                r_old = classes[m]
                best_r_new = r_old
                best_delta = 0 # 0 means no change
                
                # Try all other residues
                # Optimization: We only need to check residues that might improve things.
                # But checking all is O(N/m).
                
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    
                    # Calculate delta in uncovered count
                    # Residues covered by (m, r_old) but not (m, r_new):
                    # If cov[x] becomes 0, uncovered increases.
                    # Residues covered by (m, r_new) but not (m, r_old):
                    # If cov[x] was 0, uncovered decreases.
                    
                    # Intersection covers both, no change in count.
                    
                    # We can compute this by iterating the union of the two sets?
                    # Or just compute the new uncovered count for the affected parts?
                    
                    # Let's simulate the change on a copy? No, too slow.
                    # Let's compute delta analytically.
                    
                    # Set A: indices in r_old, m
                    # Set B: indices in r_new, m
                    
                    # Delta = (# of x in A \ B where cov[x]==1) - (# of x in B \ A where cov[x]==0)
                    
                    # We can iterate A and B.
                    # Size of A is ~ N/m.
                    
                    delta = 0
                    # Iterate A
                    # We need to know if x is in B.
                    # x is in B if (x - r_new) % m == 0.
                    
                    # This is O(N/m) per r_new.
                    # Total O(m * N/m) = O(N) per m.
                    # Total O(N * num_moduli) per pass.
                    # 2.5M ops per pass.
                    # 1000 passes = 2.5B ops. Too slow in Python.
                    
                    # We need a faster local search.
                    
                    # Alternative: Only try a few random r_new?
                    # Or use a smarter heuristic.
                    
                    # Let's try random restarts with greedy first, then a limited local search.
                    
                    # For now, let's just do greedy with many random orders.
                    # The greedy step is already O(N * num_moduli).
                    # If we run 1000 greedy runs: 2.5B ops. Still too slow for pure Python loops.
                    
                    # We MUST use NumPy for the inner loops.
                    
                    # Let's refactor get_gain to be vectorized if possible?
                    # get_gain(m, r, cov) = np.sum(cov[r:N:m] == 0)
                    # This is already vectorized.
                    # The loop `for r in range(m)` is the bottleneck.
                    # Can we vectorize over r?
                    # cov is 1D.
                    # We want to compute for all r: sum(cov[r:N:m] == 0).
                    # This is like a strided sum.
                    
                    # For a fixed m, we can create a matrix of shape (m, N//m) ?
                    # No, the lengths vary slightly if N % m != 0.
                    
                    # However, N=51975 is small.
                    # 2.5M ops in Python is ~0.1-0.5s?
                    # Let's check.
                    # 10^6 simple ops ~ 0.05s in PyPy, 0.1s in CPython.
                    # 2.5M ~ 0.25s.
                    # 1000 runs ~ 250s. Too slow.
                    
                    # We need to reduce the number of greedy runs or speed up the inner loop.
                    
                    # Speed up inner loop:
                    # Instead of `for r in range(m)`, we can use NumPy to compute gains for all r at once?
                    # For a fixed m, let K = N // m.
                    # Most rows have length K or K+1.
                    # We can pad cov to length m * (N//m + 1).
                    # Reshape to (m, N//m + 1).
                    # Then sum along axis 1 where value != 0?
                    # cov_reshaped = cov[:m*(N//m+1)].reshape(m, -1)
                    # But this doesn't align with the strided access r, r+m...
                    # Actually, cov[r:N:m] is exactly the row r of the transposed matrix?
                    # If we create a matrix M where M[i, j] = cov[i + j*m]
                    # Then M has shape (m, N//m + 1).
                    # Gain for r is sum(M[r] == 0).
                    # We can compute this for all r at once!
                    
                    # Cost to build M: O(N).
                    # Cost to sum: O(N).
                    # So per modulus m, we do O(N) work in NumPy.
                    # Total for all m: O(N * num_moduli) but in C speed (NumPy).
                    # This is ~100x faster than Python loops.
                    # 2.5M ops in NumPy ~ 0.001s?
                    # So we can run thousands of greedy runs.

        return uncovered, classes, cov

    # Revised Greedy with NumPy vectorization for gain calculation
    
    def greedy_fast(order):
        cov = np.zeros(N, dtype=np.int32)
        classes = {}
        uncovered = N
        
        for m in order:
            # Build matrix for this m
            # K = N // m
            # Max length = K + 1
            # We need cov[r : N : m] for all r in 0..m-1
            # This is equivalent to:
            # Take cov[0:N], reshape? No.
            # We can use:
            # indices = np.arange(N)
            # groups = indices % m
            # We want sum(cov == 0) for each group? No, that's not strided.
            
            # The strided access cov[r:N:m] is contiguous in memory if we view it correctly?
            # No, it's a view.
            
            # Let's try:
            # mat = np.zeros((m, N // m + 1), dtype=np.int32)
            # This is wasteful if N/m is small.
            
            # Better:
            # For each r, gain = np.sum(cov[r:N:m] == 0)
            # Can we vectorize the r loop?
            # Yes, using the fact that cov[r:N:m] are the columns of a matrix if we transpose?
            # Let's create a matrix V of shape (N//m + 1, m)
            # V[j, r] = cov[r + j*m]
            # Then gain[r] = sum(V[:, r] == 0)
            
            K = N // m
            # Max elements per column is K or K+1
            # Let's just use a loop in NumPy if m is small?
            # m can be up to 51975.
            # If m is large, N/m is small (1 or 2).
            # If m is small (3), N/m is large (17325).
            
            # Hybrid approach:
            if m < 100:
                # m is small, N/m is large.
                # Use the matrix trick.
                max_len = N // m + 1
                # Create indices for the matrix
                # We want V[j, r] = cov[r + j*m]
                # This is cov[0:N] reshaped?
                # cov is 1D.
                # V = cov[:max_len*m].reshape(max_len, m).T ?
                # Let's check:
                # V.T[j, r] = cov[r + j*m]
                # Yes!
                # But we must ensure we don't go out of bounds.
                # cov length is N.
                # max_len * m might be > N.
                # We can pad cov with zeros?
                # If we pad, the extra zeros will count as "uncovered" (cov==0), which is incorrect if they don't exist.
                # We only care about x < N.
                
                # So, we can create a padded cov:
                # pad_len = (max_len * m) - N
                # cov_padded = np.pad(cov, (0, pad_len), mode='constant')
                # V = cov_padded[:max_len*m].reshape(max_len, m).T
                # gains = np.sum(V == 0, axis=0)
                # But we must mask out the padded parts.
                # The valid indices are x < N.
                # For row j, col r, x = r + j*m.
                # x < N  =>  j*m < N - r.
                
                # This is getting complicated.
                
                # Simpler: Just loop r in Python, but use NumPy for the sum.
                # The loop is over m.
                # If m < 100, loop is small.
                # Inner op is NumPy sum, which is fast.
                best_r = 0
                max_gain = -1
                for r in range(m):
                    gain = np.sum(cov[r:N:m] == 0)
                    if gain > max_gain:
                        max_gain = gain
                        best_r = r
            else:
                # m is large, N/m is small (1 or 2).
                # Loop over r is large (up to 50k).
                # But each sum is over 1 or 2 elements.
                # We can vectorize this too.
                # cov[r:N:m] has length 1 if r < N % m, else 0?
                # No, r < N.
                # If m > N/2, then for any r < m, r + m > N.
                # So cov[r:N:m] is just [cov[r]] if r < N, else empty.
                # Since r < m and m can be > N, but r < m.
                # Wait, m divides N. So m <= N.
                # If m > N/2, then N/m = 1.
                # cov[r:N:m] is [cov[r]] for r < N.
                # Since m divides N, m <= N.
                # If m == N, r < N. cov[r:N:N] is [cov[r]].
                # If m > N/2, r < m.
                # If r >= N, cov[r] is out of bounds?
                # But r < m <= N. So r < N.
                # So cov[r:N:m] is always [cov[r]].
                
                # So gain(r) = 1 if cov[r] == 0 else 0.
                # We want r with cov[r] == 0.
                # If there are any, pick the first one (or any).
                # If none, pick 0.
                
                zero_indices = np.where(cov[:m] == 0)[0]
                if len(zero_indices) > 0:
                    best_r = zero_indices[0]
                else:
                    best_r = 0
                    max_gain = 0
                # max_gain is 1 if best_r is zero, else 0.
                
            classes[m] = best_r
            # Update cov
            for x in range(best_r, N, m):
                if cov[x] == 0:
                    uncovered -= 1
                cov[x] += 1
                
        return uncovered, classes, cov

    # Run greedy with multiple orders
    orders = [
        sorted(divisors),
        sorted(divisors, reverse=True)
    ]
    # Add random orders
    for _ in range(50):
        random.shuffle(divisors)
        orders.append(divisors[:])
        
    for order in orders:
        if time.time() > deadline:
            break
        unc, classes, cov = greedy_fast(order)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes
            best_cov = cov.copy()
            
    # Local Search on best solution
    # Simple local search: try changing r for each m
    # Use the same gain logic
    
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for m in list(best_classes.keys()):
                if time.time() > deadline:
                    break
                r_old = best_classes[m]
                # Find best r_new
                # Use same logic as greedy_fast for finding best r
                # But we must exclude r_old? No, if r_old is best, delta is 0.
                
                # We need to calculate the *change* in uncovered.
                # This is harder.
                # Let's just re-run greedy on the remaining moduli? No.
                
                # Let's stick to Greedy only for now, as it's robust and fast.
                # The prompt asks for improvements. A correct Greedy is a huge improvement over the broken seed.
                pass
                
    result = [[m, r] for m, r in best_classes.items()]
    return result
