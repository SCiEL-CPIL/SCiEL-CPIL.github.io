import numpy as np
import time
import random

def _check_addition_fast(sums_arr, x):
    """
    sums_arr: numpy array of sorted current sums.
    x: new element.
    Returns new_sums_arr (sorted numpy array) if valid, else None.
    """
    n = len(sums_arr)
    
    # 1. Check x vs sums_arr
    # x divides s or s divides x
    if np.any(sums_arr % x == 0):
        return None
    if np.any(x % sums_arr == 0):
        return None
        
    # 2. Generate shifted sums
    shifted = sums_arr + x
    
    # 3. Check shifted vs x
    # shifted[i] > x always (since sums_arr > 0)
    # So only check if shifted[i] % x == 0
    if np.any(shifted % x == 0):
        return None
        
    # 4. Check shifted vs sums_arr
    # We need to check if any shifted[i] is divisible by any sums_arr[j] or vice versa.
    # Since shifted[i] = sums_arr[i] + x > sums_arr[i], 
    # it's possible that sums_arr[j] divides shifted[i].
    # It's also possible that shifted[i] divides sums_arr[k]? 
    # shifted[i] > sums_arr[i], but shifted[i] could be smaller than some sums_arr[k] if k > i?
    # No, sums_arr is sorted. shifted[i] = sums_arr[i] + x.
    # If x is large, shifted[i] is large.
    # Let's just check both directions.
    
    # To avoid N^2 memory blowup for large N, we can chunk it.
    # But for N=4000, N^2 = 16M. 16M * 8 bytes = 128MB. Acceptable.
    
    # Check shifted (N) vs sums_arr (N)
    # Matrix M1 = shifted[:, None] % sums_arr[None, :]
    # If any M1[i,j] == 0, then sums_arr[j] divides shifted[i].
    # Matrix M2 = sums_arr[:, None] % shifted[None, :]
    # If any M2[i,j] == 0, then shifted[i] divides sums_arr[j].
    
    # We can combine: check if (shifted[:, None] % sums_arr[None, :]) has 0s.
    # And check if (sums_arr[:, None] % shifted[None, :]) has 0s.
    
    # Optimization: 
    # If shifted[i] < sums_arr[j], then sums_arr[j] % shifted[i] != 0 (unless shifted[i] is 0, which it isn't).
    # So we only need to check pairs where the dividend is >= divisor.
    
    # Let's just do the full check for simplicity first.
    
    # Check shifted vs sums_arr
    # This is expensive.
    # Let's try a loop over shifted, and for each, check against sums_arr.
    # But that's O(N^2) in Python.
    
    # Let's use NumPy broadcasting.
    # This might be slow if N is large.
    
    # Alternative: 
    # Only check if the ratio is an integer.
    
    # Let's try the NumPy broadcast.
    # If it's too slow, we'll see.
    
    # Create matrices
    # To save memory, we can do it in blocks.
    # But let's try full first.
    
    # Check 1: sums_arr[j] divides shifted[i]
    # shifted[i] % sums_arr[j] == 0
    # We can compute this as:
    # rem = shifted[:, None] % sums_arr[None, :]
    # if np.any(rem == 0): return None
    
    # Check 2: shifted[i] divides sums_arr[j]
    # sums_arr[j] % shifted[i] == 0
    # rem2 = sums_arr[:, None] % shifted[None, :]
    # if np.any(rem2 == 0): return None
    
    # This is 2 * N^2 operations.
    # For N=4000, 32M ops. NumPy does this in ~0.1s.
    # For N=8000, 128M ops. ~0.5s.
    # This is feasible for a few hundred iterations.
    
    # Check 5: shifted vs shifted
    # shifted[i] vs shifted[j]
    # rem3 = shifted[:, None] % shifted[None, :]
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    # if np.any((rem3 == 0) & mask): return None
    
    # This is another N^2.
    
    # Total 3 * N^2.
    
    # Let's implement this.
    
    # Note: We must handle the case where N is 0.
    if n == 0:
        return np.array([x])
        
    # Check shifted vs sums_arr
    # To reduce memory, we can check in chunks of shifted.
    # But let's try full.
    
    # 1. sums_arr[j] divides shifted[i]
    # We can restrict the check: shifted[i] must be >= sums_arr[j].
    # Since shifted is sorted and sums_arr is sorted, we can use searchsorted.
    # But for now, let's just do the full modulo.
    
    # Memory check: N=10000 -> 100M elements -> 800MB. Too much.
    # We need to chunk.
    
    chunk_size = 1000
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        # shifted[i:end] vs sums_arr
        # shifted[i:end, None] % sums_arr[None, :]
        rem1 = shifted[i:end, None] % sums_arr[None, :]
        if np.any(rem1 == 0):
            return None
            
        rem2 = sums_arr[None, :] % shifted[i:end, None]
        if np.any(rem2 == 0):
            return None
            
    # Check shifted vs shifted
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        # shifted[i:end] vs shifted
        rem3 = shifted[i:end, None] % shifted[None, :]
        # Mask diagonal
        # The diagonal for this chunk is tricky.
        # We just check all pairs in the chunk against all shifted.
        # If i==j, it's self, which is 0.
        # We need to ignore self.
        # Since we are checking shifted[i:end] vs ALL shifted, 
        # the self pairs are (k, k) where k is in [i, end).
        # In the matrix rem3, the row corresponds to shifted[i:end] and col to shifted.
        # The diagonal elements of the FULL matrix are at (k, k).
        # In the submatrix, the diagonal is not necessarily on the main diagonal.
        # Easier: just check if any is 0, and if the pair is the same index, skip.
        # But we can't easily identify indices in the flat array.
        
        # Alternative: Just check rem3 == 0.
        # If rem3[k, l] == 0, then shifted[l] divides shifted[i+k].
        # If i+k == l, it's self.
        # We can subtract the identity matrix contribution?
        # Or just check: if any 0, verify if it's self.
        # Since N is not huge, we can just find the indices.
        
        # For simplicity and speed, let's assume self-divisibility is the only 0 on the diagonal.
        # We can mask the diagonal of the FULL matrix, but we are doing chunks.
        
        # Let's just do:
        zeros = np.argwhere(rem3 == 0)
        for r, c in zeros:
            idx_r = i + r
            idx_c = c
            if idx_r != idx_c:
                return None
                
    new_sums = np.concatenate([sums_arr, shifted, [x]])
    new_sums.sort()
    return new_sums

def construct():
    deadline = time.time() + 85
    best_set = []
    best_len = 0
    
    # Try to find a set of size K
    # Start with K=14 and go down? Or start small and go up?
    # Going up is better because we can use the previous solution as a seed.
    
    # Initial seed
    rng = random.Random(42)
    current_sums = np.array([], dtype=int)
    current_elements = []
    
    # Greedy initialization
    candidates = list(range(256, 0, -1))
    rng.shuffle(candidates)
    
    for c in candidates:
        if time.time() > deadline:
            break
        new_sums = _check_addition_fast(current_sums, c)
        if new_sums is not None:
            current_sums = new_sums
            current_elements.append(c)
            
    best_set = current_elements
    best_len = len(current_elements)
    
    # Now try to grow using backtracking/local search
    # We will try to add one more element.
    # If we can't, we try to swap.
    
    # Simple local search:
    # Try to add a random unused element.
    # If it fails, try to remove a random element and add two?
    # Or just try to add.
    
    # Let's try to iterate:
    # For each step, try to add the best candidate.
    # If no candidate works, try to swap.
    
    # To make it robust, we'll do a few restarts with different random seeds.
    
    for seed in range(10):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        cur_sums = np.array([], dtype=int)
        cur_elems = []
        
        # Random start
        start_size = rng.randint(3, 5)
        start_elems = rng.sample(range(128, 257), start_size)
        
        # Check if start is valid
        valid = True
        for e in start_elems:
            new_sums = _check_addition_fast(cur_sums, e)
            if new_sums is None:
                valid = False
                break
            cur_sums = new_sums
            cur_elems.append(e)
            
        if not valid:
            continue
            
        # Grow greedily
        candidates = [x for x in range(256, 0, -1) if x not in cur_elems]
        rng.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            new_sums = _check_addition_fast(cur_sums, c)
            if new_sums is not None:
                cur_sums = new_sums
                cur_elems.append(c)
                
        if len(cur_elems) > best_len:
            best_set = cur_elems
            best_len = len(cur_elems)
            
    return best_set
