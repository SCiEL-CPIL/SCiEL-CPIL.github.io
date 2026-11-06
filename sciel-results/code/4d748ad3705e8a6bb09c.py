"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _precompute_ap_matrix():
    """
    Precompute a sparse-like binary matrix A where A[i, j] = 1 if index i (0-based)
    is part of the j-th AP.
    Returns:
        A: numpy array of shape (N, num_aps), dtype int8
        ap_lengths: numpy array of shape (num_aps,), dtype int
    """
    # We will store APs as (start, step, length) and build the matrix.
    # To save memory and time, we can just build the list of indices for each AP first?
    # No, we need the matrix for fast multiplication.
    # Number of APs is approx sum_{d=1}^{N/2} (N-d) * (N/d - 1) ~ N^2 * log N ~ 256^2 * 5 ~ 300k.
    # Matrix size: 256 * 300k = 76M entries. int8 -> 76MB. This is acceptable.
    
    # Let's count exactly how many APs we generate to allocate.
    # Actually, building a dense numpy array of (N, num_aps) might be slow to initialize if num_aps is large.
    # Let's collect indices first in a list of lists, then convert.
    
    ap_indices_list = []
    
    # Iterate over step
    for step in range(1, N // 2 + 1):
        # Iterate over start
        for start in range(1, N - step + 1):
            # Max length
            max_len = (N - start) // step + 1
            if max_len < 2:
                continue
            
            # For each possible length L from 2 to max_len
            # The AP is start, start+step, ..., start+(L-1)*step
            # Indices in 0-based: start-1, start-1+step, ...
            
            # Optimization: Instead of iterating L from 2 to max_len and creating lists,
            # we can just append the full list of indices for the max length?
            # No, the task asks for ANY length.
            # However, notice that the sum of a prefix of an AP is a sub-sum.
            # If we minimize the max sum over all *full* APs (max length), does it cover shorter ones?
            # No. A shorter AP could have a larger sum than the full one if signs cancel out later.
            # Example: +1, -1, +1. Full sum = 1. Prefix sum (len 2) = 0. Prefix sum (len 1) = 1.
            # But wait, length 1 is trivial (sum is +/-1). The constraint is max abs sum.
            # If we include length 1, the max is at least 1.
            # The prompt says: "including one". So singletons are APs.
            # Max abs sum for a singleton is always 1.
            # So the lower bound is 1.
            
            # We must consider all lengths L >= 2 as well.
            
            # Let's generate all (start, step, L) tuples.
            for L in range(2, max_len + 1):
                # Generate indices
                # idxs = [start - 1 + k * step for k in range(L)]
                # This is slow in Python loop.
                # Let's just collect the tuples and build the matrix in one go?
                # Or build the matrix incrementally?
                
                # Let's just store the start, step, length.
                ap_indices_list.append((start, step, L))
                
    num_aps = len(ap_indices_list)
    
    # Initialize matrix
    # Using float32 or int8? int8 is enough for 0/1.
    # But numpy multiplication with int8 and float might need casting.
    # Let's use float32 for the matrix to allow fast dot products with float signs.
    A = np.zeros((N, num_aps), dtype=np.float32)
    ap_lengths = np.zeros(num_aps, dtype=np.int32)
    
    for j, (start, step, L) in enumerate(ap_indices_list):
        ap_lengths[j] = L
        # Fill column j
        # Indices: start-1, start-1+step, ...
        idxs = np.arange(start - 1, N, step)[:L]
        A[idxs, j] = 1.0
        
    return A, ap_lengths

def _compute_max_abs_sum_vectorized(signs, A, ap_lengths):
    """
    Compute max absolute sum over all APs using matrix multiplication.
    signs: shape (N,), float or int
    A: shape (N, num_aps)
    """
    # sums = A.T @ signs  (shape num_aps,)
    sums = A.T.dot(signs)
    max_abs = np.max(np.abs(sums))
    return max_abs

def _construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    print("Precomputing AP matrix...", flush=True)
    A, ap_lengths = _precompute_ap_matrix()
    num_aps = A.shape[1]
    print(f"Matrix shape: {A.shape}, Time: {time.time() - start_time:.2f}s", flush=True)
    
    # Precompute A.T for faster access? A.T is a view, usually fine.
    AT = A.T  # shape (num_aps, N)
    
    # Initial candidates
    rng = random.Random(42)
    
    def get_random_signs(seed):
        r = random.Random(seed)
        return np.array([r.choice([-1.0, 1.0]) for _ in range(N)], dtype=np.float32)
    
    def get_thue_morse():
        signs = np.zeros(N, dtype=np.float32)
        for i in range(N):
            # Count set bits
            c = bin(i).count('1')
            signs[i] = 1.0 if c % 2 == 0 else -1.0
        return signs
    
    candidates = [
        get_thue_morse(),
        get_random_signs(1),
        get_random_signs(2),
        get_random_signs(3),
        get_random_signs(4),
    ]
    
    # Evaluate initial candidates
    best_signs = None
    best_val = float('inf')
    
    for cand in candidates:
        val = _compute_max_abs_sum_vectorized(cand, A, ap_lengths)
        if val < best_val:
            best_val = val
            best_signs = cand.copy()
            
    print(f"Best initial value: {best_val}", flush=True)
    
    # Optimization: Coordinate Descent / Greedy Flip
    # We want to minimize max(|sums|).
    # Current sums: S = AT @ signs
    # If we flip sign at index i:
    # new_signs[i] = -signs[i]
    # new_S = S - 2 * signs[i] * AT[:, i]
    # We can compute the new max for a specific flip in O(num_aps) which is ~300k.
    # Doing this for all 256 positions in each step is 256 * 300k = 76M ops per step.
    # In NumPy, this can be done in parallel if we vectorize the flip evaluation?
    
    # Let's try a batched evaluation of all possible single flips.
    # For each position i, calculate the change in sums: delta_i = -2 * signs[i] * AT[:, i]
    # new_sums_i = S + delta_i
    # max_abs_i = max(|new_sums_i|)
    # We want the i that minimizes max_abs_i.
    
    # This can be done with a matrix multiplication:
    # Deltas shape: (num_aps, N) -> AT * (-2 * signs)
    # But we need the max over the first dimension for each column.
    
    # Let's implement a function to find the best single flip.
    
    def find_best_flip(signs, current_sums):
        # signs: (N,)
        # current_sums: (num_aps,)
        
        # Calculate delta for each position
        # delta_ji = -2 * signs[i] * A[i, j]
        # We can compute the matrix of new sums for all flips at once?
        # New_sums_matrix[i, j] = current_sums[j] - 2 * signs[i] * A[i, j]
        # This matrix is (N, num_aps). 256 * 300k = 76M floats. 300MB. Might be too slow/memory heavy.
        
        # Alternative: Iterate over positions i, but vectorize the check for that i.
        # Or, use a smarter heuristic.
        
        # Let's stick to evaluating a subset of flips or using a faster loop if memory is tight.
        # However, 76M float32 ops is very fast in NumPy (<10ms).
        
        # Let's try the full matrix approach.
        # signs_col = signs.reshape(-1, 1) # (N, 1)
        # A_transposed = A.T # (num_aps, N)
        # Deltas = -2.0 * A_transposed * signs_col.T # (num_aps, N)
        # New_Sums = current_sums.reshape(1, -1) + Deltas # (num_aps, N)
        # Max_Abs_Sums = np.max(np.abs(New_Sums), axis=0) # (N,)
        # best_idx = np.argmin(Max_Abs_Sums)
        
        # Let's test if this fits in time.
        # 76M multiplications and additions. Should be < 50ms.
        # If we do this for 1000 steps, that's 50s. Might be tight.
        
        # Let's optimize: Only consider flips that might reduce the max.
        # The max is currently M.
        # A flip is useful only if it reduces the sum of the APs that are currently at +/- M.
        
        # Simpler: Just do the full vectorized check. It's robust.
        
        signs_col = signs.reshape(1, -1) # (1, N)
        # A is (N, num_aps). We need (num_aps, N) to multiply with (1, N) to get (num_aps, N)?
        # No.
        # We want Deltas[j, i] = -2 * signs[i] * A[i, j]
        # A.T is (num_aps, N).
        # Deltas = -2.0 * A.T * signs_col # (num_aps, N)
        
        Deltas = -2.0 * A.T * signs_col
        New_Sums = current_sums.reshape(1, -1) + Deltas
        Max_Abs_Sums = np.max(np.abs(New_Sums), axis=0)
        
        best_idx = np.argmin(Max_Abs_Sums)
        best_val = Max_Abs_Sums[best_idx]
        
        return best_idx, best_val

    # Main loop: Greedy Coordinate Descent
    # We will perform flips as long as they improve the objective.
    # To escape local minima, we can add Simulated Annealing later if time permits.
    
    signs = best_signs.copy()
    current_sums = AT.dot(signs)
    current_max = np.max(np.abs(current_sums))
    
    steps = 0
    no_improve_count = 0
    
    while time.time() < deadline - 5.0:
        if no_improve_count > 100:
            # Try a random restart or annealing step
            # For now, let's try a random flip if stuck
            if rng.random() < 0.1:
                idx = rng.randint(0, N - 1)
                signs[idx] *= -1
                current_sums = AT.dot(signs)
                current_max = np.max(np.abs(current_sums))
                no_improve_count = 0
                continue
            else:
                break
        
        idx, new_max = find_best_flip(signs, current_sums)
        
        if new_max < current_max:
            # Apply flip
            signs[idx] *= -1
            # Update sums efficiently:
            # current_sums -= 2 * old_sign * A[:, idx]
            # But we already computed new_max via the full matrix.
            # We can just recompute current_sums = AT.dot(signs) for simplicity,
            # or update it.
            # Update:
            old_sign = -signs[idx] # since we flipped
            current_sums -= 2.0 * old_sign * A[:, idx]
            
            current_max = new_max
            no_improve_count = 0
            steps += 1
        else:
            no_improve_count += 1
            
    print(f"Greedy descent finished. Steps: {steps}, Max: {current_max}, Time: {time.time() - start_time:.2f}s", flush=True)
    
    # If time remains, try Simulated Annealing from this best state
    if time.time() < deadline - 10.0:
        sa_start_time = time.time()
        sa_deadline = deadline - 5.0
        
        T = 5.0 # Initial temperature
        cooling = 0.99
        
        sa_signs = signs.copy()
        sa_sums = current_sums.copy()
        sa_max = current_max
        best_sa_signs = sa_signs.copy()
        best_sa_max = sa_max
        
        while time.time() < sa_deadline:
            if T < 0.1:
                break
                
            idx = rng.randint(0, N - 1)
            old_sign = sa_signs[idx]
            sa_signs[idx] = -old_sign
            
            # Calculate new max
            # new_sums = sa_sums - 2 * old_sign * A[:, idx]
            # Instead of full update, just compute the new max for this specific flip
            # We can reuse the find_best_flip logic but for a single index?
            # Or just compute the new sums and max.
            
            # Efficient single flip evaluation:
            # delta = -2 * old_sign * A[:, idx]
            # new_sums = sa_sums + delta
            # new_max = max(abs(new_sums))
            
            delta = -2.0 * old_sign * A[:, idx]
            new_sums = sa_sums + delta
            new_max = np.max(np.abs(new_sums))
            
            delta_E = new_max - sa_max
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept
                sa_sums = new_sums
                sa_max = new_max
                if sa_max < best_sa_max:
                    best_sa_max = sa_max
                    best_sa_signs = sa_signs.copy()
            else:
                # Reject
                sa_signs[idx] = old_sign
                
            T *= cooling
            
        if best_sa_max < current_max:
            current_max = best_sa_max
            signs = best_sa_signs
            print(f"Simulated Annealing improved to: {current_max}", flush=True)
    
    # Convert back to list of ints
    final_signs = [int(s) for s in signs]
    
    # Final verification (optional, but good for sanity)
    # val = _compute_max_abs_sum_vectorized(np.array(final_signs, dtype=np.float32), A, ap_lengths)
    # print(f"Verified final value: {val}", flush=True)
    
    return final_signs

def construct():
    return _construct()

# EVOLVE-BLOCK-END
