import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    n_vecs = 1 << d
    
    # Precompute all vectors in { -1, 1 }^11
    # We use 0/1 representation for Hamming distance calculation, then convert to -1/1 for output
    # Or just work with 0/1 and convert at the end.
    # Let's work with 0/1 internally for faster XOR-based distance?
    # np.bitwise_xor and np.count_nonzero is fast.
    
    # Generate all 0/1 vectors
    indices = np.arange(n_vecs)
    # Convert to binary strings of length d, then to int array
    # This is slow for 2048 vectors, but done once.
    points_bin = np.zeros((n_vecs, d), dtype=np.int8)
    for i in range(n_vecs):
        for j in range(d):
            points_bin[i, j] = (i >> j) & 1
            
    # Start with a good initial code: Shortened Hamming Code
    # Hamming Code [15, 11, 3]
    # Parity check matrix H is 4x15, columns are 1..15 in binary.
    # We select 11 columns. To maximize dimension (minimize rank of selected H), 
    # we should pick columns that are linearly dependent? 
    # No, we want to maximize the number of codewords.
    # Codewords are x such that H_short * x^T = 0.
    # Dimension = 11 - rank(H_short).
    # To maximize dimension, we want to minimize rank(H_short).
    # Min rank is 1? No, columns are non-zero.
    # If we pick 11 columns from 15, can we make rank 3?
    # Max number of distinct non-zero vectors in GF(2)^3 is 7.
    # So rank must be at least 4 if we have more than 7 columns?
    # No, rank of a 4x11 matrix can be 3 if all columns are in a 3-dim subspace.
    # There are only 7 non-zero vectors in a 3-dim subspace.
    # So we can't pick 11 distinct columns with rank 3.
    # We must have rank 4.
    # So dimension is 11 - 4 = 7. Size 128.
    
    # Let's just use a greedy approach with multiple random starts, but faster.
    # Optimization: Use bitsets?
    # For k ~ 100, loop is fine.
    
    start_time = time.time()
    timeout = 550 # Safety margin
    
    best_k = 0
    best_points_bin = None
    
    # Initial guess: Greedy on random permutation
    # We will run many attempts.
    
    # To speed up, we can precompute pairwise distances? 2048^2 is 4M. 
    # 4M * 11 bytes ~ 44MB. Feasible.
    # But we only need to check against selected points.
    
    # Let's try to improve the greedy by sorting candidates by some heuristic?
    # Or just run more attempts.
    
    # Let's try to use the 128-point Hamming code as a seed for local search.
    
    # Construct 128-point code
    # H_short: 4x11 matrix.
    # Pick first 11 columns of H (1..11).
    # H columns:
    # 1: 0001
    # 2: 0010
    # 3: 0011
    # ...
    # 11: 1011
    # Rank of these 11 columns is 4.
    # Kernel of H_short is the code.
    # We can find the basis of the kernel.
    
    H_full = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.int8)
    
    # Take first 11 columns
    H_short = H_full[:, :11]
    
    # Find null space
    # Use numpy linalg? Or simple Gaussian elimination.
    # Since it's small, we can just iterate through all 2^11 = 2048 vectors and check?
    # 2048 * 11 * 4 ops = ~90k ops. Very fast.
    
    init_code = []
    for i in range(1 << 11):
        v = np.zeros(11, dtype=np.int8)
        for j in range(11):
            v[j] = (i >> j) & 1
        # Check H_short @ v == 0
        # h @ v
        if np.all((H_short @ v) % 2 == 0):
            init_code.append(v)
            
    # init_code should have 128 vectors.
    
    # Now, try to extend this code by adding more vectors from the full set.
    # Convert init_code to a list of indices in points_bin
    # points_bin is indexed 0..2047.
    # v is 0/1 array. Convert to index.
    init_indices = []
    for v in init_code:
        idx = 0
        for j in range(11):
            idx |= (v[j] << j)
        init_indices.append(idx)
        
    # Greedy extension
    selected_indices = set(init_indices)
    
    # List of all other vectors
    remaining = [i for i in range(n_vecs) if i not in selected_indices]
    
    # Shuffle remaining
    rng = np.random.default_rng(42)
    rng.shuffle(remaining)
    
    for idx in remaining:
        v_new = points_bin[idx]
        valid = True
        # Check against all selected
        for sel_idx in selected_indices:
            v_sel = points_bin[sel_idx]
            # Hamming distance
            # np.sum(v_new != v_sel)
            # XOR is faster
            diff = np.bitwise_xor(v_new, v_sel)
            if np.count_nonzero(diff) < 3:
                valid = False
                break
        if valid:
            selected_indices.add(idx)
            
    # Convert selected_indices to points
    if len(selected_indices) > best_k:
        best_k = len(selected_indices)
        best_points_bin = points_bin[list(selected_indices)]
        
    # Now run random greedy attempts to see if we can beat this
    for attempt in range(2000):
        if time.time() - start_time > timeout:
            break
            
        rng = np.random.default_rng(attempt + 1000)
        perm = rng.permutation(n_vecs)
        
        selected = []
        selected_set = set()
        
        for idx in perm:
            v_new = points_bin[idx]
            valid = True
            for sel_idx in selected_set:
                v_sel = points_bin[sel_idx]
                diff = np.bitwise_xor(v_new, v_sel)
                if np.count_nonzero(diff) < 3:
                    valid = False
                    break
            if valid:
                selected.append(idx)
                selected_set.add(idx)
                
        if len(selected) > best_k:
            best_k = len(selected)
            best_points_bin = points_bin[selected]
            
    if best_k == 0:
        return np.array([np.ones(d), -1*np.ones(d)], dtype=np.int64)
        
    # Convert 0/1 to -1/1
    # 0 -> -1, 1 -> 1
    # p = 2 * p_bin - 1
    result = 2 * best_points_bin - 1
    return result.astype(np.int64)
