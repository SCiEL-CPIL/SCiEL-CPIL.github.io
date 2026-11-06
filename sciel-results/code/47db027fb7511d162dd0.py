"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 256
    # We will optimize a continuous vector x in [-1, 1] to minimize
    # the maximum absolute sum over all arithmetic progressions.
    
    # Precompute all arithmetic progressions (start, step, length)
    # An AP is defined by start index s (0-based), step d, and length L.
    # The indices are s, s+d, s+2d, ..., s+(L-1)d.
    # These indices must be < n.
    
    ap_indices = []
    for d in range(1, n):
        for s in range(n):
            # Find max length for this start and step
            # Max index is s + (L-1)*d < n => L <= (n-1-s)//d + 1
            max_l = (n - 1 - s) // d + 1
            # We need to include all lengths from 1 to max_l
            # Actually, for a given start and step, the APs are prefixes of the full sequence
            # starting at s with step d.
            # So for each (s, d), we have APs of length 1, 2, ..., max_l.
            # However, note that an AP of length L starting at s is the same set of indices
            # as a subset of the AP of length L+1.
            # But the sum is different.
            # We need to consider all distinct APs.
            # An AP is uniquely identified by (s, d, L).
            # But wait, does (s, d, L) cover all APs?
            # Yes, any AP has a first term, a step, and a length.
            
            # To be efficient, we can just generate the indices for each (s, d, L).
            # But there are many.
            # Number of (s, d) pairs is roughly n^2/2 = 32768.
            # For each, max_l is at most n/d.
            # Total APs is sum_{d=1}^{n-1} sum_{s=0}^{n-1} (n-1-s)//d + 1
            # This is roughly sum_{d=1}^{n} n^2/d = n^2 * H_n ~ 256^2 * 6 ~ 393k.
            # This is manageable.
            
            for L in range(1, max_l + 1):
                indices = np.arange(s, s + L * d, d)
                ap_indices.append(indices)
    
    # Convert to a matrix A where each row is an indicator vector for an AP
    # A[i, j] = 1 if j is in AP i, else 0
    # This might be memory heavy: 393k rows * 256 cols * 8 bytes = ~800 MB. Too much.
    
    # Alternative: Instead of storing all APs, we can compute the loss on the fly
    # or use a subset of APs that are likely to be the worst.
    # Or, we can use a smarter representation.
    
    # Let's try a different approach: use a small set of "critical" APs.
    # Or, use stochastic gradient descent where we sample a batch of APs each iteration.
    
    # Let's use SGD with a batch of APs.
    # Initialize x to random values in [-1, 1]
    np.random.seed(42)
    x = np.random.uniform(-1, 1, n)
    
    # Learning rate schedule
    lr = 0.1
    batch_size = 1000
    
    # Precompute a list of all AP index arrays for sampling
    # To save memory, we store them as lists of lists or arrays.
    # 393k arrays of small size is okay if we don't make a big matrix.
    
    # Let's precompute the list of AP index arrays
    all_aps = []
    for d in range(1, n):
        for s in range(n):
            max_l = (n - 1 - s) // d + 1
            for L in range(1, max_l + 1):
                all_aps.append(np.arange(s, s + L * d, d))
    
    num_aps = len(all_aps)
    
    # SGD loop
    for step in range(5000):
        # Sample a batch of APs
        if batch_size < num_aps:
            batch_idx = np.random.choice(num_aps, batch_size, replace=False)
        else:
            batch_idx = np.arange(num_aps)
        
        # Compute sums for the batch
        # sums[i] = sum_{j in AP_i} x[j]
        sums = np.zeros(batch_size)
        for i, idx in enumerate(batch_idx):
            sums[i] = np.sum(x[idx])
        
        # Loss: sum of squares of absolute sums
        # To make it differentiable and smooth, we use sum(sums^2)
        # Gradient of sum(sums^2) w.r.t. x is 2 * sum_i sums[i] * indicator_i
        grad = np.zeros(n)
        for i, idx in enumerate(batch_idx):
            grad[idx] += 2 * sums[i]
        
        # Update x
        x -= lr * grad
        
        # Clip x to [-1, 1] to keep it in the feasible region for the final discretization
        # Actually, we don't need to clip during optimization if we use a smooth loss,
        # but clipping can help with convergence.
        x = np.clip(x, -1, 1)
        
        # Decay learning rate
        lr = 0.1 / (1 + step * 0.001)
    
    # Discretize to ±1
    signs = np.sign(x)
    # Handle zeros (if any)
    signs[signs == 0] = 1
    
    return [int(s) for s in signs]
# EVOLVE-BLOCK-END
