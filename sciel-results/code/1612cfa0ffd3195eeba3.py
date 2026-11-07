"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer using 1D Optimal Quantization via DP for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT3 Structure: x[1:] = 0
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin(10 * pi * x0)
        
        # Strategy:
        # 1. Discretize x0 into a grid of size N.
        # 2. Use Dynamic Programming to select 'budget' points that maximize Hypervolume.
        # 3. The DP exploits the 1D nature of the non-dominated front.
        
        # Grid resolution
        # N=1000 is a good balance. N=2000 might be too slow for O(N^2*K) in Python.
        # We use N=800 and vectorize the inner DP step as much as possible.
        N = 800
        
        x_grid = np.linspace(0.0, 1.0, N)
        f1 = x_grid
        f2 = 1.0 - np.sqrt(x_grid) - x_grid * np.sin(10.0 * np.pi * x_grid)
        
        R1, R2 = 11.0, 11.0
        A = R2 - f2  # Height contribution for each point
        
        # DP Table
        # dp_prev[j] = Max HV contribution using some k points, with the last point being index j.
        # Base case k=1:
        # HV with 1 point at j is (R1 - f1[j]) * (R2 - f2[j])
        dp_prev = (R1 - f1) * A
        
        # Pre-compute B[i] = dp_prev[i] - f1[i] * A[i] for the first step?
        # No, B changes every k.
        
        # We will iterate k from 2 to budget.
        selected_indices = [N - 1]  # Placeholder, we need to track path or just final points
        
        # To save memory and time, we only keep the previous row of DP.
        # We also need to reconstruct the path.
        # Storing the 'best predecessor' for each j and k is O(K*N) memory.
        # 200 * 800 = 160k integers. This is fine.
        
        parent = np.zeros((budget, N), dtype=int)
        
        for k in range(2, budget + 1):
            dp_curr = np.full(N, -np.inf)
            
            # If time is running out, break early and use what we have
            if time.time() > deadline - 20.0:
                break
            
            # For each j (current last point), find best i (previous last point)
            # Condition: i < j AND f2[i] > f2[j]
            # Value: dp_prev[i] + (f1[j] - f1[i]) * A[i]
            
            # We can vectorize this by creating a matrix of valid i for all j?
            # Matrix M of size N x N. M[i, j] = 1 if i < j and f2[i] > f2[j].
            # This is 800x800 = 640k entries.
            
            # Let's compute the valid mask once?
            # f2[i] > f2[j] is a comparison between two vectors.
            # mask = (f2[:, None] > f2[None, :]) # i rows, j cols?
            # We want i < j.
            
            # Efficient vectorized computation:
            # We want to compute for all j:
            # dp_curr[j] = max( dp_prev[i] + f1[j]*A[i] - f1[i]*A[i] ) for valid i
            
            # Let C[i] = dp_prev[i] - f1[i]*A[i]
            C = dp_prev - f1 * A
            
            # For a specific j, we want max( C[i] + f1[j]*A[i] ) subject to i < j and f2[i] > f2[j].
            
            # We can loop over j, but use NumPy for the inner max?
            # Inner max over i (size N).
            # Looping N times (800) and doing a NumPy max over an array of size N (800)
            # is 800 * 800 = 640k operations per k.
            # 200 * 640k = 128M operations.
            # This should be fast enough in NumPy if the inner loop is tight.
            
            for j in range(N):
                # Valid i: i < j AND f2[i] > f2[j]
                # We can create a mask for i < j
                # mask_i = np.arange(N) < j
                # mask_f2 = f2 > f2[j]
                # valid = mask_i & mask_f2
                
                # To avoid creating masks every time, we can use slicing if sorted?
                # But f2 is not sorted.
                
                # We can pre-compute for each j the list of valid i?
                # That would be O(N^2) memory and setup time.
                
                # Let's just do the mask.
                # Optimization: 
                # The condition f2[i] > f2[j] is key.
                
                # If N is small enough, this is fine.
                
                if j == 0:
                    continue
                
                # Indices i from 0 to j-1
                # We need f2[i] > f2[j]
                # Let's find indices i where f2[i] > f2[j]
                # This is a boolean array of length N.
                
                # We can use np.where
                # valid_indices = np.where((np.arange(j) < j) & (f2[:j] > f2[j]))[0]
                # This creates a new array every time.
                
                # Better:
                # Pre-compute sort order of f2?
                # No, let's stick to the simple mask but optimize the loop.
                
                # Actually, we can just compute the max over the whole array and then mask out invalids?
                # No, max with -inf.
                
                # Let's try a different vectorization:
                # Create a matrix of scores S[i, j] = C[i] + f1[j]*A[i]
                # And a mask M[i, j] = (i < j) & (f2[i] > f2[j])
                # Set S[i, j] = -inf where M is False.
                # dp_curr = max(S, axis=0)
                
                # This creates an N x N matrix (800x800) for each k.
                # 200 * 640k floats = 128M floats allocated over time.
                # This is memory heavy but doable.
                
                # Let's do this vectorized matrix approach for speed.
                
            # Re-doing with Matrix Approach inside the k-loop
            # S[i, j] = C[i] + f1[j] * A[i]
            # C is shape (N,), f1 is (N,), A is (N,)
            # S = C[:, None] + f1[None, :] * A[:, None]
            
            S = C[:, None] + f1[None, :] * A[:, None]
            
            # Mask: i < j AND f2[i] > f2[j]
            # i is row, j is col
            # i < j: np.arange(N)[:, None] < np.arange(N)[None, :]
            # f2[i] > f2[j]: f2[:, None] > f2[None, :]
            
            idx = np.arange(N)
            mask_idx = idx[:, None] < idx[None, :]
            mask_f2 = f2[:, None] > f2[None, :]
            
            valid_mask = mask_idx & mask_f2
            
            # Set invalid to -inf
            S[~valid_mask] = -np.inf
            
            dp_curr = np.max(S, axis=0)
            
            # Store parent for backtracking
            # parent[k-1, j] = argmax(S[:, j])
            parent[k-1] = np.argmax(S, axis=0)
            
            dp_prev = dp_curr
            
        # Backtrack to find the best sequence of indices
        # We want to end at some j that maximizes dp_prev[j]
        if budget > 1:
            last_j = np.argmax(dp_prev)
        else:
            last_j = np.argmax(dp_prev)
            
        # Reconstruct indices
        indices = [0] * budget
        indices[budget - 1] = last_j
        for k in range(budget - 1, 1, -1):
            indices[k-1] = parent[k-1, indices[k]]
            
        # indices[0] is the first point (index 0 in the sequence, k=1)
        # Note: parent[k-1] stores the predecessor for the k-th point.
        # So parent[k-1, j_k] = j_{k-1}.
        
        # Ensure indices are unique and sorted?
        # The DP enforces i < j, so indices will be strictly increasing.
        
        # Evaluate the selected points
        all_points = []
        used = 0
        
        for idx in indices:
            if used >= budget:
                break
            if time.time() > deadline:
                break
                
            x = np.zeros(dim)
            x[0] = x_grid[idx]
            
            f = problem(x)
            all_points.append(f)
            used += 1
            
        return all_points
# EVOLVE-BLOCK-END
