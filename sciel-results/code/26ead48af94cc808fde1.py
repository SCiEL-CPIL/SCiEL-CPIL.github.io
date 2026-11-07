import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands, dtype=float)
    
    # Sort demands for efficient counting
    sorted_demands = np.sort(demands)
    
    # Initialize heuristic matrix
    H = np.zeros((n, n), dtype=float)
    
    # Precompute counts for each possible residual?
    # Residuals are continuous, but we only need counts for specific R_ij.
    
    for i in range(n):
        di = demands[i]
        for j in range(n):
            if i == j:
                H[i, j] = 0
                continue
            dj = demands[j]
            pair_sum = di + dj
            if pair_sum > capacity:
                H[i, j] = 0
            else:
                residual = capacity - pair_sum
                # Count items k (k != i, k != j) such that demands[k] <= residual
                # Use binary search on sorted_demands
                # Number of items <= residual
                count_le = np.searchsorted(sorted_demands, residual, side='right')
                
                # Exclude i and j if they fit in the residual
                # Note: If di <= residual, then di is counted in count_le.
                # We need to remove i and j from the count if they were included.
                
                # Check if di <= residual
                if di <= residual:
                    count_le -= 1
                # Check if dj <= residual
                if dj <= residual:
                    count_le -= 1
                
                # Ensure count is not negative (shouldn't happen if logic is right, but safe)
                if count_le < 0:
                    count_le = 0
                    
                # Heuristic: Inverse of (count + 1)
                # Higher is better. 
                # If count=0 (dead gap), H=1.
                # If count=1, H=0.5.
                H[i, j] = 1.0 / (count_le + 1)
                
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # Our H is either 0 or positive.
    # So we can just replace 0 with 1e-9.
    H = np.where(H > 0, H, 1e-9)
    
    return H
