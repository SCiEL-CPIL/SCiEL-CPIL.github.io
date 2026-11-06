"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Scale weights and capacity to integers for DP
    # Using a scale factor of 100 to balance precision and speed
    scale = 100
    C = int(remaining_capacity * scale)
    if C <= 0:
        return -1
    
    # Convert weights to scaled integers
    # Using round to handle floating point imprecision
    W = [int(round(w * scale)) for w in remaining_weights]
    V = list(remaining_values)
    
    # DP Table: dp[i][j] represents the max value using the first i items 
    # with a capacity of j
    # Dimensions: (n + 1) x (C + 1)
    # Since C is small (<= 1250), this is efficient.
    
    # Initialize DP table
    # Using a list of lists for faster access in Python loops than NumPy arrays
    dp = [[0] * (C + 1) for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        wi = W[i - 1]
        vi = V[i - 1]
        dpi = dp[i]
        dpi_prev = dp[i - 1]
        
        if wi > C:
            # Item is too heavy for any capacity in our scaled range
            # So we just copy the previous row
            dpi[:] = dpi_prev[:]
        else:
            # Iterate through capacities
            # We can optimize by starting from 0, but standard forward fill is fine
            for j in range(C + 1):
                # Option 1: Do not take item i
                best = dpi_prev[j]
                
                # Option 2: Take item i (if it fits)
                if j >= wi:
                    val_take = dpi_prev[j - wi] + vi
                    if val_take > best:
                        best = val_take
                        
                dpi[j] = best
                
    # Backtrack to find one item that is part of the optimal solution
    # We start from the last item index n and capacity C
    i = n
    j = C
    while i > 0:
        # If the value changed when we included item i (index i-1), 
        # then item i-1 is part of the optimal solution
        if dp[i][j] != dp[i - 1][j]:
            return i - 1
        else:
            # Item i-1 is not in this optimal path, move to previous item
            i -= 1
            # Capacity j remains the same
            
    # If no item was found, it means the optimal value is 0 (no items picked)
    return -1
# EVOLVE-BLOCK-END
