import numpy as np

def construct():
    p = 1009
    # Compute inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Convert to numpy for vectorization
    inv_arr = np.array(inv)
    
    best_start = 2
    best_len = 0
    
    for s in range(2, 1008):
        # For this start, find max end
        # We need: for all x in [s, e], inv[x] not in [s, e]
        # i.e., inv[x] < s or inv[x] > e
        
        # Try all e from s to 1007
        # Use vectorized check
        max_e = s
        for e in range(s, 1008):
            # Check if [s, e] is valid
            # Get all x in [s, e]
            # Check if any has inv[x] in [s, e]
            if e - s + 1 > 0:
                x_range = np.arange(s, e + 1)
                inv_vals = inv_arr[x_range]
                # Check if any inv_val is in [s, e]
                if np.any((inv_vals >= s) & (inv_vals <= e)):
                    break
                else:
                    max_e = e
        
        length = max_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
