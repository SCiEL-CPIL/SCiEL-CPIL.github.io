# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n_max = 512
    max_size = 48
    
    # dp_bits[k] is a Python integer where bit s is set if there is a subset of size k with sum s
    # Initialize for empty set
    dp_bits = [0] * (max_size + 1)
    dp_bits[0] = 1  # Sum 0 with size 0
    
    selected = []
    
    # Try descending order to maximize gaps
    for x in range(n_max, 0, -1):
        if len(selected) >= max_size:
            break
            
        current_size = len(selected)
        new_size = current_size + 1
        
        # Calculate new DP bits
        new_dp_bits = [0] * (new_size + 1)
        
        # Inherit from old set
        for k in range(current_size + 1):
            new_dp_bits[k] = dp_bits[k]
            
        # Add subsets including x
        for k in range(1, new_size):
            new_dp_bits[k] |= (dp_bits[k-1] << x)
            
        # Check validity
        valid = True
        
        # Check existing elements
        for a in selected:
            for k in range(2, new_size + 1):
                if (new_dp_bits[k] >> (k * a)) & 1:
                    valid = False
                    break
            if not valid:
                break
                
        # Check x
        if valid:
            for k in range(2, new_size + 1):
                if (new_dp_bits[k] >> (k * x)) & 1:
                    valid = False
                    break
                    
        if valid:
            selected.append(x)
            dp_bits = new_dp_bits
            
    return selected

# EVOLVE-BLOCK-END
