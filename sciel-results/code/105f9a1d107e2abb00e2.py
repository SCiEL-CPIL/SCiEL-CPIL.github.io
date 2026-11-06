<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16

# EVOLVE-BLOCK-START
def _verify_network(network, n=N):
    """Verify that the network sorts all 2^n inputs using zero-one principle."""
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        for i, j in network:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        if wires != sorted(wires):
            return False
    return True
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _verify_network(network, n=N):
    """
    Verify that the network sorts all 2^n inputs using zero-one principle.
    Uses NumPy vectorization for speed.
    """
    if not network:
        return n <= 1
    
    # Create all 2^n inputs as a matrix of shape (2^n, n)
    # Each row is a binary representation of the mask, with bit 0 at index 0
    masks = np.arange(1 << n, dtype=np.uint8)
    # We need to map mask bits to wires. 
    # Input wires[i] is bit i of the mask.
    # We can create a matrix where rows are masks and cols are wires.
    # wire_matrix[k, i] = (mask_k >> i) & 1
    
    # Efficient way to generate the bit matrix:
    # shifts = 1 << np.arange(n)
    # bits = (masks[:, None] & shifts[None, :]) > 0
    
    shifts = 1 << np.arange(n, dtype=np.uint8)
    # Initialize wires with all possible inputs
    wires = (masks[:, None] & shifts[None, :]).astype(np.uint8)
    
    # Apply comparators
    for i, j in network:
        # wires[:, i] and wires[:, j]
        wi = wires[:, i]
        wj = wires[:, j]
        # min is 0 if either is 0, max is 1 if either is 1
        # Since inputs are 0/1:
        # min(wi, wj) = wi & wj
        # max(wi, wj) = wi | wj
        wires[:, i] = wi & wj
        wires[:, j] = wi | wj
        
    # Check if sorted: wires should be 0...0 1...1
    # For each row, check if it is non-decreasing
    # A binary string is sorted if it is of the form 0...01...1
    # This is equivalent to checking if there is no '1' followed by '0'
    # i.e., wires[:, k] > wires[:, k+1] for any k
    
    # Check adjacent pairs
    if n > 1:
        diffs = wires[:, 1:] - wires[:, :-1]
        # If any diff is negative (1 to 0), it's not sorted
        if np.any(diffs < 0):
            return False
            
    return True
>>>>>>> REPLACE
