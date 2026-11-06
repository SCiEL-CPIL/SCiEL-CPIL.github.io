"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque

def _reset_length_bitmask(a, b):
    """
    Compute shortest reset word length using bitmasks for efficiency.
    a, b: lists of ints 0..N-1
    Returns length or -1 if not synchronizing.
    """
    start_mask = (1 << N) - 1
    
    # Check if a or b is already a reset
    # If a is constant, length 1.
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 1

    visited = {start_mask}
    queue = deque([(start_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
            
        # Apply a
        next_a = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            next_a |= (1 << a[i])
            m -= lsb
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
            
        # Apply b
        next_b = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            next_b |= (1 << b[i])
            m -= lsb
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    # Deterministic construction of the Černý automaton
    # This automaton is known to achieve the maximum reset length of (N-1)^2.
    a = [(i + 1) % N for i in range(N)]
    b = list(range(N - 1)) + [N - 2]
    
    return a, b
# EVOLVE-BLOCK-END
