<<<<<<< SEARCH
import time
import random
from collections import deque

N = 12

def calculate_threshold_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using bitmask representation.
    Much faster than frozenset approach.
    """
    start = (1 << N) - 1  # All states set
    
    # If start is already a single state (only possible if N=1, but N=12 here)
    # Check for single state: power of 2
    if (start & (start - 1)) == 0:
        return 0
    
    visited = {start: 0}
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        # Check if curr is a single state (power of 2)
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        # Apply letter 'a'
        next_a = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_a |= (1 << a_trans[idx])
            temp ^= lsb
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        # Apply letter 'b'
        next_b = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_b |= (1 << b_trans[idx])
            temp ^= lsb
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing
=======
import time
import random
from collections import deque

N = 12

def calculate_threshold_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using bitmask representation
    with precomputed subset transitions for speed.
    """
    start = (1 << N) - 1  # All states set
    
    # Precompute transition for all possible subsets
    # For each subset mask, compute where a and b take it
    # We'll use arrays indexed by mask
    full_mask = (1 << N) - 1
    
    # Precompute for a
    a_next = [0] * (1 << N)
    for mask in range(1, 1 << N):
        next_a = 0
        temp = mask
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_a |= (1 << a_trans[idx])
            temp ^= lsb
        a_next[mask] = next_a
    
    # Precompute for b
    b_next = [0] * (1 << N)
    for mask in range(1, 1 << N):
        next_b = 0
        temp = mask
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_b |= (1 << b_trans[idx])
            temp ^= lsb
        b_next[mask] = next_b
    
    visited = {start: 0}
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        # Check if curr is a single state (power of 2)
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        # Apply letter 'a'
        next_a = a_next[curr]
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        # Apply letter 'b'
        next_b = b_next[curr]
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing
>>>>>>> REPLACE

