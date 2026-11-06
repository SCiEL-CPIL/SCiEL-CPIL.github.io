<<<<<<< SEARCH
import time
import random

N = 12
SIZE = 1 << N

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    """
    # Precompute transition for each subset for letter a and b
    size = SIZE
    trans_a_sub = [0] * size
    trans_b_sub = [0] * size
    
    for subset in range(size):
        if subset == 0:
            continue
        sa = 0
        sb = 0
        temp = subset
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= (1 << trans_a[i])
            sb |= (1 << trans_b[i])
            temp ^= lsb
        trans_a_sub[subset] = sa
        trans_b_sub[subset] = sb

    start = (1 << N) - 1
    if start == 1:
        return 0
        
    visited = [False] * size
    queue = [start]
    visited[start] = True
    dist = 0
    
    while queue:
        new_queue = []
        for subset in queue:
            # Check if singleton
            if subset & (subset - 1) == 0:
                return dist
            
            # Apply a
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                new_queue.append(next_a)
            
            # Apply b
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                new_queue.append(next_b)
        
        queue = new_queue
        dist += 1
        
    return -1
=======
import time
import random

N = 12
SIZE = 1 << N

# Precompute single-bit masks for fast subset operations
_SINGLETON_MASKS = [1 << i for i in range(N)]

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    Optimized: precompute subset transitions incrementally, use bytearray for visited.
    """
    # Precompute transition for each subset for letter a and b
    # Use a more efficient approach: build up from smaller subsets
    trans_a_sub = [0] * SIZE
    trans_b_sub = [0] * SIZE
    
    # For each state, precompute its contribution
    # trans_a[i] = destination of state i under letter a
    # For subset S, trans_a_sub[S] = union of {trans_a[i] for i in S}
    
    # Efficient computation: for each subset, use the lowest set bit
    for subset in range(1, SIZE):
        lsb = subset & (-subset)
        i = lsb.bit_length() - 1
        rest = subset ^ lsb
        if rest:
            trans_a_sub[subset] = trans_a_sub[rest] | (1 << trans_a[i])
            trans_b_sub[subset] = trans_b_sub[rest] | (1 << trans_b[i])
        else:
            trans_a_sub[subset] = 1 << trans_a[i]
            trans_b_sub[subset] = 1 << trans_b[i]

    start = (1 << N) - 1
    # Check if already singleton (N=1 case, but N=12 so no)
    if start & (start - 1) == 0:
        return 0
        
    # Use bytearray for visited (faster than list of bools)
    visited = bytearray(SIZE)
    visited[start] = 1
    
    # Use two lists for BFS levels
    current_level = [start]
    dist = 0
    
    while current_level:
        next_level = []
        for subset in current_level:
            # Apply a
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = 1
                # Early check: is it a singleton?
                if next_a & (next_a - 1) == 0:
                    return dist + 1
                next_level.append(next_a)
            
            # Apply b
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = 1
                if next_b & (next_b - 1) == 0:
                    return dist + 1
                next_level.append(next_b)
        
        current_level = next_level
        dist += 1
        
    return -1
>>>>>>> REPLACE

