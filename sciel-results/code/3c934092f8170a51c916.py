<<<<<<< SEARCH
def _compute_threshold(trans):
    """
    Compute exact shortest reset word length via BFS on subset automaton.
    Optimized with precomputed subset transitions.
    trans: list of two lists, trans[letter][state] = dest state.
    Returns (threshold_length, is_synchronizing).
    """
    full = (1 << N) - 1  # bitmask for all states
    num_subsets = 1 << N
    
    # Precompute transitions for all subsets for both letters
    # next_subset[l][mask] = image of mask under letter l
    next_subset = [[0] * num_subsets for _ in range(2)]
    
    for l in range(2):
        # dest_bit[s] is the bitmask for the destination of state s under letter l
        dest_bit = [1 << trans[l][s] for s in range(N)]
        
        for mask in range(num_subsets):
            new_mask = 0
            # Efficiently compute image: iterate set bits
            m = mask
            while m:
                bit = m & (-m)
                s = bit.bit_length() - 1
                new_mask |= dest_bit[s]
                m ^= bit
            next_subset[l][mask] = new_mask
    
    # BFS from full set
    # visited_dist[mask] = distance
    # Use a list for visited distances, initialized to -1
    dist = [-1] * num_subsets
    dist[full] = 0
    queue = deque([full])
    
    while queue:
        subset = queue.popleft()
        d = dist[subset]
        # Check if singleton
        if subset & (subset - 1) == 0:  # power of 2
            return (d, True)
        for l in range(2):
            new_subset = next_subset[l][subset]
            if dist[new_subset] == -1:
                dist[new_subset] = d + 1
                queue.append(new_subset)
    
    return (-1, False)
=======
def _compute_threshold(trans):
    """
    Compute exact shortest reset word length via BFS on subset automaton.
    Optimized for speed: computes transitions on-the-fly to avoid large precomputation overhead
    which is expensive when called frequently with different 'trans'.
    """
    full = (1 << N) - 1  # bitmask for all states
    num_subsets = 1 << N
    
    # Precompute destination bits for fast lookup
    dest_bits = [[1 << trans[l][s] for s in range(N)] for l in range(2)]
    
    # BFS from full set
    dist = [-1] * num_subsets
    dist[full] = 0
    queue = deque([full])
    
    while queue:
        subset = queue.popleft()
        d = dist[subset]
        # Check if singleton
        if subset & (subset - 1) == 0:  # power of 2
            return (d, True)
        
        # Calculate next subsets on the fly
        for l in range(2):
            new_mask = 0
            m = subset
            while m:
                bit = m & (-m)
                s = bit.bit_length() - 1
                new_mask |= dest_bits[l][s]
                m ^= bit
            if dist[new_mask] == -1:
                dist[new_mask] = d + 1
                queue.append(new_mask)
    
    return (-1, False)
>>>>>>> REPLACE

