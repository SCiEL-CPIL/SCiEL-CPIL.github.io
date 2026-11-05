def synthesize(matrix, depth_limit):
    import numpy as np
    
    m, n = matrix.shape
    mat = matrix.astype(np.uint8)
    
    gates = []
    next_wire = n + 1
    cache = {}
    
    # Initialize cache with inputs and zero
    # Wire n is the constant 0
    cache[frozenset()] = n
    for j in range(n):
        cache[frozenset([j])] = j
        
    def xor_gates(w1, w2):
        nonlocal next_wire
        # Optimization: identity elements
        if w1 == w2:
            return n
        if w1 == n:
            return w2
        if w2 == n:
            return w1
            
        # Create new gate
        gates.append([w1, w2])
        w = next_wire
        next_wire += 1
        return w

    def build(subset):
        # subset is a frozenset of input indices
        if subset in cache:
            return cache[subset]
            
        if len(subset) == 0:
            return n
        if len(subset) == 1:
            return list(subset)[0]
            
        # Split subset into two halves based on sorted order for consistent sharing
        lst = sorted(subset)
        mid = len(lst) // 2
        left = frozenset(lst[:mid])
        right = frozenset(lst[mid:])
        
        w_left = build(left)
        w_right = build(right)
        
        w_new = xor_gates(w_left, w_right)
        cache[subset] = w_new
        return w_new

    outputs = []
    for i in range(m):
        # Get the set of 1s in row i
        indices = np.where(mat[i] == 1)[0]
        subset = frozenset(indices)
        w = build(subset)
        outputs.append(w)
        
    return {'gates': gates, 'outputs': outputs}
