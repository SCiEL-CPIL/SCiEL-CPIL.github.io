import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Generate reduced candidate pool:
    # Weight 4 vectors, first non-zero entry (smallest index) is positive.
    candidates = []
    
    for idxs in itertools.combinations(range(D), 4):
        i1, i2, i3, i4 = idxs
        # v[i1] = 1
        # v[i2], v[i3], v[i4] in {1, -1}
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                for s4 in [1, -1]:
                    v = [0] * D
                    v[i1] = 1
                    v[i2] = s2
                    v[i3] = s3
                    v[i4] = s4
                    candidates.append(v)

    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # Build conflict graph
    M = candidates @ candidates.T
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    deg = Conflict.sum(axis=1)
    order = np.argsort(-deg)
    
    # Precompute neighbors as bitmasks
    neighbors = [0] * N
    for i in range(N):
        mask = 0
        js = np.where(Conflict[i] == 1)[0]
        for j in js:
            mask |= (1 << j)
        neighbors[i] = mask
    
    # Greedy coloring for upper bound
    def greedy_color_bound(A):
        if A == 0:
            return 0
        
        colors = 0
        remaining = A
        while remaining:
            colors += 1
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            remaining &= ~(v_bit | neighbors[v])
        return colors

    sys.setrecursionlimit(10000)
    
    best_size = 0
    best_I = 0
    
    # Greedy initial solution
    def greedy_mis():
        A = (1 << N) - 1
        I = 0
        size = 0
        while A:
            # Pick vertex with lowest index in A for speed
            v_bit = A & (-A)
            v = v_bit.bit_length() - 1
            I |= v_bit
            size += 1
            A &= ~(v_bit | neighbors[v])
        return I, size

    init_I, init_size = greedy_mis()
    best_size = init_size
    best_I = init_I

    def branch_and_bound(A, I, size):
        nonlocal best_size, best_I
        
        if time.time() > deadline:
            return
        
        if A == 0:
            if size > best_size:
                best_size = size
                best_I = I
            return
        
        ub = size + greedy_color_bound(A)
        if ub <= best_size:
            return
        
        # Select v from static order
        v = -1
        for u in order:
            if A & (1 << u):
                v = u
                break
        
        if v == -1:
            return
            
        # Include v
        new_A = A & ~( (1 << v) | neighbors[v] )
        new_I = I | (1 << v)
        branch_and_bound(new_A, new_I, size + 1)
        
        # Exclude v
        new_A2 = A & ~(1 << v)
        branch_and_bound(new_A2, new_I, size)

    full_A = (1 << N) - 1
    branch_and_bound(full_A, 0, 0)
    
    selected_indices = []
    for i in range(N):
        if best_I & (1 << i):
            selected_indices.append(i)
            
    if not selected_indices:
        selected_indices = [0]
        
    return candidates[selected_indices].astype(np.int64)
