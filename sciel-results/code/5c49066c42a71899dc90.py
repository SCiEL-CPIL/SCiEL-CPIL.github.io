import random
import itertools
import numpy as np

N = 16
M = N * N

# Precompute conflict masks
# conflict_mask[i][j] is a bitset of points k such that i,j,k form an isosceles triangle
# We only need i < j
conflict_mask = [[0] * M for _ in range(M)]

def dist_sq(p, q):
    return (p[0] - q[0])**2 + (p[1] - q[1])**2

# Generate all points
points = [(x, y) for y in range(N) for x in range(N)]
point_to_idx = {p: i for i, p in enumerate(points)}

# Precompute distances between all pairs? Or just compute on the fly for masks.
# M=256, pairs ~ 32k. Dist calc is cheap.

for i in range(M):
    for j in range(i + 1, M):
        p_i = points[i]
        p_j = points[j]
        d_ij = dist_sq(p_i, p_j)
        mask = 0
        for k in range(M):
            if k == i or k == j:
                continue
            p_k = points[k]
            d_ik = dist_sq(p_i, p_k)
            d_jk = dist_sq(p_j, p_k)
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                mask |= (1 << k)
        conflict_mask[i][j] = mask
        conflict_mask[j][i] = mask

def is_valid(S_mask, new_point_idx):
    """Check if adding new_point_idx to S_mask keeps it no-isosceles."""
    # Check if new_point forms isosceles with any pair in S_mask
    # Or simpler: Check if new_point is in conflict_mask[i][j] for any i,j in S_mask
    # And also check if new_point forms isosceles with i, j? 
    # Actually, if S_mask is valid, we just need to check if new_point creates a new isosceles.
    # This happens if there exist i, j in S_mask such that {i, j, new_point} is isosceles.
    # This is equivalent to: new_point is in conflict_mask[i][j] for some i, j in S_mask.
    # Wait, conflict_mask[i][j] contains k if i,j,k is isosceles.
    # So we need to check if bit new_point_idx is set in any conflict_mask[i][j] where i,j in S_mask.
    
    # Optimization: Iterate through set bits in S_mask
    s = S_mask
    while s:
        lsb = s & -s
        i = lsb.bit_length() - 1
        s ^= lsb
        # For each i in S, check all j in S (j > i to avoid dup, but we can just check all)
        # Actually, we can precompute for each point k, the list of pairs?
        # Let's just iterate. |S| is small.
        s2 = S_mask
        while s2:
            lsb2 = s2 & -s2
            j = lsb2.bit_length() - 1
            s2 ^= lsb2
            if i == j:
                continue
            if (conflict_mask[i][j] >> new_point_idx) & 1:
                return False
    return True

def greedy_construct(order):
    S_mask = 0
    for idx in order:
        if is_valid(S_mask, idx):
            S_mask |= (1 << idx)
    return S_mask

def is_maximal(S_mask):
    """Check if S_mask is a maximal no-isosceles set."""
    # 1. S_mask must be valid (assumed)
    # 2. Every point not in S_mask must be covered by some pair in S_mask
    #    i.e., for all k not in S_mask, exists i,j in S_mask such that bit k in conflict_mask[i][j]
    
    cover = 0
    s = S_mask
    while s:
        lsb = s & -s
        i = lsb.bit_length() - 1
        s ^= lsb
        s2 = S_mask
        while s2:
            lsb2 = s2 & -s2
            j = lsb2.bit_length() - 1
            s2 ^= lsb2
            if i == j:
                continue
            cover |= conflict_mask[i][j]
            
    # Points not in S_mask
    not_in_S = (~S_mask) & ((1 << M) - 1)
    # Uncovered points
    uncovered = not_in_S & ~cover
    return uncovered == 0

def minimize_maximal(S_mask):
    """Try to remove points while maintaining maximality."""
    changed = True
    while changed:
        changed = False
        # Get list of indices in S_mask
        s = S_mask
        indices = []
        while s:
            lsb = s & -s
            idx = lsb.bit_length() - 1
            s ^= lsb
            indices.append(idx)
        
        random.shuffle(indices)
        
        for idx in indices:
            S_new = S_mask & ~(1 << idx)
            if is_maximal(S_new):
                S_mask = S_new
                changed = True
                # Restart loop with new set
                break
    return S_mask

def construct():
    best_S = None
    best_size = M + 1
    
    # Run multiple times with different random seeds/orders
    for _ in range(20):
        order = list(range(M))
        random.shuffle(order)
        S = greedy_construct(order)
        S = minimize_maximal(S)
        
        size = bin(S).count('1')
        if size < best_size:
            best_size = size
            best_S = S
            
    # Convert best_S to list of points
    result = []
    s = best_S
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        s ^= lsb
        result.append(points[idx])
        
    return result
