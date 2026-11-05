import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140  # safety margin before 150s kill

def _check_deadline():
    return time.time() - _START_TIME > _DEADLINE

def construct():
    n, m, p = DIMENSIONS
    # M(3,4,6) = M(3,4,2) + M(3,4,2) + M(3,4,2) with p split 2+2+2
    # Each M(3,4,2) = M(1,4,2) + M(2,4,2)
    # M(1,4,2) = schoolbook rank 8
    # M(2,4,2) = 2 * M(2,2,2) = 2 * Strassen(7) = 14
    # Total: 3 * (8 + 14) = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    # Store rank-1 tensors as tuples of (U_vec, V_vec, W_vec) to check for duplicates
    tensors = []
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        
        # Create sparse vectors
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, val in zip(u_indices, u_vals):
            u_vec[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v_vec[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w_vec[idx] += val
            
        # Store as tuple of tuples for hashing
        tensors.append((tuple(u_vec), tuple(v_vec), tuple(w_vec)))
        current_rank += 1

    def add_1_4_2(i_off, j_off, k_off):
        # n=1, m=4, p=2
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # m1 = (a00+a11)(b00+b11) -> c00, c11
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        # m2 = (a10+a11)(b00) -> c10, c11
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        # m3 = (a00)(b01-b11) -> c01, c11
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        # m4 = (a11)(b10-b00) -> c00, c10
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        # m5 = (a00+a01)(b11) -> c00, c01
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        # m6 = (a10-a00)(b00+b01) -> c11
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        # m7 = (a01-a11)(b10+b11) -> c00
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
    
    def add_2_4_2(i_off, j_off, k_off):
        # M(2,4,2) = M(2,2,2) + M(2,2,2) with m split 2+2
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    # Build three M(3,4,2) blocks
    for block_k in range(3):
        if _check_deadline():
            break
        k_off = block_k * 2
        # M(1,4,2) at n=0
        add_1_4_2(0, 0, k_off)
        # M(2,4,2) at n=1
        add_2_4_2(1, 0, k_off)
        
    # Post-processing: Combine identical tensors and cancel opposites
    # Count occurrences
    counts = defaultdict(int)
    for t in tensors:
        counts[t] += 1
        
    # Create a dictionary of tensor -> coefficient (integer)
    tensor_coeffs = defaultdict(int)
    for t in tensors:
        tensor_coeffs[t] += 1
        
    # Check for negatives
    # If t is in tensor_coeffs, and -t is in tensor_coeffs
    # We need to define -t
    def neg_tensor(t):
        u, v, w = t
        return (tuple(-x for x in u), tuple(-x for x in v), tuple(-x for x in w))
        
    # Process
    final_coeffs = {}
    processed = set()
    for t, c in tensor_coeffs.items():
        if t in processed:
            continue
        neg_t = neg_tensor(t)
        if neg_t in tensor_coeffs:
            neg_c = tensor_coeffs[neg_t]
            # Net coefficient for t is c - neg_c?
            # Sum = c*t + neg_c*(-t) = (c - neg_c)*t
            net_c = c - neg_c
            if net_c != 0:
                final_coeffs[t] = net_c
            processed.add(t)
            processed.add(neg_t)
        else:
            if c != 0:
                final_coeffs[t] = c
            processed.add(t)
            
    # Now, final_coeffs contains {tensor: coefficient}
    # Each entry is a rank-1 term with that coefficient.
    # The rank is len(final_coeffs)
    
    new_rank = len(final_coeffs)
    
    # Reconstruct U, V, W
    U_new = [[0] * new_rank for _ in range(n * m)]
    V_new = [[0] * new_rank for _ in range(m * p)]
    W_new = [[0] * new_rank for _ in range(p * n)]
    
    for idx, (t, coeff) in enumerate(final_coeffs.items()):
        u, v, w = t
        for i in range(n * m):
            U_new[i][idx] = u[i] * coeff
        for i in range(m * p):
            V_new[i][idx] = v[i] * coeff
        for i in range(p * n):
            W_new[i][idx] = w[i] * coeff
            
    return {"U": U_new, "V": V_new, "W": W_new}
