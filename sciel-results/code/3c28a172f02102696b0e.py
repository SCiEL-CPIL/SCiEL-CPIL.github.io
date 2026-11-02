import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 42 decomposition using Strassen's algorithm for each 2x2 block
    # There are p/2 = 6 blocks of 2 columns
    num_blocks = p // 2
    rank = num_blocks * 7
    R = rank
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen terms for 2x2x2 multiplication
    # Indices: U[i,j], V[j,k], W[k,i]
    # Term 1: (a+d)(e+g) -> C00 + C11
    # Term 2: (c+d)e -> C10
    # Term 3: a(f-g) -> C01
    # Term 4: d(h-f) -> C00 + C10
    # Term 5: (a+b)(h+g) -> C00 - C01
    # Term 6: (c-a)(e+f) -> C11
    # Term 7: (b-d)(g+h) -> C00 + C11
    
    # We need to map these to U, V, W vectors.
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Term 1: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
    # This corresponds to (a+d)(e+g) = ae + ag + de + dg
    # C00 gets ae, dg? No.
    # P1 = (a+d)(e+g).
    # Contribution to C00 is P1 + P4 - P5 + P7.
    
    # Let's use the explicit linear combinations.
    # We define 7 rank-1 tensors.
    
    terms = [
        # U, V, W as lists of (index, value)
        # U index: i*m + j
        # V index: j*p + k (local k 0,1)
        # W index: k*n + i (local k 0,1)
        
        # P1 = (a+d)(e+g)
        {
            'U': [(0, 1), (3, 1)], # a, d
            'V': [(0, 1), (3, 1)], # e, g  (j=0,k=0; j=1,k=1) -> 0, 3
            'W': [(0, 1), (3, 1)]  # C00 (k=0,i=0), C11 (k=1,i=1) -> 0, 3
        },
        # P2 = (c+d)e
        {
            'U': [(2, 1), (3, 1)], # c, d
            'V': [(0, 1)],         # e
            'W': [(2, 1)]          # C10 (k=0, i=1) -> 2
        },
        # P3 = a(f-g)
        {
            'U': [(0, 1)],         # a
            'V': [(1, 1), (3, -1)],# f, -g (j=0,k=1; j=1,k=1) -> 1, 3
            'W': [(1, 1)]          # C01 (k=1, i=0) -> 1
        },
        # P4 = d(h-f)
        {
            'U': [(3, 1)],         # d
            'V': [(3, 1), (2, -1)],# h, -f (j=1,k=1; j=1,k=0) -> 3, 2
            'W': [(0, 1), (2, 1)]  # C00 (k=0,i=0), C10 (k=0,i=1) -> 0, 2
        },
        # P5 = (a+b)(h+g)
        {
            'U': [(0, 1), (1, 1)], # a, b
            'V': [(3, 1)],         # h+g? No, V is j,k.
            # Wait, P5 = (a+b)(h+g).
            # a is (0,0), b is (0,1).
            # h is (1,1), g is (1,0)?
            # B matrix: [[e, f], [g, h]].
            # j=0: e, f. j=1: g, h.
            # g is (1,0) -> V index 2.
            # h is (1,1) -> V index 3.
            # So V should have 2 and 3.
            'V': [(2, 1), (3, 1)], # g, h
            'W': [(0, -1), (1, 1)] # C00 - C01?
            # C00 = ... - P5. C01 = ... + P5.
            # So W[0,0] = -1, W[0,1] = 1.
        },
        # P6 = (c-a)(e+f)
        {
            'U': [(2, 1), (0, -1)], # c, -a
            'V': [(0, 1), (1, 1)],  # e, f
            'W': [(3, 1)]           # C11
        },
        # P7 = (b-d)(g+h)
        {
            'U': [(1, 1), (3, -1)], # b, -d
            'V': [(2, 1), (3, 1)],  # g, h
            'W': [(0, 1), (3, 1)]   # C00, C11
        }
    ]
    
    for b in range(num_blocks):
        for t_idx in range(7):
            r = b * 7 + t_idx
            term = terms[t_idx]
            
            for (idx, val) in term['U']:
                U[idx, r] = val
                
            for (v_idx, val) in term['V']:
                # v_idx is local j*p + k? No, I used 0,1,2,3 for 2x2.
                # 0: j=0, k=0
                # 1: j=0, k=1
                # 2: j=1, k=0
                # 3: j=1, k=1
                j = v_idx // 2
                k_loc = v_idx % 2
                k_glob = b * 2 + k_loc
                v_glob = j * p + k_glob
                V[v_glob, r] = val
                
            for (w_idx, val) in term['W']:
                # w_idx is local k*n + i?
                # 0: k=0, i=0
                # 1: k=0, i=1
                # 2: k=1, i=0
                # 3: k=1, i=1
                k_loc = w_idx // 2
                i = w_idx % 2
                k_glob = b * 2 + k_loc
                w_glob = k_glob * n + i
                W[w_glob, r] = val

    return {"U": U, "V": V, "W": W}
