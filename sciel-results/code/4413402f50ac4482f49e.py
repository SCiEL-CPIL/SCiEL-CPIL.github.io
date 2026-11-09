import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # We will construct a decomposition using 4 disjoint 2x2 Strassen blocks.
    # Blocks are defined by (i_start, j_start, k_start)
    # Block size is 2x2x2.
    # Available blocks:
    # i: 0..1 (start 0)
    # j: 0..1 (start 0), 2..3 (start 2)
    # k: 0..1 (start 0), 2..3 (start 2)
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2)
    ]
    
    # Total rank estimate: 4 * 7 (strassen) + (60 - 4*8) (schoolbook) = 28 + 28 = 56
    # We will build the matrices dynamically and then trim.
    
    # To be safe, we can pre-allocate with max possible rank (schoolbook) and then trim,
    # or just build lists of terms and then convert to matrices.
    # Building lists is easier for verification and trimming.
    
    terms = []
    
    # Helper to add a Strassen block
    # Local indices:
    # a: (0,0), (0,1), (1,0), (1,1) -> global u: (i0+m*0+j0), etc
    # b: (0,0), (0,1), (1,0), (1,1) -> global v: (j0+p*0+k0), etc
    # c: (0,0), (0,1), (1,0), (1,1) -> global w: (k0+n*0+i0), etc
    
    def add_strassen_block(i0, j0, k0):
        # Map local (li, lj) to global u index
        def u_idx(li, lj):
            return (i0 + li) * m + (j0 + lj)
            
        def v_idx(lj, lk):
            return (j0 + lj) * p + (k0 + lk)
            
        def w_idx(lk, li):
            return (k0 + lk) * n + (i0 + li)
        
        # Strassen terms for 2x2
        # z1 = (a00+a11)(b00+b11)
        # u: {00:1, 11:1}, v: {00:1, 11:1}, w: {00:1, 11:1}
        # Note: w index for c_ij is k*n+i. 
        # c00 corresponds to k=0, i=0. c11 corresponds to k=1, i=1.
        # In local block:
        # c00: lk=0, li=0 -> w_idx(0,0)
        # c11: lk=1, li=1 -> w_idx(1,1)
        terms.append(({u_idx(0,0): 1, u_idx(1,1): 1}, 
                      {v_idx(0,0): 1, v_idx(1,1): 1}, 
                      {w_idx(0,0): 1, w_idx(1,1): 1}))
        
        # z2 = (a10+a11)b00
        # u: {10:1, 11:1}, v: {00:1}, w: {10:1, 11:-1}
        # c10: k=1, i=0 -> w_idx(1,0)
        # c11: k=1, i=1 -> w_idx(1,1)
        terms.append(({u_idx(1,0): 1, u_idx(1,1): 1}, 
                      {v_idx(0,0): 1}, 
                      {w_idx(1,0): 1, w_idx(1,1): -1}))
                      
        # z3 = a00(b10-b11)
        # u: {00:1}, v: {10:1, 11:-1}, w: {10:1, 11:1}
        # c10: k=1, i=0 -> w_idx(1,0)
        # c11: k=1, i=1 -> w_idx(1,1)
        terms.append(({u_idx(0,0): 1}, 
                      {v_idx(1,0): 1, v_idx(1,1): -1}, 
                      {w_idx(1,0): 1, w_idx(1,1): 1}))
                      
        # z4 = a11(b01+b11)
        # u: {11:1}, v: {01:1, 11:1}, w: {00:1, 11:1}
        # c00: k=0, i=0 -> w_idx(0,0)
        # c11: k=1, i=1 -> w_idx(1,1)
        terms.append(({u_idx(1,1): 1}, 
                      {v_idx(0,1): 1, v_idx(1,1): 1}, 
                      {w_idx(0,0): 1, w_idx(1,1): 1}))
                      
        # z5 = (a00+a01)b11
        # u: {00:1, 01:1}, v: {11:1}, w: {00:-1, 01:1}
        # c00: k=0, i=0 -> w_idx(0,0)
        # c01: k=0, i=1 -> w_idx(0,1)
        terms.append(({u_idx(0,0): 1, u_idx(0,1): 1}, 
                      {v_idx(1,1): 1}, 
                      {w_idx(0,0): -1, w_idx(0,1): 1}))
                      
        # z6 = (a10-a00)(b00+b01)
        # u: {10:1, 00:-1}, v: {00:1, 01:1}, w: {01:1}
        # c01: k=0, i=1 -> w_idx(0,1)
        terms.append(({u_idx(1,0): 1, u_idx(0,0): -1}, 
                      {v_idx(0,0): 1, v_idx(0,1): 1}, 
                      {w_idx(0,1): 1}))
                      
        # z7 = (a01-a11)(b10+b11)
        # u: {01:1, 11:-1}, v: {10:1, 11:1}, w: {00:-1}
        # c00: k=0, i=0 -> w_idx(0,0)
        terms.append(({u_idx(0,1): 1, u_idx(1,1): -1}, 
                      {v_idx(1,0): 1, v_idx(1,1): 1}, 
                      {w_idx(0,0): -1}))

    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0)
        
    # Track which (i,j,k) are covered by Strassen blocks
    covered = set()
    for i0, j0, k0 in blocks:
        for i in range(i0, i0+2):
            for j in range(j0, j0+2):
                for k in range(k0, k0+2):
                    covered.add((i, j, k))
                    
    # Add schoolbook terms for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
                
    rank = len(terms)
    
    # Build U, V, W matrices
    U = np.zeros((size_u, rank))
    V = np.zeros((size_v, rank))
    W = np.zeros((size_w, rank))
    
    for c, (u_data, v_data, w_data) in enumerate(terms):
        for idx, val in u_data.items():
            U[idx, c] += val
        for idx, val in v_data.items():
            V[idx, c] += val
        for idx, val in w_data.items():
            W[idx, c] += val
            
    # Verify
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    target = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target[u, v, w] = 1.0
                
    if not np.allclose(recon, target, atol=1e-9):
        # Fallback to schoolbook
        rank = rank_schoolbook
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, c] = 1
                    V[v, c] = 1
                    W[w, c] = 1
                    c += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
