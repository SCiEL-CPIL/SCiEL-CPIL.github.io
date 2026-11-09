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
    
    # Initialize matrices
    # We will build a rank 59 decomposition
    # Rank 60 schoolbook minus 8 (block) plus 7 (Strassen)
    
    # Identify the block to replace: i in [0,1], j in [0,1], k in [0,1]
    # Note: This block is valid because n>=2, m>=2, p>=2
    
    # We need to determine the total rank.
    # Let's just use a list of terms to build U, V, W
    # Each term is a tuple (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # Or simpler: accumulate into dense matrices
    
    # We don't know the final rank in advance if we want to be generic, 
    # but here it is fixed at 59.
    
    U = np.zeros((size_u, 59))
    V = np.zeros((size_v, 59))
    W = np.zeros((size_w, 59))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # 1. Add Strassen terms for block i=0,1; j=0,1; k=0,1
    # Indices:
    # A: i*m + j
    # B: j*p + k
    # C: k*n + i
    
    # z1 = (a00+a11)(b00+b11)
    # u: 0,0 and 1,1 -> 0, 5
    # v: 0,0 and 1,1 -> 0, 6
    # w: C00 (0,0)->0, C11 (1,1)->4
    add_term(0, 1, 0, 1, 0, 1) # Wait, add_term adds to a single column.
    # I need to add multiple non-zeros to the same column.
    # My add_term above only sets one index.
    # Let's change the helper or manual assignment.
    
    # Reset U, V, W
    U = np.zeros((size_u, 59))
    V = np.zeros((size_v, 59))
    W = np.zeros((size_w, 59))
    
    col = 0
    
    def set_term(u_idx_list, v_idx_list, w_idx_list):
        nonlocal col
        for idx in u_idx_list:
            U[idx, col] = 1
        for idx in v_idx_list:
            V[idx, col] = 1
        for idx in w_idx_list:
            W[idx, col] = 1
        col += 1

    # Strassen Terms
    # z1: u={0,5}, v={0,6}, w={0,4} (C00=1, C11=1)
    set_term([0, 5], [0, 6], [0, 4])
    
    # z2: (a10+a11)b00
    # u={4,5} (1,0 and 1,1), v={0} (0,0)
    # C10 = z2+... -> C10 has +z2. C11 has -z2.
    # w: C10 (1,0) -> 1*3+0=3, C11 (1,1) -> 1*3+1=4.
    # Values: W[3]=1, W[4]=-1
    # My set_term only sets 1. I need values.
    
    # Let's use a more flexible method.
    U = np.zeros((size_u, 59))
    V = np.zeros((size_v, 59))
    W = np.zeros((size_w, 59))
    col = 0
    
    def add_rank1(u_data, v_data, w_data):
        nonlocal col
        for idx, val in u_data.items():
            U[idx, col] += val
        for idx, val in v_data.items():
            V[idx, col] += val
        for idx, val in w_data.items():
            W[idx, col] += val
        col += 1

    # Strassen z1: (a00+a11)(b00+b11)
    # u: {0:1, 5:1}
    # v: {0:1, 6:1}
    # w: {0:1, 4:1}
    add_rank1({0:1, 5:1}, {0:1, 6:1}, {0:1, 4:1})
    
    # Strassen z2: (a10+a11)b00
    # u: {4:1, 5:1}
    # v: {0:1}
    # w: C10(3)=1, C11(4)=-1
    add_rank1({4:1, 5:1}, {0:1}, {3:1, 4:-1})
    
    # Strassen z3: a00(b10-b11)
    # u: {0:1}
    # v: {5:1, 6:-1} (1,0 and 1,1)
    # w: C10(3)=1, C11(4)=1
    add_rank1({0:1}, {5:1, 6:-1}, {3:1, 4:1})
    
    # Strassen z4: a11(b01+b11)
    # u: {5:1}
    # v: {1:1, 6:1} (0,1 and 1,1)
    # w: C00(0)=1, C11(4)=1
    add_rank1({5:1}, {1:1, 6:1}, {0:1, 4:1})
    
    # Strassen z5: (a00+a01)b11
    # u: {0:1, 1:1}
    # v: {6:1}
    # w: C00(0)=-1, C01(1)=1
    add_rank1({0:1, 1:1}, {6:1}, {0:-1, 1:1})
    
    # Strassen z6: (a10-a00)(b00+b01)
    # u: {4:1, 0:-1}
    # v: {0:1, 1:1}
    # w: C01(1)=1
    add_rank1({4:1, 0:-1}, {0:1, 1:1}, {1:1})
    
    # Strassen z7: (a01-a11)(b10+b11)
    # u: {1:1, 5:-1}
    # v: {5:1, 6:1}
    # w: C00(0)=-1
    add_rank1({1:1, 5:-1}, {5:1, 6:1}, {0:-1})
    
    # Now add schoolbook terms for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in the Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                add_rank1({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                
    # The total rank used is col.
    # It should be 7 + (60 - 8) = 59.
    
    # Verify
    # We should verify to be safe.
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
        # Fallback to schoolbook if Strassen construction has a bug
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
