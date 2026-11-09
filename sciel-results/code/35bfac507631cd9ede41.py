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
    
    # We will use Strassen for 4 blocks of 2x2x2
    # i in {0, 1}
    # j in {0, 1, 2, 3}
    # k in {0, 1, 2, 3}
    #
    # Blocks:
    # 1. j in {0,1}, k in {0,1}
    # 2. j in {2,3}, k in {0,1}
    # 3. j in {0,1}, k in {2,3}
    # 4. j in {2,3}, k in {2,3}
    #
    # All other terms use schoolbook.
    #
    # Total Strassen rank: 4 * 7 = 28
    # Remaining terms:
    # i in {0,1}, k=4: 2 * 4 * 1 = 8
    # i=2, all j,k: 1 * 4 * 5 = 20
    # Total Rank: 28 + 8 + 20 = 56
    
    U = np.zeros((size_u, 56))
    V = np.zeros((size_v, 56))
    W = np.zeros((size_w, 56))
    
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

    def add_strassen_block(i_offset, j_offset, k_offset):
        """
        Adds Strassen terms for a 2x2x2 block starting at (i_offset, j_offset, k_offset).
        Assumes i_offset is 0 (since we only use i in {0,1}).
        """
        nonlocal col
        
        # Local indices to Global
        # i_glob = i_offset + i_loc
        # j_glob = j_offset + j_loc
        # k_glob = k_offset + k_loc
        
        # u_idx = i_glob * m + j_glob
        # v_idx = j_glob * p + k_glob
        # w_idx = k_glob * n + i_glob
        
        # Strassen Terms:
        # z1: (a00+a11)(b00+b11)
        # u: (0,0), (1,1) -> u_idx(0,0), u_idx(1,1)
        # v: (0,0), (1,1) -> v_idx(0,0), v_idx(1,1)
        # w: (0,0), (1,1) -> w_idx(0,0), w_idx(1,1)
        
        u00 = (i_offset + 0) * m + (j_offset + 0)
        u11 = (i_offset + 1) * m + (j_offset + 1)
        u10 = (i_offset + 1) * m + (j_offset + 0)
        u01 = (i_offset + 0) * m + (j_offset + 1)
        
        v00 = (j_offset + 0) * p + (k_offset + 0)
        v11 = (j_offset + 1) * p + (k_offset + 1)
        v10 = (j_offset + 1) * p + (k_offset + 0)
        v01 = (j_offset + 0) * p + (k_offset + 1)
        
        w00 = (k_offset + 0) * n + (i_offset + 0)
        w11 = (k_offset + 1) * n + (i_offset + 1)
        w10 = (k_offset + 0) * n + (i_offset + 1)
        w01 = (k_offset + 1) * n + (i_offset + 0)
        
        # z1
        add_rank1({u00: 1, u11: 1}, {v00: 1, v11: 1}, {w00: 1, w11: 1})
        
        # z2: (a10+a11)b00
        # u: u10, u11
        # v: v00
        # w: w10 (1), w11 (-1)
        add_rank1({u10: 1, u11: 1}, {v00: 1}, {w10: 1, w11: -1})
        
        # z3: a00(b10-b11)
        # u: u00
        # v: v10 (1), v11 (-1)
        # w: w10 (1), w11 (1)
        add_rank1({u00: 1}, {v10: 1, v11: -1}, {w10: 1, w11: 1})
        
        # z4: a11(b01+b11)
        # u: u11
        # v: v01 (1), v11 (1)
        # w: w00 (1), w11 (1)
        add_rank1({u11: 1}, {v01: 1, v11: 1}, {w00: 1, w11: 1})
        
        # z5: (a00+a01)b11
        # u: u00 (1), u01 (1)
        # v: v11
        # w: w00 (-1), w01 (1)
        add_rank1({u00: 1, u01: 1}, {v11: 1}, {w00: -1, w01: 1})
        
        # z6: (a10-a00)(b00+b01)
        # u: u10 (1), u00 (-1)
        # v: v00 (1), v01 (1)
        # w: w01 (1)
        add_rank1({u10: 1, u00: -1}, {v00: 1, v01: 1}, {w01: 1})
        
        # z7: (a01-a11)(b10+b11)
        # u: u01 (1), u11 (-1)
        # v: v10 (1), v11 (1)
        # w: w00 (-1)
        add_rank1({u01: 1, u11: -1}, {v10: 1, v11: 1}, {w00: -1})

    # Add the 4 Strassen blocks
    # i_offset is always 0
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 2)
    
    # Add schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is covered by a Strassen block
                # Strassen covers:
                # i in {0, 1}
                # j in {0, 1, 2, 3}
                # k in {0, 1, 2, 3}
                
                covered = False
                if i < 2 and j < 4 and k < 4:
                    # It is in the 2x4x4 region covered by Strassen
                    # We need to ensure it was actually included in one of the 4 blocks.
                    # Since the blocks partition j in {0,1,2,3} and k in {0,1,2,3},
                    # all such (i,j,k) are covered.
                    covered = True
                
                if not covered:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_rank1({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
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
