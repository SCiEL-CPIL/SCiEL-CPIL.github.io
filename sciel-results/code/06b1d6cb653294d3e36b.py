import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _deadline_exceeded(start_time, limit=145):
    return (time.time() - start_time) > limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Use 2+2+1 block partition for 5x5 matrix multiplication
    block_sizes = [2, 2, 1]
    
    def block_offset(sizes, idx):
        offset = 0
        for i in range(idx):
            offset += sizes[i]
        return offset
    
    # Precompute block offsets
    i_offsets = [block_offset(block_sizes, i) for i in range(3)]
    j_offsets = [block_offset(block_sizes, j) for j in range(3)]
    k_offsets = [block_offset(block_sizes, k) for k in range(3)]
    
    U_rows = []
    V_rows = []
    W_rows = []
    rank_counter = 0
    
    def add_schoolblock(i_blk, j_blk, k_blk):
        """Add schoolbook decomposition for a block"""
        nonlocal rank_counter
        si = block_sizes[i_blk]
        sj = block_sizes[j_blk]
        sk = block_sizes[k_blk]
        
        io = i_offsets[i_blk]
        jo = j_offsets[j_blk]
        ko = k_offsets[k_blk]
        
        for a in range(si):
            for b in range(sj):
                for c in range(sk):
                    r = rank_counter
                    rank_counter += 1
                    
                    i_global = io + a
                    j_global = jo + b
                    k_global = ko + c
                    
                    u_idx = i_global * m + j_global
                    v_idx = j_global * p + k_global
                    w_idx = k_global * n + i_global
                    
                    U_rows.append((u_idx, r, 1.0))
                    V_rows.append((v_idx, r, 1.0))
                    W_rows.append((w_idx, r, 1.0))
    
    def add_strassen_2x2(i_blk, j_blk, k_blk):
        """Add Strassen decomposition for 2x2x2 block"""
        nonlocal rank_counter
        io = i_offsets[i_blk]
        jo = j_offsets[j_blk]
        ko = k_offsets[k_blk]
        
        def gi(li): return io + li
        def gj(lj): return jo + lj
        def gk(lk): return ko + lk
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        r = rank_counter; rank_counter += 1
        U_rows.append((gi(0)*m + gj(0), r, 1.0))
        U_rows.append((gi(1)*m + gj(1), r, 1.0))
        V_rows.append((gj(0)*p + gk(0), r, 1.0))
        V_rows.append((gj(1)*p + gk(1), r, 1.0))
        W_rows.append((gk(0)*n + gi(0), r, 1.0))
        W_rows.append((gk(1)*n + gi(1), r, 1.0))
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        r = rank_counter; rank_counter += 1
        U_rows.append((gi(1)*m + gj(0), r, 1.0))
        U_rows.append((gi(1)*m + gj(1), r, 1.0))
        V_rows.append((gj(0)*p + gk(0), r, 1.0))
        W_rows.append((gk(0)*n + gi(1), r, 1.0))
        W_rows.append((gk(1)*n + gi(1), r, -1.0))
        
        # M3 = A[0,0]*(B[1,0]-B[1,1])
        r = rank_counter; rank_counter += 1
        U_rows.append((gi(0)*m + gj(0), r, 1.0))
        V_rows.append((gj(1)*p + gk(0), r, 1.0))
        V_rows.append((gj(1)*p + gk(1), r, -1.0))
        W_rows.append((gk(0)*n + gi(0), r, 1.0))
        W_rows.append((gk(1)*n + gi(1), r, 1.0))
        
        # M4 = A[1,1]*(B[0,1]-B[0,0])
        r = rank_counter; rank_counter += 1
        U_rows.append((gi(1)*m + gj(1), r, 1.0))
        V_rows.append((gj(0)*p + gk(1), r, 1.0))
        V_rows.append((gj(0)*p + gk(0), r, -1.0))
        W_rows.append((gk(0)*n + gi(1), r, 1.0))
        W_rows.append((gk(1)*n + gi(1), r, 1.0))
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        r = rank_counter; rank_counter += 1
        U_rows.append((gi(0)*m + gj(0), r, 1.0))
        U_rows.append((gi(0)*m + gj(1), r, 1.0))
        V_rows.append((gj(1)*p + gk(1), r, 1.0))
        W_rows.append((gk(0)*n + gi(0), r, -1.0))
        W_rows.append((gk(1)*n + gi(0), r, 1.0))
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        r = rank_counter; rank_counter += 1
        U_rows.append((gi(1)*m + gj(0), r, 1.0))
        U_rows.append((gi(0)*m + gj(0), r, -1.0))
        V_rows.append((gj(0)*p + gk(0), r, 1.0))
        V_rows.append((gj(0)*p + gk(1), r, 1.0))
        W_rows.append((gk(1)*n + gi(1), r, 1.0))
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        r = rank_counter; rank_counter += 1
        U_rows.append((gi(0)*m + gj(1), r, 1.0))
        U_rows.append((gi(1)*m + gj(1), r, -1.0))
        V_rows.append((gj(1)*p + gk(0), r, 1.0))
        V_rows.append((gj(1)*p + gk(1), r, 1.0))
        W_rows.append((gk(0)*n + gi(0), r, 1.0))
        W_rows.append((gk(1)*n + gi(1), r, 1.0))
    
    # Iterate over all block triples
    for i_blk in range(3):
        for j_blk in range(3):
            for k_blk in range(3):
                if _deadline_exceeded(start_time):
                    rank = n * m * p
                    U = np.zeros((n * m, rank), dtype=float)
                    V = np.zeros((m * p, rank), dtype=float)
                    W = np.zeros((p * n, rank), dtype=float)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                r = i * (m * p) + j * p + k
                                U[i * m + j, r] = 1.0
                                V[j * p + k, r] = 1.0
                                W[k * n + i, r] = 1.0
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                si = block_sizes[i_blk]
                sj = block_sizes[j_blk]
                sk = block_sizes[k_blk]
                
                if si == 2 and sj == 2 and sk == 2:
                    add_strassen_2x2(i_blk, j_blk, k_blk)
                else:
                    add_schoolblock(i_blk, j_blk, k_blk)
    
    # Build the matrices
    rank = rank_counter
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for (idx, r, val) in U_rows:
        U[idx, r] += val
    for (idx, r, val) in V_rows:
        V[idx, r] += val
    for (idx, r, val) in W_rows:
        W[idx, r] += val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
