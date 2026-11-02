import numpy as np
import time

def construct():
    """
    Returns a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses block-wise Strassen algorithm, splitting the 2x14 matrix into seven 2x2 blocks.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 2, 14
    
    # Number of 2x2 blocks in the 2x14 matrix B
    num_blocks = p // 2
    # Rank of 2x2 matrix multiplication (Strassen)
    strassen_rank = 7
    total_rank = num_blocks * strassen_rank
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 49)
    # V: (m*p, rank) = (28, 49)
    # W: (p*n, rank) = (28, 49)
    U = np.zeros((n * m, total_rank), dtype=np.int32)
    V = np.zeros((m * p, total_rank), dtype=np.int32)
    W = np.zeros((p * n, total_rank), dtype=np.int32)
    
    # Strassen decomposition for 2x2 * 2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: 0=A00, 1=A01, 2=A10, 3=A11
    # Indices for B: 0=B00, 1=B01, 2=B10, 3=B11 (local to block)
    # Indices for C: 0=C00, 1=C01, 2=C10, 3=C11 (local to block)
    
    # The 7 products and their contributions to U, V, W
    # Each entry is (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
    # Note: V and W indices are local to the 2x2 block, we will offset them by k_offset * 2
    
    strassen_patterns = [
        # P1 = (A00 + A11) * (B00 + B11)
        # C00 += P1, C11 += P1
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        # P2 = (A10 + A11) * B00
        # C10 += P2, C11 -= P2
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, -1)]
        },
        # P3 = A00 * (B01 - B11)
        # C01 += P3, C11 += P3
        {
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(1, 1), (3, 1)]
        },
        # P4 = A11 * (B10 - B00)
        # C00 += P4, C10 += P4
        {
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(0, 1), (2, 1)]
        },
        # P5 = (A00 + A01) * B11
        # C00 -= P5, C01 += P5
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)]
        },
        # P6 = (A10 - A00) * (B00 + B01)
        # C11 += P6
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        # P7 = (A01 - A11) * (B10 + B11)
        # C00 += P7
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)]
        }
    ]

    for t in range(num_blocks):
        k_offset = t * 2
        
        for s in range(strassen_rank):
            col = t * strassen_rank + s
            pattern = strassen_patterns[s]
            
            for u_idx, u_val in pattern['U']:
                U[u_idx, col] += u_val
                
            for v_idx, v_val in pattern['V']:
                # v_idx is local 0..3 in 2x2 block
                # Global index in V is j*p + k
                # Here j is row of B (0 or 1), k is col of B (local 0 or 1)
                # Wait, V shape is (m*p, rank) = (2*14, rank).
                # Index in V corresponds to (j, k) where j is row of B (0..1) and k is col of B (0..13).
                # In our block t, the columns of B are k_offset and k_offset+1.
                # The local indices 0,1,2,3 in the 2x2 block correspond to:
                # 0: (0,0) -> global V index 0*p + (k_offset+0)
                # 1: (0,1) -> global V index 0*p + (k_offset+1)
                # 2: (1,0) -> global V index 1*p + (k_offset+0)
                # 3: (1,1) -> global V index 1*p + (k_offset+1)
                
                # Mapping local 2x2 index to global V index
                # local_idx = r * 2 + c
                # global_idx = r * p + (k_offset + c)
                r = v_idx // 2
                c = v_idx % 2
                global_v_idx = r * p + (k_offset + c)
                V[global_v_idx, col] += v_val

            for w_idx, w_val in pattern['W']:
                # w_idx is local 0..3 in 2x2 block C
                # C is 2x2, indices (i, k) where i is row of C (0..1) and k is col of C (0..1)
                # W shape is (p*n, rank) = (14*2, rank).
                # Index in W corresponds to (k, i) where k is col of B (0..13) and i is row of A (0..1).
                # Wait, the trace convention is W[k*n+i, r].
                # So index is k*n + i.
                # In our block t, the columns of B are k_offset and k_offset+1.
                # The local indices 0,1,2,3 in the 2x2 block C correspond to:
                # 0: (0,0) -> i=0, k=0 (local) -> global k = k_offset+0
                # 1: (0,1) -> i=0, k=1 (local) -> global k = k_offset+1
                # 2: (1,0) -> i=1, k=0 (local) -> global k = k_offset+0
                # 3: (1,1) -> i=1, k=1 (local) -> global k = k_offset+1
                
                # Mapping local 2x2 index to global W index
                # local_idx = i * 2 + k (standard row-major for C)
                # But wait, the previous code used:
                # W[(k_offset + 0) * n + 0, col] for C00
                # This implies index = k_global * n + i.
                # Let's check the local index mapping again.
                # C00: i=0, k_local=0.
                # C01: i=0, k_local=1.
                # C10: i=1, k_local=0.
                # C11: i=1, k_local=1.
                
                # If we use row-major for C (2x2):
                # 0: C00 (i=0, k=0)
                # 1: C01 (i=0, k=1)
                # 2: C10 (i=1, k=0)
                # 3: C11 (i=1, k=1)
                
                i = w_idx // 2
                k_local = w_idx % 2
                k_global = k_offset + k_local
                global_w_idx = k_global * n + i
                W[global_w_idx, col] += w_val

    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Quick verification
    if time.time() < deadline - 5.0:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
        if max_err > 1e-9:
            # Fallback to schoolbook if verification fails
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb), dtype=np.int32)
            V_fb = np.zeros((m * p, rank_fb), dtype=np.int32)
            W_fb = np.zeros((p * n, rank_fb), dtype=np.int32)
            col = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_fb[ii * m + jj, col] = 1
                        V_fb[jj * p + kk, col] = 1
                        W_fb[kk * n + ii, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
            
    return {"U": U_list, "V": V_list, "W": W_list}
