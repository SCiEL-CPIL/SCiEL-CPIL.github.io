import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    
    # We need to decompose the tensor T such that T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # where A is n x m, B is m x p, C is n x p.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # The directive suggests a rank of 38 using a "known rank-13 decomposition for 4x4".
    # However, the best known rank for 4x4x4 is 49 (Strassen) or 48 (other algos), and lower bound is 36.
    # Rank 13 is definitely wrong for 4x4x4.
    # The schoolbook rank for 4x4x4 is 64.
    # The schoolbook rank for 4x4x3 is 48.
    # Total schoolbook: 112.
    
    # Let's stick to a provably correct and efficient decomposition.
    # Strassen for 4x4x4 gives rank 49.
    # Schoolbook for 4x4x3 gives rank 48.
    # Total rank: 97.
    
    # This is a significant improvement over 112.
    
    U = []
    V = []
    W = []
    
    # --- Part 1: 4x4 * 4x4 using Strassen (Rank 49) ---
    # We use the Kronecker product of the 2x2 Strassen decomposition.
    # The 2x2 Strassen terms (u, v, w) are 4-vectors.
    # The 4x4 terms are u_i \otimes u_j, v_i \otimes v_j, w_i \otimes w_j for i,j in 1..7.
    
    # 2x2 Strassen vectors (flattened 2x2 matrices)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    s2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,-1]),   # P1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # P2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),   # P3
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),   # P4
        ([1,1,0,0], [0,0,0,1], [0,0,-1,1]),   # P5
        ([0,0,1,-1], [1,1,0,0], [0,0,0,1]),   # P6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # P7
    ]
    
    for i in range(7):
        u_i, v_i, w_i = s2x2[i]
        for j in range(7):
            u_j, v_j, w_j = s2x2[j]
            
            # Kronecker product
            u_kron = np.kron(u_i, u_j)
            v_kron = np.kron(v_i, v_j)
            w_kron = np.kron(w_i, w_j)
            
            # These are for the top-left 4x4 block of C (which is 4x4).
            # The U vector corresponds to A (4x4).
            # The V vector corresponds to B (4x4 part, cols 0-3).
            # The W vector corresponds to C (4x4 part, cols 0-3).
            
            # We need to embed these into the full vectors.
            # U is length 16 (4x4). u_kron is length 16.
            # V is length 28 (4x7). v_kron is length 16. We pad with 12 zeros.
            # W is length 28 (7x4). w_kron is length 16. We need to place it correctly.
            
            # Let's check the indexing conventions.
            # U index: i*m + j. For 4x4, 0..15.
            # V index: j*p + k. For 4x7, 0..27.
            # W index: k*n + i. For 7x4, 0..27.
            
            # The Kronecker product mapping:
            # If we view 4x4 as 2x2 blocks of 2x2, the flattening is consistent.
            # u_kron[k] corresponds to A[row, col] where row = k // 4, col = k % 4.
            # This matches U index = row * 4 + col.
            
            # v_kron[k] corresponds to B[row, col] where row = k // 4, col = k % 4.
            # This matches V index = row * 7 + col.
            # BUT v_kron is for a 4x4 B matrix. So col is 0..3.
            # We need to map B[row, col] to V[row * 7 + col].
            
            # w_kron[k] corresponds to C[row, col] where row = k // 4, col = k % 4.
            # This matches C index in the 4x4 block.
            # But W index is k * n + i, which is C[col, row] if we follow the trace convention W[k*n+i, r]?
            # The prompt says: "output factor follows the official trace convention W[k*n+i,r]"
            # Usually C = AB. C_{i,k} = sum_j A_{i,j} B_{j,k}.
            # Tensor T_{i,j,k}.
            # U_{i,j,r}, V_{j,k,r}, W_{k,i,r}.
            # So W index is (k, i). k is col of C, i is row of C.
            # In the 4x4 block, C is 4x4.
            # w_kron is a flattened 4x4 matrix. Let's say it represents C_{row, col}.
            # We need to map C_{row, col} to W index (col, row).
            # So W_index = col * 4 + row.
            # In w_kron, index k corresponds to C_{row, col} with row = k // 4, col = k % 4.
            # So we need to transpose the w_kron vector before placing it into W.
            
            # Let's construct the full vectors.
            u_full = np.zeros(16)
            v_full = np.zeros(28)
            w_full = np.zeros(28)
            
            u_full[:] = u_kron
            
            # Map v_kron (4x4 flattened) to v_full (4x7 flattened)
            for idx in range(16):
                r = idx // 4
                c = idx % 4
                v_full[r * 7 + c] = v_kron[idx]
                
            # Map w_kron (4x4 flattened) to w_full (7x4 flattened, transposed index)
            for idx in range(16):
                r = idx // 4  # row of C
                c = idx % 4   # col of C
                # W index is c * 4 + r
                w_full[c * 4 + r] = w_kron[idx]
                
            U.append(u_full)
            V.append(v_full)
            W.append(w_full)
            
    # --- Part 2: 4x4 * 4x3 using Schoolbook (Rank 48) ---
    # A is 4x4, B is 4x3 (cols 4,5,6 of the 4x7 B), C is 4x3 (cols 4,5,6 of the 4x7 C).
    # Schoolbook: for each i in 0..3, j in 0..3, k in 4..6:
    # C[i,k] += A[i,j] * B[j,k]
    # Term: U_{i,j} = 1, V_{j,k} = 1, W_{k,i} = 1.
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_full = np.zeros(16)
                v_full = np.zeros(28)
                w_full = np.zeros(28)
                
                u_full[i * 4 + j] = 1
                v_full[j * 7 + k] = 1
                w_full[k * 4 + i] = 1
                
                U.append(u_full)
                V.append(v_full)
                W.append(w_full)
                
    # Convert to lists
    U = [list(map(float, u)) for u in U]
    V = [list(map(float, v)) for v in V]
    W = [list(map(float, w)) for w in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
