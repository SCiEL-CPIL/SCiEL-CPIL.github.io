"""Matrix multiplication 2x3x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS

    # Construct a deterministic rank-44 decomposition
    # based on a rank-11 decomposition of 2x3x2.
    # Rank 11 for 2x3x2 comes from Strassen (rank 7 for 2x2x2) + Schoolbook (rank 4 for 2x1x2).
    
    # 1. Define Strassen factors for 2x2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_strassen = np.array([
        [1, 0, 0, 1, 1, -1, 0], # a+d, c-a, b-d
        [0, 0, 0, 0, 1, 0, 1], # b
        [0, 1, 1, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, 0, -1], # d
    ], dtype=float)
    
    V_strassen = np.array([
        [1, 1, 0, 0, 0, 1, 0], # e+f
        [0, 0, 1, 1, 0, 0, 0], # f
        [0, 0, 0, 0, 0, 0, 1], # g
        [1, 0, -1, -1, 1, 0, 1], # h
    ], dtype=float)
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 1, 0, 1, 0, 0], # C01
        [0, 0, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0], # C11
    ], dtype=float)

    # 2. Define Schoolbook factors for 2x1x2
    # A indices: 0, 1
    # B indices: 0, 1
    # C indices: 0, 1, 2, 3
    
    U_school = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)
    
    V_school = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ], dtype=float)
    
    W_school = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    # 3. Combine into 2x3x2 decomposition (Rank 11)
    # A is 2x3. Split into A_L (2x2) and A_R (2x1).
    # B is 3x2. Split into B_L (2x2) and B_R (1x2).
    # C is 2x2.
    
    U_232 = np.zeros((6, 11))
    V_232 = np.zeros((6, 11))
    W_232 = np.zeros((4, 11))
    
    # Map Strassen (cols 0-6)
    # A_L indices in 2x3: 0,1,3,4 corresponding to local 0,1,2,3
    # Local 0 (00) -> Global 0
    # Local 1 (01) -> Global 1
    # Local 2 (10) -> Global 3
    # Local 3 (11) -> Global 4
    
    U_232[0, 0:7] = U_strassen[0, :]
    U_232[1, 0:7] = U_strassen[1, :]
    U_232[3, 0:7] = U_strassen[2, :]
    U_232[4, 0:7] = U_strassen[3, :]
    
    # B_L indices in 3x2: 0,1,2,3 corresponding to local 0,1,2,3
    # Local 0 (00) -> Global 0
    # Local 1 (01) -> Global 1
    # Local 2 (10) -> Global 2
    # Local 3 (11) -> Global 3
    
    V_232[0, 0:7] = V_strassen[0, :]
    V_232[1, 0:7] = V_strassen[1, :]
    V_232[2, 0:7] = V_strassen[2, :]
    V_232[3, 0:7] = V_strassen[3, :]
    
    # C indices are the same
    
    W_232[:, 0:7] = W_strassen[:, :]
    
    # Map Schoolbook (cols 7-10)
    # A_R indices in 2x3: 2,5 corresponding to local 0,1
    # Local 0 -> Global 2
    # Local 1 -> Global 5
    
    U_232[2, 7:11] = U_school[0, :]
    U_232[5, 7:11] = U_school[1, :]
    
    # B_R indices in 3x2: 4,5 corresponding to local 0,1
    # Local 0 -> Global 4
    # Local 1 -> Global 5
    
    V_232[4, 7:11] = V_school[0, :]
    V_232[5, 7:11] = V_school[1, :]
    
    # C indices are the same
    
    W_232[:, 7:11] = W_school[:, :]
    
    # 4. Tile to 2x3x8
    # A is 2x3. U is same for all 4 blocks.
    # B is 3x8. V is block diagonal.
    # C is 2x8. W is block diagonal.
    
    rank_total = 44
    U_full = np.zeros((6, rank_total))
    V_full = np.zeros((24, rank_total))
    W_full = np.zeros((16, rank_total))
    
    for g in range(4):
        col_start = g * 11
        col_end = col_start + 11
        
        # U is identical
        U_full[:, col_start:col_end] = U_232
        
        # V mapping
        # Local V index l (0..5) -> j = l//2, k = l%2
        # Global V index -> j*8 + (2g + k)
        for l in range(6):
            j = l // 2
            k = l % 2
            global_idx = j * 8 + (2 * g + k)
            V_full[global_idx, col_start:col_end] = V_232[l, :]
            
        # W mapping
        # Local W index l (0..3) -> k_local = l//2, i = l%2
        # Global W index -> (2g + k_local)*2 + i
        for l in range(4):
            k_local = l // 2
            i = l % 2
            global_idx = (2 * g + k_local) * 2 + i
            W_full[global_idx, col_start:col_end] = W_232[l, :]
            
    # Verification
    T_full = np.zeros((6, 24, 16))
    for i in range(2):
        for j in range(3):
            for k in range(8):
                T_full[i*3+j, j*8+k, k*2+i] = 1.0

    recon_full = np.zeros((6, 24, 16))
    for c in range(rank_total):
        recon_full += np.einsum('i,j,k->ijk', U_full[:,c], V_full[:,c], W_full[:,c])
        
    if np.linalg.norm(T_full - recon_full) < 1e-9:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback just in case
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
