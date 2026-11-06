import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return 7 factor triples for 2x2x2 Strassen decomposition."""
    return [
        ([1, 0, 0, 1],   [1, 0, 0, 1],   [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1],   [1, 0, 0, 0],   [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0],   [0, 1, 0, -1],  [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1],   [-1, 0, 1, 0],  [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0],   [0, 0, 0, 1],   [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0],  [1, 1, 0, 0],   [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1],  [0, 0, 1, 1],   [1, 0, 0, 0]),   # M7
    ]

def _strassen_4x4x4():
    """
    Construct 4x4x4 decomposition using recursive Strassen.
    Returns list of (U_vec, V_vec, W_vec) where each vec has length 16.
    """
    s2 = _strassen_2x2x2()
    factors = []
    
    # Index maps for 4x4 blocks
    a_block_map = [
        [0, 2, 8, 10],   # A_eff[0,0]
        [1, 3, 9, 11],   # A_eff[0,1]
        [4, 6, 12, 14],  # A_eff[1,0]
        [5, 7, 13, 15],  # A_eff[1,1]
    ]
    b_block_map = [
        [0, 2, 8, 10],   # B_eff[0,0]
        [1, 3, 9, 11],   # B_eff[0,1]
        [4, 6, 12, 14],  # B_eff[1,0]
        [5, 7, 13, 15],  # B_eff[1,1]
    ]
    c_block_map = [
        [0, 2, 8, 10],   # C_eff[0,0]
        [1, 3, 9, 11],   # C_eff[0,1]
        [4, 6, 12, 14],  # C_eff[1,0]
        [5, 7, 13, 15],  # C_eff[1,1]
    ]
    
    for t in range(7):
        U_t, V_t, W_t = s2[t]
        for s in range(7):
            U_s, V_s, W_s = s2[s]
            U_full = np.zeros(16)
            V_full = np.zeros(16)
            W_full = np.zeros(16)
            
            for comp in range(4):
                coeff_u = U_s[comp]
                if coeff_u != 0:
                    for block_idx in range(4):
                        val = coeff_u * U_t[block_idx]
                        if val != 0:
                            U_full[a_block_map[comp][block_idx]] += val
                
                coeff_v = V_s[comp]
                if coeff_v != 0:
                    for block_idx in range(4):
                        val = coeff_v * V_t[block_idx]
                        if val != 0:
                            V_full[b_block_map[comp][block_idx]] += val
                
                coeff_w = W_s[comp]
                if coeff_w != 0:
                    for block_idx in range(4):
                        val = coeff_w * W_t[block_idx]
                        if val != 0:
                            W_full[c_block_map[comp][block_idx]] += val
            
            factors.append((U_full.tolist(), V_full.tolist(), W_full.tolist()))
    
    return factors

def _verify(n, m, p, U, V, W, tol=1e-9):
    """Verify the decomposition."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_5x5x5():
    """
    Construct 5x5x5 decomposition using 4+1 partition.
    Core 4x4x4: Rank 49.
    Borders: 7 blocks, each Rank 1.
    Total Rank: 56.
    """
    n, m, p = 5, 5, 5
    
    strassen_4x4x4 = _strassen_4x4x4()
    rank_core = len(strassen_4x4x4) # 49
    rank_borders = 7
    total_rank = rank_core + rank_borders
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    
    # 1. Core 4x4x4
    for (uf, vf, wf) in strassen_4x4x4:
        for idx in range(16):
            if uf[idx] != 0:
                U[idx, col] += uf[idx]
            if vf[idx] != 0:
                V[idx, col] += vf[idx]
            if wf[idx] != 0:
                W[idx, col] += wf[idx]
        col += 1
    
    # 2. Border 4x4x1 (k=4)
    # U[i, j], V[j, 4], W[4, i]
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
    col += 1
    
    # 3. Border 4x1x4 (j=4)
    # U[i, 4], V[4, k], W[k, i]
    for i in range(4):
        for k in range(4):
            U[i * m + 4, col] = 1.0
            V[4 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
    col += 1
    
    # 4. Border 1x4x4 (i=4)
    # U[4, j], V[j, k], W[k, 4]
    for j in range(4):
        for k in range(4):
            U[4 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 4, col] = 1.0
    col += 1
    
    # 5. Border 4x1x1 (j=4, k=4)
    # U[i, 4], V[4, 4], W[4, i]
    for i in range(4):
        U[i * m + 4, col] = 1.0
        V[4 * p + 4, col] = 1.0
        W[4 * n + i, col] = 1.0
    col += 1
    
    # 6. Border 1x4x1 (i=4, k=4)
    # U[4, j], V[j, 4], W[4, 4]
    for j in range(4):
        U[4 * m + j, col] = 1.0
        V[j * p + 4, col] = 1.0
        W[4 * n + 4, col] = 1.0
    col += 1
    
    # 7. Border 1x1x4 (i=4, j=4)
    # U[4, 4], V[4, k], W[k, 4]
    for k in range(4):
        U[4 * m + 4, col] = 1.0
        V[4 * p + k, col] = 1.0
        W[k * n + 4, col] = 1.0
    col += 1
    
    # 8. Border 1x1x1 (i=4, j=4, k=4)
    # U[4, 4], V[4, 4], W[4, 4]
    U[4 * m + 4, col] = 1.0
    V[4 * p + 4, col] = 1.0
    W[4 * n + 4, col] = 1.0
    col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

def construct():
    """Main entry point."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    result, rank = _construct_5x5x5()
    
    if _verify(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: schoolbook rank 125
    U = np.zeros((n * m, 125), dtype=np.float64)
    V = np.zeros((m * p, 125), dtype=np.float64)
    W = np.zeros((p * n, 125), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
