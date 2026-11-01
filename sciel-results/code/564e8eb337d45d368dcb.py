"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # We use 4 Strassen blocks (rank 7 each = 28) + 32 standard terms = 60 total
    # But we can potentially merge some standard terms with Strassen terms if they overlap
    # Actually, the Strassen blocks cover specific (j,k) pairs. Let's count uncovered terms.
    # Blocks: (0,1,0,1), (2,3,2,3), (0,1,2,3), (2,3,0,1)
    # Covered (j,k) pairs: 
    # Block 1: j in {0,1}, k in {0,1} -> (0,0),(0,1),(1,0),(1,1)
    # Block 2: j in {2,3}, k in {2,3} -> (2,2),(2,3),(3,2),(3,3)
    # Block 3: j in {0,1}, k in {2,3} -> (0,2),(0,3),(1,2),(1,3)
    # Block 4: j in {2,3}, k in {0,1} -> (2,0),(2,1),(3,0),(3,1)
    # Total covered (j,k): 16 pairs
    # Total (j,k) pairs: 5*6 = 30
    # Uncovered (j,k): 30 - 16 = 14 pairs
    # For each uncovered (j,k), we need n=2 terms (one for each i)
    # So uncovered terms: 14 * 2 = 28
    # Total rank: 28 (Strassen) + 28 (standard) = 56

    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Define the Strassen blocks
    blocks = [
        (0, 1, 0, 1),
        (2, 3, 2, 3),
        (0, 1, 2, 3),
        (2, 3, 0, 1)
    ]
    
    # Set of (j,k) pairs covered by Strassen blocks
    covered_jk = set()
    for j1, j2, k1, k2 in blocks:
        for j in [j1, j2]:
            for k in [k1, k2]:
                covered_jk.add((j, k))
    
    # Add standard terms for uncovered (j,k) pairs
    for j in range(m):
        for k in range(p):
            if (j, k) in covered_jk:
                continue
            for i in range(n):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Add Strassen terms for each block
    for j1, j2, k1, k2 in blocks:
        u_00 = 0 * m + j1
        u_01 = 0 * m + j2
        u_10 = 1 * m + j1
        u_11 = 1 * m + j2
        
        v_00 = j1 * p + k1
        v_01 = j1 * p + k2
        v_10 = j2 * p + k1
        v_11 = j2 * p + k2
        
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_10 = k2 * n + 0
        w_11 = k2 * n + 1
        
        # M1: (A00 + A11)(B00 + B11)
        U[u_00, col] = 1
        U[u_11, col] = 1
        V[v_00, col] = 1
        V[v_11, col] = 1
        W[w_00, col] = 1
        W[w_11, col] = 1
        col += 1
        
        # M2: (A10 + A11)B00
        U[u_10, col] = 1
        U[u_11, col] = 1
        V[v_00, col] = 1
        W[w_10, col] = 1
        W[w_11, col] = -1
        col += 1
        
        # M3: A00(B01 - B11)
        U[u_00, col] = 1
        V[v_01, col] = 1
        V[v_11, col] = -1
        W[w_01, col] = 1
        W[w_11, col] = 1
        col += 1
        
        # M4: A11(B10 - B00)
        U[u_11, col] = 1
        V[v_10, col] = 1
        V[v_00, col] = -1
        W[w_00, col] = 1
        W[w_10, col] = 1
        col += 1
        
        # M5: (A00 + A01)B11
        U[u_00, col] = 1
        U[u_01, col] = 1
        V[v_11, col] = 1
        W[w_00, col] = -1
        W[w_01, col] = 1
        col += 1
        
        # M6: (A10 - A00)(B00 + B01)
        U[u_10, col] = 1
        U[u_00, col] = -1
        V[v_00, col] = 1
        V[v_01, col] = 1
        W[w_11, col] = 1
        col += 1
        
        # M7: (A01 - A10)(B10 + B11)
        U[u_01, col] = 1
        U[u_10, col] = -1
        V[v_10, col] = 1
        V[v_11, col] = 1
        W[w_00, col] = 1
        col += 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 60 if verification fails
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i*m+j, c] = 1
                    V_fallback[j*p+k, c] = 1
                    W_fallback[k*n+i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

# EVOLVE-BLOCK-END
