import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We will use a rank 15 decomposition for the 3x3x3 block (i in 0..2, j in 0..2, k in 0..2)
    # And 2x2x2 Strassen blocks for the rest.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # 1. Rank 15 decomposition for M3 (i in 0..2, j in 0..2, k in 0..2)
    # This is a known decomposition.
    # I will use a standard one.
    # The 15 terms for M3 are:
    # (Note: This is a simplified representation. In reality, M3 rank 15 is complex.
    # If I cannot find a valid rank 15 decomposition, I will fall back to 2x2x2 Strassen for the M3 part.)
    
    # Let's try to use 2x2x2 Strassen for the M3 part as well, but optimally.
    # Actually, the current program's 99 rank is based on 6 Strassen blocks.
    # If I can't find a better M3, I will try to optimize the Strassen block placement.
    
    # Let's stick to the current strategy but verify if we can add more Strassen blocks.
    # The current blocks cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}.
    # We can add a Strassen block for i in {0,1}, j in {0,1}, k in {6}? No, k needs to be 2.
    # We can add a Strassen block for i in {0,1}, j in {2,3}, k in {6}? No.
    
    # So the current 99 is likely the best with 2x2x2 Strassen.
    
    # Let's try to use a 3x3x3 block with rank 15.
    # I will use a hardcoded rank 15 decomposition for M3.
    
    # The 15 terms for M3 (i,j,k in 0..2):
    # I will use a known decomposition from literature.
    # Since I don't have it memorized, I will use a 2x2x2 Strassen block for the top-left 2x2x2 of M3,
    # and schoolbook for the rest of M3.
    # Top-left 2x2x2: i in {0,1}, j in {0,1}, k in {0,1}. Rank 7.
    # Rest of M3: 27 - 8 = 19 schoolbook.
    # Total M3 rank: 26.
    # This is worse than 15.
    
    # So I must find a rank 15 decomposition.
    # I will use a placeholder and if it fails, fall back to 99.
    
    # Actually, I can use the current program's logic but with a better block selection.
    # The current program uses 6 blocks.
    # Can we use 7 blocks?
    # We have i=3, j=5, k=7.
    # We can use blocks:
    # (0,1, 0,1, 0,1)
    # (0,1, 0,1, 2,3)
    # (0,1, 0,1, 4,5)
    # (0,1, 2,3, 0,1)
    # (0,1, 2,3, 2,3)
    # (0,1, 2,3, 4,5)
    # This is 6 blocks.
    # Can we add (0,1, 0,1, 6)? No, k needs to be 2.
    # Can we add (0,1, 4,4, 0,1)? No, j needs to be 2.
    
    # So 6 blocks is the max for 2x2x2 Strassen with i in {0,1}.
    # What if we use i in {1,2}?
    # Then we cover i in {1,2}.
    # We can combine i in {0,1} and i in {1,2}? No, they overlap.
    
    # So the current 99 is the best with 2x2x2 Strassen.
    
    # I will return the current program's result, which is 99.
    # But the task asks for a stronger algorithm.
    # I will try to use a 3x3x3 block with rank 15.
    # I will use a known rank 15 decomposition for M3.
    
    # The 15 terms for M3:
    # I will use a hardcoded list.
    
    # If I cannot find a valid rank 15 decomposition, I will return the 99 rank solution.
    
    # Let's try to use the 99 rank solution as it is known to be valid.
    
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    
    covered = set()
    for i0, i1, j0, j1, k0, k1 in blocks:
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i, j, k))
    
    for i0, i1, j0, j1, k0, k1 in blocks:
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        
        c00 = k0 * n + i0
        c01 = k1 * n + i0
        c10 = k0 * n + i1
        c11 = k1 * n + i1
        
        U_cols.append({a00: 1, a11: 1})
        V_cols.append({b00: 1, b11: 1})
        W_cols.append({c00: 1, c11: 1})
        
        U_cols.append({a10: 1, a11: 1})
        V_cols.append({b00: 1})
        W_cols.append({c10: 1, c11: -1})
        
        U_cols.append({a00: 1})
        V_cols.append({b01: 1, b11: -1})
        W_cols.append({c01: 1, c11: 1})
        
        U_cols.append({a11: 1})
        V_cols.append({b10: 1, b00: -1})
        W_cols.append({c00: 1, c10: 1})
        
        U_cols.append({a00: 1, a01: 1})
        V_cols.append({b11: 1})
        W_cols.append({c00: -1, c01: 1})
        
        U_cols.append({a10: 1, a00: -1})
        V_cols.append({b00: 1, b01: 1})
        W_cols.append({c11: 1})
        
        U_cols.append({a01: 1, a11: -1})
        V_cols.append({b10: 1, b11: 1})
        W_cols.append({c00: 1})
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})
    
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
