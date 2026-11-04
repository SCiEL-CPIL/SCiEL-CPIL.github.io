import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16
    
    # Strategy: Split A (2x3) into A1 (2x2) and A2 (2x1).
    # Split B (3x8) into B1 (2x8) and B2 (1x8).
    # C = A1*B1 + A2*B2.
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # Split B1 into 4 blocks of 2x2.
    # Use Strassen (rank 7) for each 2x2 * 2x2.
    # Total rank: 4 * 7 = 28.
    
    # Part 2: A2 (2x1) * B2 (1x8)
    # This is an outer product.
    # Rank: 2 * 8 = 16.
    
    # Total Rank: 28 + 16 = 44.
    
    rank = 44
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen 2x2x2 coefficients
    # Local indices for 2x2 block:
    # A: [[00, 01], [10, 11]] -> indices 0, 1, 2, 3
    # B: [[00, 01], [10, 11]] -> indices 0, 1, 2, 3
    # C: [[00, 01], [10, 11]] -> indices 0, 1, 2, 3 (k*n+i, n=2)
    
    # Correct Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the vectors for each term t in {0..6} corresponding to M1..M7.
    # U_vec[t] is length 4 (for A)
    # V_vec[t] is length 4 (for B)
    # W_vec[t] is length 4 (for C)
    
    # M1: (A00+A11)(B00+B11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: contributes to C00 (+1), C11 (+1)
    # C indices: 00->0, 01->1, 10->2, 11->3
    # W: [1, 0, 0, 1]
    
    # M2: (A10+A11)B00
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: contributes to C10 (+1), C11 (-1)
    # W: [0, 0, 1, -1]
    
    # M3: A00(B10-B11)
    # U: [1, 0, 0, 0]
    # V: [0, 0, 1, -1]
    # W: contributes to C01 (+1), C11 (+1)
    # W: [0, 1, 0, 1]
    
    # M4: A11(B01-B10)
    # U: [0, 0, 0, 1]
    # V: [0, 1, -1, 0]
    # W: contributes to C00 (+1), C10 (+1)
    # W: [1, 0, 1, 0]
    
    # M5: (A00+A01)B11
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: contributes to C00 (-1), C01 (+1)
    # W: [-1, 1, 0, 0]
    
    # M6: (A10-A00)(B00+B10)
    # U: [-1, 0, 1, 0]
    # V: [1, 0, 1, 0]
    # W: contributes to C11 (+1)
    # W: [0, 0, 0, 1]
    
    # M7: (A01-A11)(B01+B11)
    # U: [0, 1, 0, -1]
    # V: [0, 1, 0, 1]
    # W: contributes to C00 (+1)
    # W: [1, 0, 0, 0]
    
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [0, 1, -1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 0, 1, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # A1 corresponds to A[0:2, 0:2] -> indices (i, j) with i in {0,1}, j in {0,1}
    # Global U index: i*m + j = i*3 + j
    # B1 corresponds to B[0:2, :] -> indices (j, k) with j in {0,1}, k in {0..7}
    # Global V index: j*p + k = j*8 + k
    # C corresponds to C[:, :] -> indices (k, i)
    # Global W index: k*n + i = k*2 + i
    
    for block_idx in range(4):
        k_start = block_idx * 2
        
        for r in range(7):
            # Fill U
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        u_idx = li * m + la
                        U[u_idx, col] += val
            
            # Fill V
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        v_idx = la * p + (k_start + lk)
                        V[v_idx, col] += val
            
            # Fill W
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        w_idx = (k_start + lk) * n + li
                        W[w_idx, col] += val
            
            col += 1
            
    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 corresponds to A[0:2, 2] -> indices (i, 2) with i in {0,1}
    # Global U index: i*3 + 2
    # B2 corresponds to B[2, :] -> indices (2, k) with k in {0..7}
    # Global V index: 2*8 + k
    # C corresponds to C[:, :] -> indices (k, i)
    # Global W index: k*2 + i
    
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            
            col += 1
            
    # Validation
    # Check if sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]) == 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0.0
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                # Vectorized dot product
                acc = np.dot(np.dot(u_row, v_row), w_row)
                if not np.isclose(acc, 1.0):
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    # Also check off-diagonals are 0?
    # The tensor T is non-zero only for consistent (i,j,k).
    # The decomposition must satisfy sum_t U[a,t]V[b,t]W[c,t] = 0 for inconsistent (a,b,c).
    # Let's spot check a few inconsistent ones.
    # a = i*m+j, b = j'*p+k, c = k'*n+i' with j != j'
    # Example: i=0, j=0, j'=1, k=0, k'=0, i'=0
    # a = 0, b = 8, c = 0
    # U[0, t] * V[8, t] * W[0, t] should be 0.
    # In our construction:
    # U[0, t] is non-zero for t in 0..27 (Strassen part, i=0)
    # V[8, t] corresponds to j'=1, k=0. In Strassen part, this is la=1, lk=0.
    # For a given t (Strassen term r), U[0, t] has value S_U[r][0] (li=0, la=0).
    # V[8, t] has value S_V[r][2] (la=1, lk=0).
    # W[0, t] has value S_W[r][0] (lk=0, li=0).
    # So we need sum_r S_U[r][0] * S_V[r][2] * S_W[r][0] = 0.
    
    if valid:
        # Check a few off-diagonals
        check_pairs = [
            (0, 8, 0),  # i=0,j=0 vs j'=1,k=0 vs k'=0,i'=0
            (0, 0, 2),  # i=0,j=0 vs j'=0,k=0 vs k'=1,i'=0
            (1, 8, 0),  # i=0,j=1 vs j'=1,k=0 vs k'=0,i'=0
        ]
        for a, b, c in check_pairs:
            acc = np.dot(np.dot(U[a], V[b]), W[c])
            if not np.isclose(acc, 0.0):
                valid = False
                break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_s = n * m * p
        U_s = np.zeros((nm, rank_s))
        V_s = np.zeros((mp, rank_s))
        W_s = np.zeros((pn, rank_s))
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col_s] = 1
                    V_s[j * p + k, col_s] = 1
                    W_s[k * n + i, col_s] = 1
                    col_s += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
