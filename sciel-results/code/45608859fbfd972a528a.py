import time
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Initialize with schoolbook decomposition
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    current_rank = rank_sb
    
    # Function to apply Strassen reduction on a 2x2x2 block
    # Indices: i1, i2, j1, j2, k1, k2
    def apply_strassen(i1, i2, j1, j2, k1, k2, start_col):
        # We will replace 8 columns (start_col to start_col+7) with 7 new columns
        # The 8 columns correspond to the 8 rank-1 terms of the 2x2x2 block
        # We need to identify which columns in U, V, W correspond to these terms
        
        # The mapping of (i,j,k) to column index in schoolbook:
        # col = i*m*p + j*p + k ? No.
        # In the init loop:
        # col is incremented in order i, j, k.
        # col = i * (m*p) + j * p + k
        
        cols = []
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in [k1, k2]:
                    c = i * (m * p) + j * p + k
                    cols.append(c)
                    
        # Extract the 8 vectors
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # We will zero out these 8 columns and write 7 new ones into the first 7 of them
        
        # Strassen terms for 2x2x2:
        # M1: A00+A11, B00+B11, C00+C11 (partially)
        # Let's define the U, V, W vectors for the 7 terms
        
        # Local indices:
        # A: 00, 01, 10, 11 -> global i1*m+j1, i1*m+j2, i2*m+j1, i2*m+j2
        # B: 00, 01, 10, 11 -> global j1*p+k1, j1*p+k2, j2*p+k1, j2*p+k2
        # C: 00, 01, 10, 11 -> global k1*n+i1, k1*n+i2, k2*n+i1, k2*n+i2
        
        u_00 = i1 * m + j1
        u_01 = i1 * m + j2
        u_10 = i2 * m + j1
        u_11 = i2 * m + j2
        
        v_00 = j1 * p + k1
        v_01 = j1 * p + k2
        v_10 = j2 * p + k1
        v_11 = j2 * p + k2
        
        w_00 = k1 * n + i1
        w_01 = k2 * n + i1
        w_10 = k1 * n + i2
        w_11 = k2 * n + i2
        
        # 7 Strassen terms
        # T1: U has 1 at u_00, u_11. V has 1 at v_00, v_11. W has 1 at w_00, w_11?
        # C00 = M1 + M4 - M5 - M7
        # C11 = M1 - M2 + M6 + M7
        # M1 = (A00+A11)(B00+B11)
        # So U1: 1 at u_00, u_11. V1: 1 at v_00, v_11.
        # W1 must satisfy: W1_00 * 1 * 1 + ... 
        # Actually, W is the output factor.
        # The term is U1 * V1 * W1.
        # We want U1 * V1 * W1 to contribute to C00 and C11.
        # C00 gets +1 from M1. C11 gets +1 from M1.
        # So W1 should have 1 at w_00 and 1 at w_11.
        
        # T2: M2 = (A10+A11)B00
        # U2: 1 at u_10, u_11. V2: 1 at v_00.
        # C10 gets +1 from M2.
        # So W2: 1 at w_10.
        
        # T3: M3 = A00 B10
        # U3: 1 at u_00. V3: 1 at v_10.
        # C01 gets +1 from M3.
        # So W3: 1 at w_01.
        
        # T4: M4 = A11 B01
        # U4: 1 at u_11. V4: 1 at v_01.
        # C00 gets +1 from M4. C10 gets +1 from M4.
        # So W4: 1 at w_00, 1 at w_10.
        
        # T5: M5 = (A00+A01)B11
        # U5: 1 at u_00, u_01. V5: 1 at v_11.
        # C00 gets -1 from M5. C01 gets +1 from M5.
        # So W5: -1 at w_00, 1 at w_01.
        
        # T6: M6 = (A10-A00)(B01+B11)
        # U6: 1 at u_10, -1 at u_00. V6: 1 at v_01, 1 at v_11.
        # C11 gets +1 from M6.
        # So W6: 1 at w_11.
        
        # T7: M7 = (A01-A11)(B00+B10)
        # U7: 1 at u_01, -1 at u_11. V7: 1 at v_00, 1 at v_10.
        # C00 gets -1 from M7. C11 gets +1 from M7.
        # So W7: -1 at w_00, 1 at w_11.
        
        # Zero out the 8 old columns
        for c in cols:
            U[:, c] = 0.0
            V[:, c] = 0.0
            W[:, c] = 0.0
            
        # Write the 7 new terms into the first 7 columns
        new_cols = cols[:7]
        
        # T1
        U[u_00, new_cols[0]] += 1.0
        U[u_11, new_cols[0]] += 1.0
        V[v_00, new_cols[0]] += 1.0
        V[v_11, new_cols[0]] += 1.0
        W[w_00, new_cols[0]] += 1.0
        W[w_11, new_cols[0]] += 1.0
        
        # T2
        U[u_10, new_cols[1]] += 1.0
        U[u_11, new_cols[1]] += 1.0
        V[v_00, new_cols[1]] += 1.0
        W[w_10, new_cols[1]] += 1.0
        
        # T3
        U[u_00, new_cols[2]] += 1.0
        V[v_10, new_cols[2]] += 1.0
        W[w_01, new_cols[2]] += 1.0
        
        # T4
        U[u_11, new_cols[3]] += 1.0
        V[v_01, new_cols[3]] += 1.0
        W[w_00, new_cols[3]] += 1.0
        W[w_10, new_cols[3]] += 1.0
        
        # T5
        U[u_00, new_cols[4]] += 1.0
        U[u_01, new_cols[4]] += 1.0
        V[v_11, new_cols[4]] += 1.0
        W[w_00, new_cols[4]] -= 1.0
        W[w_01, new_cols[4]] += 1.0
        
        # T6
        U[u_10, new_cols[5]] += 1.0
        U[u_00, new_cols[5]] -= 1.0
        V[v_01, new_cols[5]] += 1.0
        V[v_11, new_cols[5]] += 1.0
        W[w_11, new_cols[5]] += 1.0
        
        # T7
        U[u_01, new_cols[6]] += 1.0
        U[u_11, new_cols[6]] -= 1.0
        V[v_00, new_cols[6]] += 1.0
        V[v_10, new_cols[6]] += 1.0
        W[w_00, new_cols[6]] -= 1.0
        W[w_11, new_cols[6]] += 1.0
        
        # The 8th column (cols[7]) is now zero, so we can ignore it or use it for padding.
        # We effectively reduced the rank by 1 for this block.
        
    # Apply Strassen to disjoint 2x2x2 blocks
    # n=3, m=4, p=7
    # i: 0,1 (pair), 2 (leftover)
    # j: 0,1 (pair), 2,3 (pair)
    # k: 0,1 (pair), 2,3 (pair), 4,5 (pair), 6 (leftover)
    
    i_pairs = [(0, 1)]
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for i1, i2 in i_pairs:
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                if time.time() - start_time > DEADLINE - 5:
                    break
                # Find the starting column for this block
                # The columns are ordered i, j, k.
                # The first column of this block is i1*m*p + j1*p + k1
                start_col = i1 * (m * p) + j1 * p + k1
                apply_strassen(i1, i2, j1, j2, k1, k2, start_col)
                current_rank -= 1 # Effectively, we used 7 cols instead of 8, so rank drops by 1 per block
                
    # The actual rank is the number of non-zero columns.
    # We can trim the matrices to the actual rank.
    # Find non-zero columns
    nz_cols = np.where(np.any(U != 0, axis=0) | np.any(V != 0, axis=0) | np.any(W != 0, axis=0))[0]
    final_rank = len(nz_cols)
    
    U_final = U[:, nz_cols]
    V_final = V[:, nz_cols]
    W_final = W[:, nz_cols]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
