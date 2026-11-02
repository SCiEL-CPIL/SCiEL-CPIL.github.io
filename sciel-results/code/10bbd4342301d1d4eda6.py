import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B)
    # Using Strassen's algorithm, rank 49.
    
    # Helper to get Strassen 2x2x2 decomposition
    def get_strassen_2x2():
        # Returns U, V, W for 2x2x2, rank 7.
        # Indices: i,j for A (0..3), j,k for B (0..3), k,i for C (0..3)
        # U row idx: i*2+j
        # V row idx: j*2+k
        # W row idx: k*2+i
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # 7 terms
        # 1: (A00+A11)(B00+B11)
        # 2: (A10+A11)B00
        # 3: A00(B01-B11)
        # 4: A11(B10-B00)
        # 5: (A00+A01)B11
        # 6: (A10-A00)(B00+B01)
        # 7: (A01-A11)(B10+B11)
        
        # U entries (coeff of A_ij)
        # V entries (coeff of B_jk)
        # W entries (coeff of C_ki)
        
        # Term 1 (idx 0)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1
        W[3, 0] = 1
        
        # Term 2 (idx 1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10 is idx 2 (k=1, i=0 -> 1*2+0=2)
        
        # Term 3 (idx 2)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01 is idx 1 (k=0, i=1 -> 0*2+1=1)
        W[3, 2] = -1
        
        # Term 4 (idx 3)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1
        W[2, 3] = 1
        
        # Term 5 (idx 4)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        W[3, 4] = 1
        
        # Term 6 (idx 5)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = -1
        
        # Term 7 (idx 6)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        W[3, 6] = -1
        
        return U, V, W

    U_s, V_s, W_s = get_strassen_2x2()
    
    # Build 4x4x4 Strassen decomposition.
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((28, 49)) # m*p = 4*7
    W_444 = np.zeros((28, 49)) # p*n = 4*7 (Wait, C is 4x7? No, C is 4x7 in this part? 
                               # The task is 4x4 * 4x7. 
                               # Part 1 is 4x4 * 4x4. So C_part1 is 4x4.
                               # But the final C is 4x7.
                               # So W_444 should only affect the first 4 columns of the final C.
                               # Indices in W_444 for C_part1 (4x4) are 0..15.
                               # In the final W (28x...), these correspond to rows 0..15?
                               # C is 4x7. Row i, col k. Index i*7+k.
                               # For cols 0..3, indices are:
                               # i=0: 0,1,2,3
                               # i=1: 7,8,9,10
                               # i=2: 14,15,16,17
                               # i=3: 21,22,23,24
                               # So we need to map local C indices (0..15) to these global indices.
                               # Local C is 4x4. Index lc = lr*4 + lc_col.
                               # Global C index = gr*7 + gc_col.
                               # Where gr = lr, gc_col = lc_col.
                               # So global_idx = lr*7 + lc_col.
                               # Note: lr = lc // 4, lc_col = lc % 4.
                               # So global_idx = (lc // 4)*7 + (lc % 4).
                               
    # High level linear combinations for A (4 blocks of 2x2)
    # A blocks: A00, A01, A10, A11
    # Block indices (br, bc):
    # A00: (0,0)
    # A01: (0,1)
    # A10: (1,0)
    # A11: (1,1)
    
    # High level terms:
    # 0: (A00+A11) * (B00+B11)
    # 1: (A10+A11) * B00
    # 2: A00 * (B01-B11)
    # 3: A11 * (B10-B00)
    # 4: (A00+A01) * B11
    # 5: (A10-A00) * (B00+B01)
    # 6: (A01-A11) * (B10+B11)
    
    # Define block contributions for each high-level term
    # Format: { 'A': [(block_idx, coeff), ...], 'B': [...], 'C': {block_idx: coeff} }
    # Block idx for A/B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Block idx for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    terms = []
    # T0
    terms.append({
        'A': [(0, 1), (3, 1)],
        'B': [(0, 1), (3, 1)],
        'C': {0: 1, 3: 1} # C00, C11
    })
    # T1
    terms.append({
        'A': [(2, 1), (3, 1)],
        'B': [(0, 1)],
        'C': {2: 1} # C10
    })
    # T2
    terms.append({
        'A': [(0, 1)],
        'B': [(1, 1), (3, -1)],
        'C': {1: 1, 3: -1} # C01, C11
    })
    # T3
    terms.append({
        'A': [(3, 1)],
        'B': [(2, 1), (0, -1)],
        'C': {0: -1, 2: 1} # C00, C10
    })
    # T4
    terms.append({
        'A': [(0, 1), (1, 1)],
        'B': [(3, 1)],
        'C': {1: 1, 3: 1} # C01, C11
    })
    # T5
    terms.append({
        'A': [(2, 1), (0, -1)],
        'B': [(0, 1), (1, 1)],
        'C': {0: -1} # C00
    })
    # T6
    terms.append({
        'A': [(1, 1), (3, -1)],
        'B': [(2, 1), (3, 1)],
        'C': {0: 1, 3: -1} # C00, C11
    })

    # Helper to map local 2x2 index to global 4x4 index for A (4x4)
    def map_A(local_idx, block_idx):
        br = block_idx // 2
        bc = block_idx % 2
        lr = local_idx // 2
        lc = local_idx % 2
        gr = br * 2 + lr
        gc = bc * 2 + lc
        return gr * 4 + gc

    # Helper to map local 2x2 index to global 4x7 index for B (4x7)
    # Note: B is 4x7. Blocks are 2x2.
    # Block (br, bc) covers rows br*2..br*2+1, cols bc*2..bc*2+1.
    # Since p=7, bc can be 0,1,2,3? No, 7 cols.
    # Part 1 uses cols 0..3. So bc is 0 or 1.
    # Block 0: cols 0,1. Block 1: cols 2,3.
    def map_B(local_idx, block_idx):
        br = block_idx // 2
        bc = block_idx % 2
        lr = local_idx // 2
        lc = local_idx % 2
        gr = br * 2 + lr
        gc = bc * 2 + lc
        return gr * 7 + gc

    # Helper to map local 2x2 index to global 4x7 index for C (4x7)
    # Part 1 uses cols 0..3.
    def map_C(local_idx, block_idx):
        br = block_idx // 2
        bc = block_idx % 2
        lr = local_idx // 2
        lc = local_idx % 2
        gr = br * 2 + lr
        gc = bc * 2 + lc
        return gr * 7 + gc

    for h in range(7):
        term = terms[h]
        for l in range(7):
            col = h * 7 + l
            
            # U
            for block_idx, coeff in term['A']:
                for i in range(4):
                    val = U_s[i, l] * coeff
                    if val != 0:
                        idx = map_A(i, block_idx)
                        U_444[idx, col] += val
                        
            # V
            for block_idx, coeff in term['B']:
                for j in range(4):
                    val = V_s[j, l] * coeff
                    if val != 0:
                        idx = map_B(j, block_idx)
                        V_444[idx, col] += val
                        
            # W
            for c_block_idx, coeff in term['C'].items():
                for k in range(4):
                    val = W_s[k, l] * coeff
                    if val != 0:
                        idx = map_C(k, c_block_idx)
                        W_444[idx, col] += val

    # Part 2: Schoolbook for 4x4 * 4x3
    # A is 4x4. B is 4x3 (cols 4..6). C is 4x3 (cols 4..6).
    # Rank 4*4*3 = 48.
    U_443 = np.zeros((16, 48))
    V_443 = np.zeros((28, 48))
    W_443 = np.zeros((28, 48))
    
    col = 0
    for i in range(4): # Row of A, Row of C
        for j in range(4): # Col of A, Row of B
            for k in range(3): # Col of B (local), Col of C (local)
                # A[i,j] * B[j, k+4] -> C[i, k+4]
                # U index: i*4 + j
                # V index: j*7 + (k+4)
                # W index: i*7 + (k+4)
                U_443[i*4 + j, col] = 1
                V_443[j*7 + (k+4), col] = 1
                W_443[i*7 + (k+4), col] = 1
                col += 1

    # Combine
    rank = 49 + 48
    U = np.hstack([U_444, U_443])
    V = np.hstack([V_444, V_443])
    W = np.hstack([W_444, W_443])
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
