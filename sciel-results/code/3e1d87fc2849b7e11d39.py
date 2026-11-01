import time
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: Split m=5 into m1=2 and m2=3.
    # The multiplication is C = A * B.
    # A is (2, 5), B is (5, 6).
    # C[i, k] = sum_{j=0}^{4} A[i, j] B[j, k]
    # Split sum into sum_{j=0}^{1} + sum_{j=2}^{4}
    
    # Sub-problem 1: A1 (2, 2) * B1 (2, 6) -> C1 (2, 6)
    # Sub-problem 2: A2 (2, 3) * B2 (3, 6) -> C2 (2, 6)
    # C = C1 + C2
    
    # We need to decompose the tensor for (2,2,6) and (2,3,6).
    
    # --- Decomposition for (2, 2, 6) ---
    # This can be seen as two independent (2,2,3) multiplications? No, the second dimension is 6.
    # Actually, B1 is (2, 6). We can view this as two (2,2,3) multiplications if we split the output k into two groups of 3?
    # No, the tensor structure is T[i, j, k] = delta(i,i') delta(j,j') delta(k,k').
    # The rank of (n, m, p) multiplication is at most rank(n, m, p1) + rank(n, m, p2) if we split p.
    # Let's split p=6 into p1=3, p2=3.
    # Then (2, 2, 6) becomes two (2, 2, 3) multiplications.
    # Rank of (2, 2, 3) is at most 2*2*3 = 12.
    # But we can do better.
    # Actually, the standard Strassen applies to (2,2,2).
    # For (2,2,6), we can treat it as 3 independent (2,2,2) multiplications?
    # Yes, if we split k into 3 pairs.
    # So Rank(2,2,6) <= 3 * Rank(2,2,2) = 3 * 7 = 21.
    # Schoolbook for (2,2,6) is 2*2*6 = 24.
    # So we save 3 ranks here.
    
    # --- Decomposition for (2, 3, 6) ---
    # Split p=6 into p1=3, p2=3.
    # We have two (2, 3, 3) multiplications.
    # Rank of (2, 3, 3)?
    # Schoolbook is 2*3*3 = 18.
    # Can we do better?
    # There is a known decomposition for (2,3,3) or we can just use schoolbook for this part if we can't find a better one easily.
    # However, the inspiration says: "Splitting ... into two independent 2x2x6 and 2x3x6 sub-problems, each using Strassen's rank-7 algorithm for their 2x2x2 sub-blocks, reduces the total rank from 59 to 58".
    # Wait, the inspiration says "2x2x6 and 2x3x6".
    # Let's re-read carefully: "each using Strassen's rank-7 algorithm for their 2x2x2 sub-blocks".
    # This implies that within the 2x2x6, we use Strassen for 2x2x2 blocks.
    # And within the 2x3x6, we use Strassen for 2x2x2 blocks?
    # A 2x3x6 multiplication can be viewed as a 2x2x6 + 2x1x6?
    # Or maybe the split is different.
    
    # Let's look at the standard rank bounds.
    # R(2,2,2) = 7 (Strassen)
    # R(2,2,6) <= 3 * 7 = 21. (Splitting k into 3 pairs of 2)
    # R(2,3,6) ?
    # We can split the m=3 dimension into 2+1.
    # R(2,3,6) <= R(2,2,6) + R(2,1,6).
    # R(2,1,6) = 2*1*6 = 12 (Schoolbook is optimal for 1-dim inner product? No, rank is n*p if m=1? Yes, C[i,k] = A[i] B[k]. Rank is n*p. Here 2*6=12).
    # So R(2,3,6) <= 21 + 12 = 33.
    # Total Rank <= 21 + 33 = 54?
    # This is much lower than 58. Why did the inspiration say 58?
    # Maybe the inspiration used a less optimal decomposition for the 2x3x6 part, or I am misinterpreting "2x2x2 sub-blocks".
    
    # Let's try to construct a rank 54 solution.
    # If we can get 54, that's a huge jump from 60.
    
    # Construction Plan:
    # 1. Define Strassen rank-7 decomposition for (2,2,2).
    #    Indices: i in {0,1}, j in {0,1}, k in {0,1}.
    #    U, V, W of shape (4,7), (4,7), (4,7).
    #    The 7 terms are:
    #    M1 = (A00+A11)(B00+B11)
    #    M2 = (A10+A11)B00
    #    M3 = A00(B10-B11)
    #    M4 = A11(B01-B00)
    #    M5 = (A00+A01)B11
    #    M6 = (A10-A00)(B00+B01)
    #    M7 = (A01-A11)(B10+B11)
    
    #    Let's write down the vectors for U, V, W for these 7 terms.
    #    Term 1: U: [1,0,0,1], V: [1,0,0,1], W: [1,1,1,1] (Output C00, C01, C10, C11? No, W index is k*n+i. For 2x2, k in 0,1, i in 0,1. Indices: 00->0, 01->1, 10->2, 11->3.
    #    Wait, W[k*n+i].
    #    C00 uses M1, M3, M4, M5?
    #    Let's verify Strassen formulas for C:
    #    C00 = M1 - M4 + M5 - M7
    #    C01 = M2 + M4
    #    C10 = M3 + M5
    #    C11 = M1 - M2 + M6 + M7
    
    #    So W vectors (length 4, indices 0,1,2,3 corresponding to C00,C01,C10,C11):
    #    W1 = [1, 0, 0, -1]
    #    W2 = [0, 1, 0, -1]
    #    W3 = [0, 0, 1, 0]
    #    W4 = [-1, 1, 0, 0]
    #    W5 = [1, 0, 1, 0]
    #    W6 = [0, 0, 0, 1]
    #    W7 = [-1, 0, 0, 1]
    
    #    U vectors (length 4, indices i*m+j -> i*2+j. 00,01,10,11):
    #    U1 = [1, 0, 0, 1]  (A00+A11)
    #    U2 = [0, 0, 1, 1]  (A10+A11)
    #    U3 = [1, 0, 0, 0]  (A00)
    #    U4 = [0, 0, 0, 1]  (A11)
    #    U5 = [1, 1, 0, 0]  (A00+A01)
    #    U6 = [-1, 0, 1, 0] (A10-A00)
    #    U7 = [0, 1, -1, 0] (A01-A11)
    
    #    V vectors (length 4, indices j*p+k -> j*2+k. 00,01,10,11):
    #    V1 = [1, 0, 0, 1]  (B00+B11)
    #    V2 = [1, 0, 0, 0]  (B00)
    #    V3 = [0, 0, 1, -1] (B10-B11)
    #    V4 = [-1, 1, 0, 0] (B01-B00) -> Wait, M4 = A11(B01-B00). So V4 should be B01-B00.
    #    Let's check V4 index: j=0, k=1 is B01. j=0, k=0 is B00.
    #    V4 = [-1, 1, 0, 0]
    #    V5 = [0, 0, 0, 1]  (B11)
    #    V6 = [1, 1, 0, 0]  (B00+B01)
    #    V7 = [0, 0, 1, 1]  (B10+B11)

    #    Let's double check M4.
    #    M4 = A11 * (B01 - B00).
    #    V4 corresponds to (B01 - B00).
    #    Indices for V: j*p+k. p=2.
    #    B00: j=0,k=0 -> idx 0.
    #    B01: j=0,k=1 -> idx 1.
    #    B10: j=1,k=0 -> idx 2.
    #    B11: j=1,k=1 -> idx 3.
    #    V4 = [ -1, 1, 0, 0 ]. Correct.

    #    Let's double check M3.
    #    M3 = A00 * (B10 - B11).
    #    V3 = [0, 0, 1, -1]. Correct.

    #    Let's double check M6.
    #    M6 = (A10 - A00) * (B00 + B01).
    #    U6 = [ -1, 0, 1, 0 ]. (A10 is idx 2, A00 is idx 0). Correct.
    #    V6 = [ 1, 1, 0, 0 ]. (B00 is idx 0, B01 is idx 1). Correct.

    #    Let's double check M7.
    #    M7 = (A01 - A11) * (B10 + B11).
    #    U7 = [ 0, 1, -1, 0 ]. (A01 is idx 1, A11 is idx 3). Correct.
    #    V7 = [ 0, 0, 1, 1 ]. (B10 is idx 2, B11 is idx 3). Correct.

    #    Okay, we have the (2,2,2) decomposition.
    
    #    Now, for (2,2,6):
    #    We split k (dim 6) into 3 pairs: (0,1), (2,3), (4,5).
    #    For each pair, we use the (2,2,2) decomposition.
    #    Total rank = 3 * 7 = 21.
    #    U shape: (4, 21).
    #    V shape: (12, 21). (m*p = 2*6 = 12).
    #    W shape: (12, 21). (p*n = 6*2 = 12).
    
    #    For pair t (t=0,1,2):
    #    k indices are 2t, 2t+1.
    #    The U vectors are the same for all t (since A is same).
    #    The V vectors are placed in the corresponding rows.
    #    The W vectors are placed in the corresponding columns.
    
    #    For (2,3,6):
    #    Split m=3 into m1=2 and m2=1.
    #    Part 1: (2,2,6). Rank 21.
    #    Part 2: (2,1,6). Rank 12.
    #    Total rank for (2,3,6) = 33.
    
    #    Total Rank = 21 (from 2,2,6) + 33 (from 2,3,6) = 54.
    
    #    Let's verify the dimensions and indexing for the final assembly.
    #    Global n=2, m=5, p=6.
    #    U: (10, Rank)
    #    V: (30, Rank)
    #    W: (12, Rank)
    
    #    Sub-problem 1: A1 (rows 0-1 of A, cols 0-1 of A? No, A is (2,5).
    #    A1 uses cols 0,1 of A. So rows of U1 are 0,1,2,3 (i*5+j).
    #    Wait, U index is i*m+j. m=5.
    #    A1 corresponds to j=0,1.
    #    So U1 rows: i=0,1. j=0,1.
    #    Indices: 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6.
    #    So U1 is not contiguous 0-3. It's sparse.
    
    #    Let's build the matrices explicitly.
    
    Rank_Total = 0
    U_total = []
    V_total = []
    W_total = []
    
    # Helper to add a block decomposition
    # block_U: (n_block, m_block, rank_block) ? No, let's use the pre-computed vectors.
    
    def get_strassen_222():
        U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, -1, 0]
        ], dtype=int).T # 7x4
        V = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, -1],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ], dtype=int).T # 7x4
        W = np.array([
            [1, 0, 0, -1],
            [0, 1, 0, -1],
            [0, 0, 1, 0],
            [-1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, 0, 0, 1]
        ], dtype=int).T # 7x4
        return U, V, W

    U_222, V_222, W_222 = get_strassen_222()
    
    # --- Build (2,2,6) decomposition ---
    # n=2, m=2, p=6.
    # Split p into 3 pairs.
    # For each pair t in 0..2:
    #   k_start = 2*t
    #   Use U_222 (7x4). Rows correspond to A indices (i,j) for i in 0,1, j in 0,1.
    #   Use V_222 (7x4). Rows correspond to B indices (j,k) for j in 0,1, k in 0,1.
    #   Use W_222 (7x4). Rows correspond to C indices (k,i) for k in 0,1, i in 0,1.
    
    # We will accumulate into global U, V, W.
    
    # Sub-problem 1: (2,2,6) using A cols 0,1 and B rows 0,1.
    # A indices: i in 0,1; j in 0,1. Global U rows: i*5+j.
    # B indices: j in 0,1; k in 0..5. Global V rows: j*6+k.
    # C indices: k in 0..5; i in 0,1. Global W rows: k*2+i.
    
    # We process 3 pairs of k.
    for t in range(3):
        k0 = 2*t
        k1 = 2*t + 1
        
        # U block: 7 terms. Each term has 4 non-zeros (or fewer).
        # The 4 positions in the 2x2 A block are (0,0), (0,1), (1,0), (1,1).
        # Global U rows:
        u_rows = [0*5+0, 0*5+1, 1*5+0, 1*5+1] # 0, 1, 5, 6
        
        # V block: 7 terms.
        # The 4 positions in the 2x2 B block are (0,0), (0,1), (1,0), (1,1) relative to local j,k.
        # Local j in 0,1. Local k in 0,1.
        # Global k: k0+local_k.
        # Global V rows: j*6 + (k0+local_k).
        v_rows = [
            0*6 + k0, 0*6 + k1, # j=0, k0, k1
            1*6 + k0, 1*6 + k1  # j=1, k0, k1
        ]
        
        # W block: 7 terms.
        # The 4 positions in the 2x2 C block are (0,0), (0,1), (1,0), (1,1) relative to local k,i.
        # Local k in 0,1. Local i in 0,1.
        # Global k: k0+local_k.
        # Global W rows: (k0+local_k)*2 + local_i.
        w_rows = [
            k0*2 + 0, k0*2 + 1, # k0, i=0, i=1
            k1*2 + 0, k1*2 + 1  # k1, i=0, i=1
        ]
        
        # Add these 7 terms to the total rank
        for r in range(7):
            u_vec = np.zeros(n*m)
            for idx, val in enumerate(U_222[r]):
                if val != 0:
                    u_vec[u_rows[idx]] = val
            
            v_vec = np.zeros(m*p)
            for idx, val in enumerate(V_222[r]):
                if val != 0:
                    v_vec[v_rows[idx]] = val
            
            w_vec = np.zeros(p*n)
            for idx, val in enumerate(W_222[r]):
                if val != 0:
                    w_vec[w_rows[idx]] = val
            
            U_total.append(u_vec)
            V_total.append(v_vec)
            W_total.append(w_vec)
            
    Rank_Total += 21
    
    # --- Build (2,3,6) decomposition ---
    # Uses A cols 2,3,4 and B rows 2,3,4.
    # Split m=3 into m1=2 (cols 2,3) and m2=1 (col 4).
    
    # Part 1: (2,2,6) using A cols 2,3 and B rows 2,3.
    for t in range(3):
        k0 = 2*t
        k1 = 2*t + 1
        
        # U rows: i in 0,1; j in 2,3.
        u_rows = [0*5+2, 0*5+3, 1*5+2, 1*5+3] # 2, 3, 7, 8
        
        # V rows: j in 2,3; k in 0..5.
        v_rows = [
            2*6 + k0, 2*6 + k1,
            3*6 + k0, 3*6 + k1
        ]
        
        # W rows: same as before.
        w_rows = [
            k0*2 + 0, k0*2 + 1,
            k1*2 + 0, k1*2 + 1
        ]
        
        for r in range(7):
            u_vec = np.zeros(n*m)
            for idx, val in enumerate(U_222[r]):
                if val != 0:
                    u_vec[u_rows[idx]] = val
            
            v_vec = np.zeros(m*p)
            for idx, val in enumerate(V_222[r]):
                if val != 0:
                    v_vec[v_rows[idx]] = val
            
            w_vec = np.zeros(p*n)
            for idx, val in enumerate(W_222[r]):
                if val != 0:
                    w_vec[w_rows[idx]] = val
            
            U_total.append(u_vec)
            V_total.append(v_vec)
            W_total.append(w_vec)
            
    Rank_Total += 21
    
    # Part 2: (2,1,6) using A col 4 and B row 4.
    # Schoolbook for (2,1,6). Rank = 2*1*6 = 12.
    # C[i,k] = A[i,4] * B[4,k].
    # Terms: for each i in 0,1 and k in 0..5.
    for i in range(2):
        for k in range(6):
            u_vec = np.zeros(n*m)
            u_vec[i*5 + 4] = 1
            
            v_vec = np.zeros(m*p)
            v_vec[4*6 + k] = 1
            
            w_vec = np.zeros(p*n)
            w_vec[k*2 + i] = 1
            
            U_total.append(u_vec)
            V_total.append(v_vec)
            W_total.append(w_vec)
            
    Rank_Total += 12
    
    # Total Rank = 21 + 21 + 12 = 54.
    
    # Convert to arrays
    U_arr = np.array(U_total).T
    V_arr = np.array(V_total).T
    W_arr = np.array(W_total).T
    
    # Verify
    # Identity: sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta(j1, j2)
    # Let's do a quick check on a few elements to be safe.
    # If time allows, check all.
    
    valid = True
    if time.time() - start_time < DEADLINE - 5:
        for i in range(n):
            for k in range(p):
                # Construct the matrix M_j1_j2 = sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r]
                # This should be Identity(m).
                # W_val = W[k*n+i, :]
                # U_slice = U[i*m:(i+1)*m, :]
                # V_slice for fixed k is not contiguous in V.
                # V_slice[j, r] = V[j*p+k, r]
                
                w_val = W_arr[k*n+i, :]
                
                # M is m x m
                M = np.zeros((m, m))
                for j1 in range(m):
                    u_val = U_arr[i*m+j1, :]
                    for j2 in range(m):
                        v_val = V_arr[j2*p+k, :]
                        M[j1, j2] = np.dot(u_val, v_val * w_val)
                
                if not np.allclose(M, np.eye(m)):
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback to schoolbook if construction failed (should not happen)
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n * m, rank_schoolbook))
        V_sb = np.zeros((m * p, rank_schoolbook))
        W_sb = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

