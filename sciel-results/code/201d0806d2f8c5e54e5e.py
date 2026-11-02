import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook fallback (rank = 60)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_s, V_s, W_s, rank_s = schoolbook()
    best_result = (U_s, V_s, W_s, rank_s)
    
    # Verify a decomposition: check T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    def verify(U, V, W, r, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # WFTA-based decomposition for 2x3x10
    # Split: A (2x3) = [A_left (2x2) | A_right (2x1)]
    #        B (3x10) = [B_top (2x10); B_bottom (1x10)]
    # C = A_left * B_top + A_right * B_bottom
    # B_top (2x10) split into 5 blocks of 2x2: B_top = [B_t1 | B_t2 | ... | B_t5]
    # B_bottom (1x10) split into 5 blocks of 1x2: B_b1, ..., B_b5
    #
    # 2x2x2 sub-problems: rank 7 (Strassen)
    # 2x1x2 sub-problems: rank 2
    # Total: 5*7 + 5*2 = 35 + 10 = 45
    
    def wfta_decomposition():
        # Strassen rank-7 decomposition for 2x2x2
        # C = A * B where A, B are 2x2
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For a 2x2x2 block, the rank-1 terms (U_col, V_col, W_col) are:
        # Each term is a product of a row of A, a column of B, and a row of C
        # Term 1: A factor (1,1) on rows (0,1), B factor (1,1) on cols (0,1), C factor (1,-1,1,1)
        # Actually let me think more carefully.
        #
        # For 2x2x2, the tensor T[i,j,k] = delta[i+k, j] (mod 2) for the standard MM
        # Wait, no. T[i,j,k] = 1 if i,j,k are valid indices, and the multiplication is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1}
        # The tensor is just all-ones in 2x2x2.
        #
        # Strassen's 7 terms for 2x2x2:
        # The rank-1 tensors are:
        # t1: A-rows (1,1), B-cols (1,1), C-entries (1, 0, 0, 1) -> C00 += 1, C11 += 1
        # t2: A-rows (0,1), B-cols (1,0), C-entries (0, 0, 1, 0) -> C10 += 1
        # t3: A-rows (1,0), B-cols (1,1), C-entries (1, 1, 0, 0) -> C00 += 1, C01 += 1
        # t4: A-rows (0,1), B-cols (0,1), C-entries (1, 1, 0, 0) -> C00 += 1, C01 += 1
        # t5: A-rows (1,0), B-cols (1,0), C-entries (1, 0, 0, 1) -> C00 += 1, C11 += 1
        # t6: A-rows (0,1), B-cols (1,1), C-entries (0, 1, 0, 1) -> C01 += 1, C11 += 1
        # t7: A-rows (1,0), B-cols (1,0), C-entries (1, 1, 0, 0) -> C00 += 1, C01 += 1
        #
        # Let me just use the standard Strassen formulas directly.
        #
        # For 2x2x2, the 7 rank-1 terms (as U, V, W vectors):
        # U is 4-dim (i*m+j = i*2+j), V is 4-dim (j*p+k = j*2+k), W is 4-dim (k*n+i = k*2+i)
        #
        # Term 1: M1 = (A00+A11)*(B00+B11)
        #   A-rows: [1, 0, 0, 1] (A00 and A11)
        #   B-cols: [1, 0, 1, 0] (B00 and B11)
        #   C-entries: C00 += 1, C11 += 1 -> [1, 0, 0, 1] (k*n+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3)
        #   Wait, W index is k*n+i = k*2+i
        #   C00: k=0,i=0 -> W[0] += 1
        #   C11: k=1,i=1 -> W[3] += 1
        #   So W = [1, 0, 0, 1]
        
        # Let me define the 7 Strassen terms for 2x2x2:
        # U indices: i*2+j, so (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V indices: j*2+k, so (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W indices: k*2+i, so (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        strassen_2x2x2 = [
            # (U, V, W) for each of the 7 terms
            # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
            (np.array([1, 0, 0, 1]), np.array([1, 0, 1, 0]), np.array([1, 0, 0, 1])),
            # M2 = (A10+A11)*B00 -> C10 += 1
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])),
            # M3 = A00*(B01-B11) -> C01 += 1, C11 += -1
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, -1])),
            # M4 = A11*(B10-B00) -> C00 += 1, C10 += 1
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])),
            # M5 = (A00+A01)*B11 -> C00 += -1, C01 += 1
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
            # M6 = (A10-A00)*(B00+B01) -> C11 += 1
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            # M7 = (A01-A11)*(B10+B11) -> C00 += 1, C10 += -1
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, -1, 0])),
        ]
        
        # 2x1x2 rank-2 decomposition
        # A is 2x1, B is 1x2, C is 2x2
        # C[i,k] = A[i,0] * B[0,k]
        # This is just: C00 = A00*B00, C01 = A00*B01, C10 = A10*B00, C11 = A10*B01
        # Rank-1 terms:
        # t1: A-rows [1,0], B-cols [1,0], C-entries [1,0,0,0] -> C00
        # t2: A-rows [1,0], B-cols [0,1], C-entries [0,1,0,0] -> C01
        # t3: A-rows [0,1], B-cols [1,0], C-entries [0,0,1,0] -> C10
        # t4: A-rows [0,1], B-cols [0,1], C-entries [0,0,0,1] -> C11
        # But rank 2! We can do better.
        # Actually, 2x1x2 has rank 2.
        # t1: A-rows [1,1], B-cols [1,0], C-entries [1,0,1,0] -> C00 += 1, C10 += 1
        # t2: A-rows [1,1], B-cols [0,1], C-entries [0,1,0,1] -> C01 += 1, C11 += 1
        # Check: C00 = 1*1*1 = 1, C01 = 1*1*0 + 1*0*1 = 0... no.
        #
        # Let me think again. For 2x1x2:
        # T[i,j,k] where i in {0,1}, j in {0}, k in {0,1}
        # T[0,0,0] = 1, T[0,0,1] = 1, T[1,0,0] = 1, T[1,0,1] = 1
        # U is 2-dim, V is 1-dim, W is 4-dim (k*2+i)
        # Rank-1: U=[1,0], V=[1], W=[1,0,0,0] -> T[0,0,0]=1
        # Rank-1: U=[0,1], V=[1], W=[0,0,1,0] -> T[1,0,0]=1
        # That's rank 2 for the two "0" columns of B.
        # Similarly for k=1: U=[1,0], V=[1], W=[0,1,0,0] and U=[0,1], V=[1], W=[0,0,0,1]
        # Total rank 4? No, that's schoolbook.
        #
        # Actually the rank of 2x1x2 is 2.
        # t1: U=[1,1], V=[1], W=[1,0,1,0] -> T[0,0,0]=1, T[1,0,0]=1
        # t2: U=[1,1], V=[1], W=[0,1,0,1] -> T[0,0,1]=1, T[1,0,1]=1
        # Yes! Rank 2.
        
        rank2_2x1x2 = [
            (np.array([1, 1]), np.array([1]), np.array([1, 0, 1, 0])),
            (np.array([1, 1]), np.array([1]), np.array([0, 1, 0, 1])),
        ]
        
        # Now build the full 2x3x10 decomposition
        # U is 6-dim (i*3+j), V is 30-dim (j*10+k), W is 20-dim (k*2+i)
        
        U = np.zeros((nm, 45))
        V = np.zeros((mp, 45))
        W = np.zeros((pn, 45))
        
        col = 0
        
        # 5 blocks of 2x2x2 (for A_left (2x2) * B_top block (2x2))
        for block in range(5):
            # B_top block: rows 0,1 of B, cols 2*block, 2*block+1
            for u_idx, v_idx, w_idx in strassen_2x2x2:
                # U: A_left rows (2x2) -> indices i*3+j for i in {0,1}, j in {0,1}
                U[0*3+0, col] = u_idx[0]  # A[0,0]
                U[0*3+1, col] = u_idx[1]  # A[0,1]
                U[1*3+0, col] = u_idx[2]  # A[1,0]
                U[1*3+1, col] = u_idx[3]  # A[1,1]
                
                # V: B_top block (2x2) -> rows 0,1 of B, cols 2*block, 2*block+1
                # V index: j*10+k
                V[0*10 + 2*block, col] = v_idx[0]    # B[0, 2*block]
                V[0*10 + 2*block+1, col] = v_idx[1]  # B[0, 2*block+1]
                V[1*10 + 2*block, col] = v_idx[2]    # B[1, 2*block]
                V[1*10 + 2*block+1, col] = v_idx[3]  # B[1, 2*block+1]
                
                # W: C entries (2x2) -> k in {2*block, 2*block+1}, i in {0,1}
                # W index: k*2+i
                W[(2*block)*2 + 0, col] = w_idx[0]    # C[0, 2*block]
                W[(2*block)*2 + 1, col] = w_idx[1]    # C[1, 2*block]
                W[(2*block+1)*2 + 0, col] = w_idx[2]  # C[0, 2*block+1]
                W[(2*block+1)*2 + 1, col] = w_idx[3]  # C[1, 2*block+1]
                
                col += 1
        
        # 5 blocks of 2x1x2 (for A_right (2x1) * B_bottom block (1x2))
        for block in range(5):
            # B_bottom block: row 2 of B, cols 2*block, 2*block+1
            for u_idx, v_idx, w_idx in rank2_2x1x2:
                # U: A_right (2x1) -> indices i*3+2 for i in {0,1}
                U[0*3+2, col] = u_idx[0]  # A[0,2]
                U[1*3+2, col] = u_idx[1]  # A[1,2]
                
                # V: B_bottom block (1x2) -> row 2 of B, cols 2*block, 2*block+1
                V[2*10 + 2*block, col] = v_idx[0]    # B[2, 2*block]
                V[2*10 + 2*block+1, col] = v_idx[1]  # B[2, 2*block+1]
                
                # W: C entries (2x2) -> k in {2*block, 2*block+1}, i in {0,1}
                W[(2*block)*2 + 0, col] = w_idx[0]    # C[0, 2*block]
                W[(2*block)*2 + 1, col] = w_idx[1]    # C[1, 2*block]
                W[(2*block+1)*2 + 0, col] = w_idx[2]  # C[0, 2*block+1]
                W[(2*block+1)*2 + 1, col] = w_idx[3]  # C[1, 2*block+1]
                
                col += 1
        
        return U, V, W, 45
    
    # Try WFTA decomposition
    if time.time() < deadline - 5:
        U_w, V_w, W_w, rank_w = wfta_decomposition()
        err = verify(U_w, V_w, W_w, rank_w)
        if err < 1e-10:
            best_result = (U_w, V_w, W_w, rank_w)
        else:
            # WFTA failed verification, try to fix with least squares
            # For each (k,i), solve for W[k*n+i,:] given U,V
            W_fixed = np.zeros((pn, rank_w))
            success = True
            for k in range(p):
                for i in range(n):
                    A_ki = np.zeros((m, rank_w))
                    for j in range(m):
                        A_ki[j, :] = U_w[i*m + j, :] * V_w[j*p + k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W_fixed[k*n + i, :] = w
                    except np.linalg.LinAlgError:
                        success = False
                        break
                if not success:
                    break
            
            if success:
                err_fixed = verify(U_w, V_w, W_fixed, rank_w)
                if err_fixed < 1e-10:
                    best_result = (U_w, V_w, W_fixed, rank_w)
    
    # Try to find even lower rank with random search
    if time.time() < deadline - 10:
        np.random.seed(42)
        current_best_rank = best_result[3]
        
        # Try ranks from 3 to current_best_rank - 1
        for r in range(3, current_best_rank):
            if time.time() > deadline - 5:
                break
            
            found = False
            max_restarts = 50 if r <= 15 else 20
            
            for trial in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.zeros((pn, r))
                success = True
                
                for k in range(p):
                    for i in range(n):
                        A_ki = np.zeros((m, r))
                        for j in range(m):
                            A_ki[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        try:
                            w, _, _, _ = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                            W[k*n + i, :] = w
                        except np.linalg.LinAlgError:
                            success = False
                            break
                    if not success:
                        break
                
                if not success:
                    continue
                
                err = verify(U, V, W, r)
                if err < 1e-10:
                    best_result = (U, V, W, r)
                    found = True
                    break
            
            if found:
                break
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
