import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Schoolbook: n*m*p = 36
    # Current best known: 33 (Strassen on 2x2x2 blocks + schoolbook remainder)
    # We aim to find a better decomposition.
    
    # Strategy:
    # 1. Start with a known valid decomposition (schoolbook or Strassen-based)
    # 2. Try to reduce rank via local search / column elimination
    # 3. Try different structural decompositions
    
    # First, let's build the schoolbook decomposition (rank 36)
    def build_schoolbook(n, m, p):
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
        return U, V, W, rank
    
    def build_strassen_2x2x2(n, m, p):
        """
        Use Strassen for 2x2x2 sub-blocks where possible.
        For 2x3x6: split into blocks.
        
        One approach: treat the 3x6 as three 3x2 blocks, and 2x3 as three 2x1 blocks.
        But Strassen works on 2x2. Let's try a different decomposition.
        
        Actually, let's use the known rank 33 construction:
        - Split the 3 columns of A into: 2 columns (for Strassen) + 1 column
        - Split the 6 columns of B into: 3 pairs (for Strassen on 2x2x2) 
        - The 2x2 part uses Strassen (rank 7)
        - The remaining parts use schoolbook
        
        More carefully:
        A is 2x3, B is 3x6.
        Write A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
        Write B = [B1; B2; B3] where B1, B2, B3 are 2x6, 1x6, ... no.
        
        Let me think differently. 
        C = A * B = A1*B1 + A2*B2 + A3*B3 where B is split into 3 row-blocks of 2x6.
        But A is 2x3, so A has 3 columns. B has 3 rows.
        C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
        
        Split: C = A[:,0:2] @ B[0:2,:] + A[:,2:3] @ B[2:3,:]
        First term: 2x2 @ 2x6 -> use Strassen on 2x2x2 blocks? 
        2x2 @ 2x6 = 3 blocks of 2x2 @ 2x2, each rank 7 -> total 21
        Second term: 2x1 @ 1x6 -> rank 12 (schoolbook)
        Total: 33
        """
        rank = 33
        U = np.zeros((n*m, rank))  # 6 x 33
        V = np.zeros((m*p, rank))  # 18 x 33
        W = np.zeros((p*n, rank))  # 12 x 33
        
        col = 0
        # Part 1: A[:,0:2] @ B[0:2,:] using Strassen on 3 blocks of 2x2x2
        # A[:,0:2] is 2x2, B[0:2,:] is 2x6 = three 2x2 blocks
        for block in range(3):
            # B block: rows 0,1, cols 2*block to 2*block+1
            # Strassen for 2x2x2:
            # M1 = (a00+a11)(b00+b11)
            # M2 = (a10+a11)b00
            # M3 = a00(b01-b11)
            # M4 = a11(b10-b00)
            # M5 = (a00+a01)b11
            # M6 = (a10-a00)(b00+b01)
            # M7 = (a01-a11)(b10+b11)
            
            b_col_offset = 2 * block
            
            # M1: (a00+a11)(b00+b11)
            U[0, col] = 1  # a00
            U[3, col] = 1  # a11
            V[b_col_offset, col] = 1  # b00
            V[b_col_offset + 1, col] = 1  # b11
            W[2*b_col_offset, col] = 1  # c00
            W[2*b_col_offset + 1, col] = 1  # c01
            W[2*b_col_offset + 2, col] = 1  # c10
            W[2*b_col_offset + 3, col] = 1  # c11
            col += 1
            
            # M2: (a10+a11)b00
            U[1, col] = 1  # a10
            U[3, col] = 1  # a11
            V[b_col_offset, col] = 1  # b00
            W[2*b_col_offset + 1, col] = 1  # c01
            W[2*b_col_offset + 3, col] = -1  # c11
            col += 1
            
            # M3: a00(b01-b11)
            U[0, col] = 1  # a00
            V[b_col_offset + 1, col] = 1  # b01
            V[b_col_offset + 1, col] = 1  # wait, b01 is row 0, col 1 of block
            # B block: row 0, col 2*block+1 -> index 0*(6) + 2*block+1 = 2*block+1
            # B block: row 1, col 2*block+1 -> index 1*(6) + 2*block+1 = 6 + 2*block+1
            V[b_col_offset + 1, col] = 1  # b01 (row 0, col 1 within block)
            V[6 + b_col_offset + 1, col] = -1  # b11 (row 1, col 1 within block)
            W[2*b_col_offset + 2, col] = 1  # c10
            W[2*b_col_offset + 3, col] = 1  # c11
            col += 1
            
            # M4: a11(b10-b00)
            U[3, col] = 1  # a11
            V[6 + b_col_offset, col] = 1  # b10
            V[b_col_offset, col] = -1  # b00
            W[2*b_col_offset, col] = 1  # c00
            W[2*b_col_offset + 1, col] = 1  # c01
            col += 1
            
            # M5: (a00+a01)b11
            U[0, col] = 1  # a00
            U[2, col] = 1  # a01
            V[6 + b_col_offset + 1, col] = 1  # b11
            W[2*b_col_offset, col] = -1  # c00
            W[2*b_col_offset + 2, col] = 1  # c10
            col += 1
            
            # M6: (a10-a00)(b00+b01)
            U[1, col] = 1  # a10
            U[0, col] = -1  # a00
            V[b_col_offset, col] = 1  # b00
            V[b_col_offset + 1, col] = 1  # b01
            W[2*b_col_offset + 3, col] = 1  # c11
            col += 1
            
            # M7: (a01-a11)(b10+b11)
            U[2, col] = 1  # a01
            U[3, col] = -1  # a11
            V[6 + b_col_offset, col] = 1  # b10
            V[6 + b_col_offset + 1, col] = 1  # b11
            W[2*b_col_offset, col] = 1  # c00
            col += 1
        
        # Part 2: A[:,2:3] @ B[2:3,:] schoolbook (2x1 @ 1x6)
        # A column 2: indices 2, 3 in U
        # B row 2: indices 12,13,14,15,16,17 in V
        for k in range(p):
            U[2, col] = 1  # a[0,2]
            V[12 + k, col] = 1  # b[2,k]
            W[k*2 + 0, col] = 1  # c[0,k]
            col += 1
            U[3, col] = 1  # a[1,2]
            V[12 + k, col] = 1  # b[2,k]
            W[k*2 + 1, col] = 1  # c[1,k]
            col += 1
        
        return U, V, W, col
    
    # Build initial solution
    U, V, W, rank = build_strassen_2x2x2(n, m, p)
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if decomposition is valid."""
        r = U_mat.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for rr in range(r):
                        actual += U_mat[i*m+j, rr] * V_mat[j*p+k, rr] * W_mat[k*n+i, rr]
                    err = abs(actual - expected)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    # Verify initial solution
    if not verify(U, V, W):
        # Fallback to schoolbook
        U, V, W, rank = build_schoolbook(n, m, p)
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Try to reduce rank by eliminating columns
    def try_eliminate(U_cur, V_cur, W_cur, col_idx):
        """Try to remove column col_idx by seeing if the tensor can still be represented."""
        mask = np.ones(U_cur.shape[1], dtype=bool)
        mask[col_idx] = False
        U_test = U_cur[:, mask]
        V_test = V_cur[:, mask]
        W_test = W_cur[:, mask]
        if verify(U_test, V_test, W_test):
            return True, U_test, V_test, W_test
        return False, None, None, None
    
    # Try to find a lower rank decomposition
    # Approach: random perturbation + column elimination
    np.random.seed(42)
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    
    # Try multiple random initializations and local search
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        # Try to eliminate columns one by one
        improved = True
        while improved:
            improved = False
            for col_idx in range(current_rank):
                if time.time() > deadline - 10:
                    break
                success, U_new, V_new, W_new = try_eliminate(current_U, current_V, current_W, col_idx)
                if success:
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank -= 1
                    improved = True
                    break
        
        if current_rank < best_rank:
            best_U, best_V, best_W, best_rank = current_U.copy(), current_V.copy(), current_W.copy(), current_rank
    
    # Also try a different structural approach: 
    # Use rank 7 decomposition for 2x2x3 (if we can find one)
    # 2x2x3 tensor: U is 4x7, V is 6x7, W is 6x7
    # Let's try to find it via random search
    
    def find_2x2x3_decomposition(max_iters=1000):
        """Find rank 7 decomposition for 2x2x3 tensor."""
        n2, m2, p2 = 2, 2, 3
        rank2 = 7
        np.random.seed(123)
        
        for it in range(max_iters):
            U2 = np.random.randn(n2*m2, rank2)
            V2 = np.random.randn(m2*p2, rank2)
            W2 = np.random.randn(p2*n2, rank2)
            
            # Check if this gives a valid decomposition
            # T[i,j,k] = sum_r U2[i*m2+j, r] * V2[j*p2+k, r] * W2[k*n2+i, r] = 1
            max_err = 0.0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        actual = 0.0
                        for r in range(rank2):
                            actual += U2[i*m2+j, r] * V2[j*p2+k, r] * W2[k*n2+i, r]
                        err = abs(actual - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                return U2, V2, W2
        
        return None, None, None
    
    # Try to find 2x2x3 decomposition
    U2, V2, W2 = find_2x2x3_decomposition(500)
    
    if U2 is not None:
        # Build 2x2x6 as two 2x2x3 blocks
        # Total rank: 2*7 + 12 = 26
        new_rank = 26
        U_new = np.zeros((n*m, new_rank))
        V_new = np.zeros((m*p, new_rank))
        W_new = np.zeros((p*n, new_rank))
        
        col = 0
        # Block 1: cols 0-1 of B
        U_new[:, col:col+7] = U2
        V_new[:6, col:col+7] = V2
        W_new[:6, col:col+7] = W2
        col += 7
        
        # Block 2: cols 2-3 of B
        U_new[:, col:col+7] = U2
        V_new[6:12, col:col+7] = V2
        W_new[6:12, col:col+7] = W2
        col += 7
        
        # Block 3: cols 4-5 of B
        U_new[:, col:col+7] = U2
        V_new[12:18, col:col+7] = V2
        W_new[12:18, col:col+7] = W2
        col += 7
        
        # Wait, this doesn't work because A has 3 columns, not 2.
        # We need to handle the 3rd column separately.
        # This approach is flawed. Skip for now.
        pass
    
    # Convert to lists and return
    U_list = [[float(x) for x in best_U[i]] for i in range(n*m)]
    V_list = [[float(x) for x in best_V[i]] for i in range(m*p)]
    W_list = [[float(x) for x in best_W[i]] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
