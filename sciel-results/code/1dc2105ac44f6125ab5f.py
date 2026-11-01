"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill

    # --- Schoolbook decomposition (rank = n*m*p) ---
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # --- Fast verification using einsum ---
    def verify_fast(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j."""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Check for integer/half-integer entries ---
    def is_valid_entries(*mats):
        for mat in mats:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True

    # --- Convert to list format ---
    def to_list(mat):
        # Round to nearest half-integer for cleanliness
        mat_rounded = np.round(mat * 2) / 2.0
        # Check if all are integers
        if np.allclose(mat_rounded, np.round(mat_rounded), atol=1e-9):
            return np.round(mat_rounded).astype(int).tolist()
        else:
            # Return as float list with half-integer values
            return mat_rounded.tolist()

    # --- Strassen-like decomposition for 2x2 * 2x2 block ---
    def strassen_2x2_2x2():
        """
        Returns U, V, W for multiplying a 2x2 matrix by a 2x2 matrix.
        Uses the 7-term Strassen decomposition.
        
        Input matrix A (2x2): [[a, b], [c, d]] indexed by (i,k) where i=row, k=col
        Input matrix B (2x2): [[e, f], [g, h]] indexed by (k,j) where k=row, j=col
        Output matrix C (2x2): [[A, B], [C, D]] indexed by (i,j) where i=row, j=col
        
        Strassen formulas:
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-f)
        M5 = (a+b)h
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        
        C[0,0] = M1 + M4 - M5 + M7
        C[0,1] = M3 + M5
        C[1,0] = M2 + M4
        C[1,1] = M1 - M2 + M3 + M6
        """
        r = 7
        U = np.zeros((4, r))  # 2x2 input A: rows (i,k)
        V = np.zeros((4, r))  # 2x2 input B: rows (k,j)
        W = np.zeros((4, r))  # 2x2 output C: rows (j,i) -- note: j*n+i convention
        
        # U entries: for each term, which (i,k) pairs are involved
        # M1 = (a+d)(e+h): a=(0,0), d=(1,1); e=(0,0), h=(1,1)
        U[0, 0] = 1  # a
        U[3, 0] = 1  # d
        V[0, 0] = 1  # e
        V[3, 0] = 1  # h
        
        # M2 = (c+d)e: c=(1,0), d=(1,1); e=(0,0)
        U[2, 1] = 1  # c
        U[3, 1] = 1  # d
        V[0, 1] = 1  # e
        
        # M3 = a(f-h): a=(0,0); f=(0,1), h=(1,1)
        U[0, 2] = 1  # a
        V[1, 2] = 1  # f
        V[3, 2] = -1 # h
        
        # M4 = d(g-f): d=(1,1); g=(1,0), f=(0,1)
        U[3, 3] = 1  # d
        V[2, 3] = 1  # g
        V[1, 3] = -1 # f
        
        # M5 = (a+b)h: a=(0,0), b=(0,1); h=(1,1)
        U[0, 4] = 1  # a
        U[1, 4] = 1  # b
        V[3, 4] = 1  # h
        
        # M6 = (c-a)(e+f): c=(1,0), a=(0,0); e=(0,0), f=(0,1)
        U[2, 5] = 1  # c
        U[0, 5] = -1 # a
        V[0, 5] = 1  # e
        V[1, 5] = 1  # f
        
        # M7 = (b-d)(g+h): b=(0,1), d=(1,1); g=(1,0), h=(1,1)
        U[1, 6] = 1  # b
        U[3, 6] = -1 # d
        V[2, 6] = 1  # g
        V[3, 6] = 1  # h
        
        # W entries: output C[j*n+i] = C[j*2+i]
        # C[0,0] = M1 + M4 - M5 + M7 -> index j=0, i=0 -> 0*2+0=0
        W[0, 0] = 1  # M1
        W[0, 3] = 1  # M4
        W[0, 4] = -1 # M5
        W[0, 6] = 1  # M7
        
        # C[0,1] = M3 + M5 -> index j=0, i=1 -> 0*2+1=1
        W[1, 2] = 1  # M3
        W[1, 4] = 1  # M5
        
        # C[1,0] = M2 + M4 -> index j=1, i=0 -> 1*2+0=2
        W[2, 1] = 1  # M2
        W[2, 3] = 1  # M4
        
        # C[1,1] = M1 - M2 + M3 + M6 -> index j=1, i=1 -> 1*2+1=3
        W[3, 0] = 1  # M1
        W[3, 1] = -1 # M2
        W[3, 2] = 1  # M3
        W[3, 5] = 1  # M6
        
        return U, V, W

    # --- Structured decomposition for 2x2 * 2x5 block ---
    def structured_2x2_2x5():
        """
        Returns U, V, W for multiplying a 2x2 matrix by a 2x5 matrix.
        Uses a rank-10 decomposition.
        
        Input matrix A (2x2): [[a, b], [c, d]] indexed by (i,k)
        Input matrix B (2x5): rows k=0,1; cols j=0,1,2,3,4
        Output matrix C (2x5): rows i=0,1; cols j=0,1,2,3,4
        
        We use a simple structured approach:
        For each output column j, we need C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
        
        We can use 2 terms per column, giving rank 10 total.
        """
        r = 10
        U = np.zeros((4, r))
        V = np.zeros((10, r))  # 2x5: rows (k,j)
        W = np.zeros((10, r))  # 2x5 output: rows (j,i)
        
        # For each column j (0..4) and each row i (0..1):
        # C[i,j] = a*B[0,j] + c*B[1,j] if i=0
        # C[i,j] = b*B[0,j] + d*B[1,j] if i=1
        
        # We'll use 2 terms per column:
        # Term 2*j:   U[i*2+0, 2*j] = 1 for i=0 (a), U[i*2+1, 2*j] = 1 for i=1 (b)
        #             V[0*5+j, 2*j] = 1 (B[0,j]), V[1*5+j, 2*j] = 0
        #             W[j*2+0, 2*j] = 1, W[j*2+1, 2*j] = 0
        # Term 2*j+1: U[i*2+0, 2*j+1] = 1 for i=0 (c), U[i*2+1, 2*j+1] = 1 for i=1 (d)
        #             V[0*5+j, 2*j+1] = 0, V[1*5+j, 2*j+1] = 1 (B[1,j])
        #             W[j*2+0, 2*j+1] = 1, W[j*2+1, 2*j+1] = 0
        
        for j in range(5):
            # Term 2*j: contributes a*B[0,j] to C[0,j] and b*B[0,j] to C[1,j]
            U[0, 2*j] = 1    # a
            U[1, 2*j] = 1    # b
            U[2, 2*j] = 0    # c
            U[3, 2*j] = 0    # d
            V[0*5+j, 2*j] = 1  # B[0,j]
            V[1*5+j, 2*j] = 0  # B[1,j]
            W[j*2+0, 2*j] = 1  # C[0,j]
            W[j*2+1, 2*j] = 1  # C[1,j]
            
            # Term 2*j+1: contributes c*B[1,j] to C[0,j] and d*B[1,j] to C[1,j]
            U[0, 2*j+1] = 0   # a
            U[1, 2*j+1] = 0   # b
            U[2, 2*j+1] = 1   # c
            U[3, 2*j+1] = 1   # d
            V[0*5+j, 2*j+1] = 0  # B[0,j]
            V[1*5+j, 2*j+1] = 1  # B[1,j]
            W[j*2+0, 2*j+1] = 1  # C[0,j]
            W[j*2+1, 2*j+1] = 1  # C[1,j]
        
        return U, V, W

    # --- Combine Strassen (2x2*2x2) and structured (2x2*2x5) ---
    def combined_decomposition():
        """
        Combine the Strassen decomposition for columns 0-1 and the structured
        decomposition for columns 2-6.
        """
        U_s, V_s, W_s = strassen_2x2_2x2()
        U_st, V_st, W_st = structured_2x2_2x5()
        
        r_s = U_s.shape[1]  # 7
        r_st = U_st.shape[1]  # 10
        r_total = r_s + r_st  # 17
        
        U = np.zeros((4, r_total))
        V = np.zeros((14, r_total))  # 2x7: rows (k,j)
        W = np.zeros((14, r_total))  # 2x7 output: rows (j,i)
        
        # Strassen part: columns 0-1
        U[:, 0:r_s] = U_s
        # V_s is for 2x2, so rows 0-1 correspond to k=0,1 and j=0,1
        V[0*7+0:2*7+2, 0:r_s] = V_s.reshape(2, 2, r_s).transpose(1, 0, 2).reshape(-1, r_s)
        # Actually, let's be more careful. V_s has shape (4, 7) where rows are (k,j) for k=0,1 and j=0,1
        # So row 0 = (k=0, j=0), row 1 = (k=0, j=1), row 2 = (k=1, j=0), row 3 = (k=1, j=1)
        # In the full V, row index = k*7 + j
        # So (k=0,j=0) -> 0, (k=0,j=1) -> 1, (k=1,j=0) -> 7, (k=1,j=1) -> 8
        V[0, 0:r_s] = V_s[0, :]   # (k=0, j=0)
        V[1, 0:r_s] = V_s[1, :]   # (k=0, j=1)
        V[7, 0:r_s] = V_s[2, :]   # (k=1, j=0)
        V[8, 0:r_s] = V_s[3, :]   # (k=1, j=1)
        
        # W_s is for 2x2 output, rows (j,i) for j=0,1 and i=0,1
        # Row 0 = (j=0, i=0), row 1 = (j=0, i=1), row 2 = (j=1, i=0), row 3 = (j=1, i=1)
        # In full W, row index = j*2 + i
        W[0, 0:r_s] = W_s[0, :]   # (j=0, i=0)
        W[1, 0:r_s] = W_s[1, :]   # (j=0, i=1)
        W[2, 0:r_s] = W_s[2, :]   # (j=1, i=0)
        W[3, 0:r_s] = W_s[3, :]   # (j=1, i=1)
        
        # Structured part: columns 2-6
        U[:, r_s:r_total] = U_st
        # V_st has shape (10, 10) where rows are (k,j) for k=0,1 and j=0..4
        # Row 0 = (k=0, j=0), row 1 = (k=0, j=1), ..., row 4 = (k=0, j=4)
        # Row 5 = (k=1, j=0), row 6 = (k=1, j=1), ..., row 9 = (k=1, j=4)
        # In full V, row index = k*7 + j, but we want j=2..6
        # So (k=0, j=2) -> 2, (k=0, j=3) -> 3, ..., (k=0, j=6) -> 6
        #    (k=1, j=2) -> 9, (k=1, j=3) -> 10, ..., (k=1, j=6) -> 13
        for jj in range(5):
            j_full = jj + 2  # actual column index in full problem
            V[0*7 + j_full, r_s:r_total] = V_st[jj, :]       # (k=0, j=jj)
            V[1*7 + j_full, r_s:r_total] = V_st[5 + jj, :]   # (k=1, j=jj)
        
        # W_st has shape (10, 10) where rows are (j,i) for j=0..4 and i=0,1
        # Row 0 = (j=0, i=0), row 1 = (j=0, i=1), ..., row 9 = (j=4, i=1)
        # In full W, row index = j*2 + i, with j=2..6
        # So (j=2, i=0) -> 4, (j=2, i=1) -> 5, ..., (j=6, i=1) -> 13
        for jj in range(5):
            j_full = jj + 2  # actual column index in full problem
            W[j_full*2 + 0, r_s:r_total] = W_st[jj*2 + 0, :]   # (j=jj, i=0)
            W[j_full*2 + 1, r_s:r_total] = W_st[jj*2 + 1, :]   # (j=jj, i=1)
        
        return U, V, W

    # --- Try the combined decomposition ---
    if time.time() < deadline:
        U_c, V_c, W_c = combined_decomposition()
        if verify_fast(U_c, V_c, W_c) and is_valid_entries(U_c, V_c, W_c):
            result = {
                "U": to_list(U_c),
                "V": to_list(V_c),
                "W": to_list(W_c)
            }
            best = result
            best_rank = U_c.shape[1]
    
    # --- If combined didn't work or we want to try lower, do random search ---
    def try_lower_rank(r, max_attempts=50):
        """Try to find a rank-r decomposition by random search."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + r)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random sparse matrices with entries in {-1, 0, 1}
            U_mat = np.zeros((n * m, r))
            V_mat = np.zeros((m * p, r))
            
            for c in range(r):
                for i in range(n):
                    for k in range(m):
                        if np.random.random() < 0.7:
                            U_mat[i * m + k, c] = np.random.choice([-1, 1])
                
                for k in range(m):
                    for j in range(p):
                        if np.random.random() < 0.7:
                            V_mat[k * p + j, c] = np.random.choice([-1, 1])
            
            W_mat = np.zeros((p * n, r))
            consistent = True
            
            for j in range(p):
                for i in range(n):
                    A_row = np.zeros((m, r))
                    for k in range(m):
                        for c in range(r):
                            A_row[k, c] = U_mat[i * m + k, c] * V_mat[k * p + j, c]
                    
                    b_row = np.ones(m)
                    
                    try:
                        sol, residuals, rank_A, s = LA.lstsq(A_row, b_row, rcond=None)
                        if not np.allclose(A_row @ sol, b_row, atol=1e-9):
                            consistent = False
                            break
                        W_mat[j * n + i, :] = sol
                    except:
                        consistent = False
                        break
                
                if not consistent:
                    break
            
            if not consistent:
                continue
            
            if verify_fast(U_mat, V_mat, W_mat):
                if is_valid_entries(U_mat, V_mat, W_mat):
                    return {
                        "U": to_list(U_mat),
                        "V": to_list(V_mat),
                        "W": to_list(W_mat)
                    }
        
        return None

    # --- Try ranks from low to high if combined didn't give best ---
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(4, 14, 14) = 14
    for r in range(14, best_rank):
        if time.time() > deadline:
            break
        
        result = try_lower_rank(r, max_attempts=30)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
# EVOLVE-BLOCK-END

