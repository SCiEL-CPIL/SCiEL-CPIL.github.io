import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build the constraint matrix C: for each column c, the bilinear form
    # sum_k U[i,k]*V[j,k]*W[k,l] must equal delta_{i,j+l} (for valid (i,j,l))
    # The tensor T[i,j,l] = 1 if i = j+l else 0, for i in [0,n), j in [0,m), l in [0,p)
    # Actually: C[i,j,l] = sum_k U[i,k]*V[j,k]*W[l,k] should equal T[i,j,l]
    # where T[i,j,l] = 1 if the output element (i,j) gets contribution from A[i,j]*B[j,l]
    
    # For 2x3x7: output C[i,j] = sum_k A[i,k]*B[k,j]
    # C[i,j] = sum over k=0..2 of A[i,k]*B[k,j]
    # In tensor form: C[i,j] = sum_{a=0}^{m-1} A[i,a] * B[a,j]
    # The tensor T has T[i,a,j] = 1 for all valid (i,a,j), else 0
    # Decomposition: T[i,a,j] = sum_c U[i,c]*V[a,c]*W[j,c]
    # Wait, let me re-read the convention: W[k*n+i, r]
    
    # Convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The multiplication: C[i,j] = sum_k A[i,k]*B[k,j]
    # In the tensor decomposition with the given convention:
    # U[i*m + a, c] * V[a*p + j, c] * W[j*n + i, c] summed over c gives C[i,j]
    # Wait, that doesn't look right. Let me think more carefully.
    
    # Standard: C[i,j] = sum_a A[i,a] * B[a,j]
    # The tensor T[i,a,j] = A[i,a]*B[a,j] summed over a gives C[i,j]
    # But the decomposition is: T[i,a,j] = sum_c U[i,a,c] * V[a,j,c] * W[?,c]
    
    # Given the convention W[k*n+i, r], where k and i are indices:
    # U has rows indexed by (i, a) where i in [0,n), a in [0,m) -> row i*m+a
    # V has rows indexed by (a, j) where a in [0,m), j in [0,p) -> row a*p+j
    # W has rows indexed by (j, i) where j in [0,p), i in [0,n) -> row j*n+i
    
    # The bilinear form for column c: sum over all (i,a,j) of U[i,a,c]*V[a,j,c]*W[j,i,c] * A[i,a]*B[a,j]
    # This should equal sum_{i,j} C[i,j] where C[i,j] = sum_a A[i,a]*B[a,j]
    # So we need: for each (i,j): sum_a sum_c U[i*m+a,c]*V[a*p+j,c]*W[j*n+i,c] = 1
    # And for each (i,a,j) triple, the coefficient of A[i,a]*B[a,j] in the expansion must match.
    
    # Actually, the tensor T[i,a,j] has T[i,a,j] = 1 for all valid indices.
    # The rank-r decomposition: T[i,a,j] = sum_{c=0}^{r-1} U[i,a,c]*V[a,j,c]*W[j,i,c]
    # Wait, W is indexed by (j,i) not (a,j). Let me re-read.
    
    # U shape (n*m, rank): row = i*m + a
    # V shape (m*p, rank): row = a*p + j  
    # W shape (p*n, rank): row = j*n + i
    
    # The product for a given column c: U[i*m+a, c] * V[a*p+j, c] * W[j*n+i, c]
    # Summed over c, this should give T[i,a,j] = 1 for all (i,a,j).
    
    # So the constraint is: for each (i,a,j) in [0,n)x[0,m)x[0,p):
    #   sum_c U[i*m+a, c] * V[a*p+j, c] * W[j*n+i, c] = 1
    
    # This is a trilinear constraint. For fixed U and W, it's linear in V, etc.
    
    # Strategy: Start with schoolbook (rank 42), then use column space reduction.
    # If we have a valid rank-r decomposition, the columns (u_c, v_c, w_c) form a set.
    # We can check if any column is in the span of others by looking at the 
    # constraint equations.
    
    # Alternative: Use the fact that the tensor has structure. For 2x3x7, 
    # known lower bounds suggest rank < 42 is possible.
    
    # Let me try a direct approach: set up the system and solve.
    
    # For a rank-r decomposition, we need 3*r variables (r columns in each of U,V,W).
    # The constraints are n*m*p = 42 trilinear equations.
    
    # Greedy approach: Start with rank 42, try to reduce by checking if columns are dependent.
    
    # Schoolbook:
    rank_schoolbook = n * m * p  # 42
    U = np.zeros((nm, rank_schoolbook))
    V = np.zeros((mp, rank_schoolbook))
    W = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for a in range(m):
            for j in range(p):
                U[i*m + a, col] = 1
                V[a*p + j, col] = 1
                W[j*n + i, col] = 1
                col += 1
    
    # Now try to reduce rank by finding linear dependencies among columns.
    # For a valid decomposition, if we replace column c with a linear combination
    # of other columns, the constraints must still hold.
    
    # The key insight: if for some column c, there exist scalars alpha_d (d != c)
    # such that for all (i,a,j):
    #   U[i*m+a, c]*V[a*p+j, c]*W[j*n+i, c] = sum_{d!=c} alpha_d * U[i*m+a,d]*V[a*p+j,d]*W[j*n+i,d]
    # then we can eliminate column c.
    
    # This is a system of 42 equations in (r-1) unknowns for each c.
    
    def is_valid(U, V, W):
        """Check if (U,V,W) exactly decomposes the multiplication tensor."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for a in range(m):
                for j in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+a, c] * V[a*p+j, c] * W[j*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_reduce_rank(U, V, W, max_time=None):
        """Try to reduce the rank by eliminating dependent columns."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        r = U.shape[1]
        
        # Precompute the "tensor" values for each column
        # col_tensor[c][i,a,j] = U[i,a,c]*V[a,j,c]*W[j,i,c]
        
        U, V, W = U.copy(), V.copy(), W.copy()
        
        reduced = True
        while reduced and r > 0:
            if max_time and time.time() - start_time > max_time:
                break
            reduced = False
            r = U.shape[1]
            
            for c in range(r):
                if time.time() - start_time > (max_time or deadline - 5):
                    return U, V, W
                
                # Try to eliminate column c: find alpha_d for d != c
                # such that for all (i,a,j):
                #   U[i,a,c]*V[a,j,c]*W[j,i,c] = sum_{d!=c} alpha_d * U[i,a,d]*V[a,j,d]*W[j,i,d]
                
                # Build the system: 42 equations, r-1 unknowns
                # For each (i,a,j): sum_{d!=c} alpha_d * (U[i,a,d]*V[a,j,d]*W[j,i,d]) = U[i,a,c]*V[a,j,c]*W[j,i,c]
                
                # If r-1 >= 42, we may have solutions. If r-1 < 42, typically no solution unless special structure.
                
                if r - 1 < n*m*p:
                    # Underdetermined or no solution typically; skip
                    continue
                
                # Build matrix A_sys: (n*m*p) x (r-1)
                rhs = np.zeros(n*m*p)
                A_sys = np.zeros((n*m*p, r-1))
                
                idx = 0
                for i in range(n):
                    for a in range(m):
                        for j in range(p):
                            rhs[idx] = U[i*m+a, c] * V[a*p+j, c] * W[j*n+i, c]
                            col_idx = 0
                            for d in range(r):
                                if d == c:
                                    continue
                                A_sys[idx, col_idx] = U[i*m+a, d] * V[a*p+j, d] * W[j*n+i, d]
                                col_idx += 1
                            idx += 1
                
                # Solve A_sys * alpha = rhs
                try:
                    alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, rhs, rcond=None)
                except:
                    continue
                
                # Check if solution is exact
                if np.allclose(A_sys @ alpha, rhs, atol=1e-9):
                    # We can eliminate column c
                    # Remove column c from U, V, W and add alpha contributions to other columns
                    # Actually, we need to verify that the decomposition still holds after removal.
                    # The trick: if the bilinear form of column c equals the sum of alpha_d * bilinear form of column d,
                    # then removing column c and not adding anything elsewhere works only if we account for it.
                    
                    # Actually, the standard approach: if column c's tensor equals sum alpha_d * tensor_d,
                    # then we can remove column c and the remaining columns still satisfy the constraints.
                    
                    # Verify by checking validity after removal
                    U_new = np.delete(U, c, axis=1)
                    V_new = np.delete(V, c, axis=1)
                    W_new = np.delete(W, c, axis=1)
                    
                    if is_valid(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        r = U.shape[1]
                        reduced = True
                        break
        
        return U, V, W
    
    U, V, W = try_reduce_rank(U, V, W, max_time=deadline - 20)
    current_rank = U.shape[1]
    
    # If we haven't improved much, try a more aggressive search with smaller initial rank.
    # Try random rank-r decompositions for r < current_rank.
    
    if current_rank == n*m*p and time.time() < deadline - 30:
        # Try to find a decomposition with rank < 42 using random initialization + Gauss-Newton
        # or try structured approaches.
        
        # Known: For 2x2x2, Strassen gives rank 7 < 8.
        # For 2x3x7, we can try to exploit block structure.
        
        # Let's try a block approach:
        # Split the 3 intermediate dimension into 2+1.
        # C[i,j] = sum_{a=0}^{1} A[i,a]*B[a,j] + A[i,2]*B[2,j]
        # First term: 2x2x7 problem. Second term: 2x1x7 (outer product, rank 14).
        # For 2x2x7: split 7 into 2+2+2+1:
        #   2x2x2 (Strassen, rank 7) x 3 + 2x2x1 (rank 4)
        #   = 3*7 + 4 = 25
        # Total: 25 + 14 = 39
        
        # Or: 2x2x7 with 7=2+2+2+1: three 2x2x2 blocks (rank 7 each) and one 2x2x1 (rank 4)
        # 2x2x1: rank 4 (schoolbook: 2*2*1=4, and it's minimal)
        # So 2x2x7 rank <= 3*7 + 4 = 25
        # 2x3x7 <= 25 + 2*1*7 = 25 + 14 = 39
        
        # Let me construct this explicitly.
        
        def construct_2x2x2_strassen():
            """Strassen decomposition for 2x2x2, rank 7."""
            # Standard Strassen:
            # M1 = (A11-A22)(B21-B22)
            # M2 = (A11+A22)(B11+B22)
            # M3 = (A11+A12)B22
            # M4 = A22(B21-B11)
            # M5 = (A11+A21)B11
            # M6 = (A12-A22)(B21+B22)
            # M7 = (A11-A12)(B11+B12)
            # C11 = M5+M4-M6-M7
            # C12 = M1+M7
            # C21 = M2+M3-M1-M5
            # C22 = M1+M6-M2-M4
            
            U = np.array([
                [1, 0, 0, 0, 1, 0, 1],   # A11
                [0, 1, 1, 0, 0, 0, 0],   # A12
                [-1, 1, 0, 1, 0, -1, -1], # A21
                [0, 1, 0, -1, 1, 1, 1],   # A22
            ])
            V = np.array([
                [0, 1, 1, 0, 1, 0, 1],   # B11
                [1, 1, 0, 1, 0, 1, 1],   # B12
                [1, 0, 1, -1, 0, 1, 0],  # B21
                [-1, 1, 1, 0, 0, 1, -1], # B22
            ])
            W = np.array([
                [-1, 0, 0, 0, 0, 1, 0],  # C11? Let me verify
                [1, 0, 0, 0, 0, 0, 1],   # C12
                [1, 1, 0, 1, 1, 0, -1],  # C21
                [1, -1, -1, -1, 0, 1, 0]  # C22
            ])
            
            # Verify
            n, m, p = 2, 2, 2
            for i in range(n):
                for a in range(m):
                    for j in range(p):
                        s = sum(U[i*m+a, c] * V[a*p+j, c] * W[j*n+i, c] for c in range(7))
                        if abs(s - 1) > 1e-9:
                            # Fix W
                            pass
            return U, V, W
        
        # Actually, let me just use a simpler verified Strassen and build up.
        # For safety, use schoolbook for all subproblems but combine them.
        
        # 2x3x7 via block decomposition:
        # C[i,j] = sum_{a=0}^{2} A[i,a]*B[a,j]
        # = sum_{a=0}^{1} A[i,a]*B[a,j] + A[i,2]*B[2,j]
        
        # Part 1: 2x2x7 (A is 2x2, B is 2x7)
        # C1[i,j] = sum_{a=0}^{1} A1[i,a]*B1[a,j]
        # Split 7 = 2+2+2+1
        # C1[i,j] = C11[i,j1] + C12[i,j2] + C13[i,j3] + C14[i,j4]
        # where j1 in {0,1}, j2 in {2,3}, j3 in {4,5}, j4 in {6}
        
        # Each 2x2x2 subproblem: use rank 7 (Strassen)
        # 2x2x1 subproblem: rank 4 (schoolbook)
        
        # Total rank for 2x2x7: 3*7 + 4 = 25
        
        # Part 2: 2x1x7 (A is 2x1, B is 1x7)
        # C2[i,j] = A2[i,0]*B2[0,j], rank = 2*7 = 14 (outer product)
        
        # Total: 25 + 14 = 39
        
        U_list = []
        V_list = []
        W_list = []
        
        # Strassen for 2x2x2
        U_s = np.array([
            [1, 0, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 0, 0],
            [-1, 1, 0, 1, 0, -1, -1],
            [0, 1, 0, -1, 1, 1, 1],
        ], dtype=float)
        V_s = np.array([
            [0, 1, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 1],
            [1, 0, 1, -1, 0, 1, 0],
            [-1, 1, 1, 0, 0, 1, -1],
        ], dtype=float)
        W_s = np.array([
            [-1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, -1],
            [1, -1, -1, -1, 0, 1, 0]
        ], dtype=float)
        
        # Verify Strassen
        n_s, m_s, p_s = 2, 2, 2
        valid_s = True
        for i in range(n_s):
            for a in range(m_s):
                for j in range(p_s):
                    s = sum(U_s[i*m_s+a, c] * V_s[a*p_s+j, c] * W_s[j*n_s+i, c] for c in range(7))
                    if abs(s - 1) > 1e-9:
                        valid_s = False
                        break
        
        if not valid_s:
            # Use schoolbook for 2x2x2
            U_s = np.zeros((4, 8))
            V_s = np.zeros((4, 8))
            W_s = np.zeros((4, 8))
            col = 0
            for i in range(2):
                for a in range(2):
                    for j in range(2):
                        U_s[i*2+a, col] = 1
                        V_s[a*2+j, col] = 1
                        W_s[j*2+i, col] = 1
                        col += 1
        
        # 2x2x1 schoolbook
        U_o = np.zeros((4, 4))
        V_o = np.zeros((2, 4))  # m*p = 2*1 = 2
        W_o = np.zeros((2, 4))  # p*n = 1*2 = 2
        for i in range(2):
            for a in range(2):
                for j in range(1):
                    col = i*2 + a*1 + j
                    U_o[i*2+a, col] = 1
                    V_o[a*1+j, col] = 1
                    W_o[j*2+i, col] = 1
        
        # Build 2x2x7 decomposition
        # Rows of U: 2*2 = 4 (i in 0,1; a in 0,1)
        # Rows of V: 2*7 = 14 (a in 0,1; j in 0..6)
        # Rows of W: 7*2 = 14 (j in 0..6; i in 0,1)
        
        U_227 = np.zeros((4, 25))
        V_227 = np.zeros((14, 25))
        W_227 = np.zeros((14, 25))
        
        col_offset = 0
        # Three 2x2x2 blocks for j pairs (0,1), (2,3), (4,5)
        for block in range(3):
            j_start = block * 2
            # Map local indices to global
            for c in range(7):
                # U: same
                U_227[:, col_offset+c] = U_s[:, c]
                # V: rows a*7+j
                for a in range(2):
                    for j_local in range(2):
                        j_global = j_start + j_local
                        V_227[a*7+j_global, col_offset+c] = V_s[a*2+j_local, c]
                # W: rows j*n+i
                for j_local in range(2):
                    for i in range(2):
                        j_global = j_start + j_local
                        W_227[j_global*2+i, col_offset+c] = W_s[j_local*2+i, c]
            col_offset += 7
        
        # One 2x2x1 block for j=6
        for c in range(4):
            U_227[:, col_offset+c] = U_o[:, c]
            for a in range(2):
                V_227[a*7+6, col_offset+c] = V_o[a*1+0, c]
            for i in range(2):
                W_227[6*2+i, col_offset+c] = W_o[0*2+i, c]
        col_offset += 4
        
        # Build 2x1x7 decomposition (outer product)
        U_217 = np.zeros((2, 14))  # i in 0,1; a=0
        V_217 = np.zeros((7, 14))  # a=0; j in 0..6
        W_217 = np.zeros((14, 14)) # j in 0..6; i in 0,1
        
        col = 0
        for i in range(2):
            for j in range(7):
                U_217[i, col] = 1
                V_217[j, col] = 1
                W_217[j*2+i, col] = 1
                col += 1
        
        # Combine into 2x3x7
        # U: 6x39, V: 21x39, W: 14x39
        U_final = np.zeros((6, 39))
        V_final = np.zeros((21, 39))
        W_final = np.zeros((14, 39))
        
        # Part 1: 2x2x7 -> uses A[i,0:2] and B[0:2, j]
        # U rows 0-3 (i*m+a for i=0,1; a=0,1)
        U_final[0:4, 0:25] = U_227
        # V rows 0-13 (a*p+j for a=0,1; j=0..6)
        V_final[0:14, 0:25] = V_227
        # W rows 0-13 (j*n+i for j=0..6; i=0,1)
        W_final[0:14, 0:25] = W_227
        
        # Part 2: 2x1x7 -> uses A[i,2] and B[2, j]
        # U rows 4-5 (i*m+2 for i=0,1)
        U_final[4:6, 25:39] = U_217
        # V rows 14-20 (2*p+j for j=0..6)
        V_final[14:21, 25:39] = V_217
        # W rows 0-13 (j*n+i) - same as part 1
        W_final[0:14, 25:39] = W_217
        
        # Verify
        if is_valid(U_final, V_final, W_final):
            U, V, W = U_final, V_final, W_final
            current_rank = 39
        else:
            # Keep the previous result
            pass
    
    # Final verification and cleanup
    if not is_valid(U, V, W):
        # Fallback to schoolbook
        U = np.zeros((nm, n*m*p))
        V = np.zeros((mp, n*m*p))
        W = np.zeros((pn, n*m*p))
        col = 0
        for i in range(n):
            for a in range(m):
                for j in range(p):
                    U[i*m+a, col] = 1
                    V[a*p+j, col] = 1
                    W[j*n+i, col] = 1
                    col += 1
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
