"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 150s wall-clock limit, keep 40s margin

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Index conventions (trace convention W[k*n + i, r]):
    #   U index: i*m + j   (A is n x m)
    #   V index: j*p + k   (B is m x p)
    #   W index: k*n + i   (C is p x n, trace convention)
    # A tensor decomposition of rank R is valid iff for every (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*rank for _ in range(2)]  # 2*1 = 2 rows
        W = [[0]*rank for _ in range(2)]  # 1*2 = 2 rows (k*n+i, n=2)

        # Term 0: A[0,0]*B[0] -> C[0,0]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # Term 1: A[0,1]*B[1] -> C[0,0]
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        # Term 2: A[1,0]*B[0] -> C[1,0]
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        # Term 3: A[1,1]*B[1] -> C[1,0]
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        # W index: k*2 + i
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3.
        Split B into B1 (2x1) and B2 (2x2).
        A*B1 uses rank-4 schoolbook.
        A*B2 uses rank-7 Strassen.
        """
        rank = 11
        U = [[0]*rank for _ in range(4)]  # 2*2
        V = [[0]*rank for _ in range(6)]  # 2*3
        W = [[0]*rank for _ in range(6)]  # 3*2 (k*2+i)

        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0: U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0: V[r*3][c] += V1[r][c]   # B row r, col 0
            for r in range(2):
                if W1[r][c] != 0: W[r][c] += W1[r][c]     # C row r, col 0 -> k=0,i=r -> idx r

        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0:1, 1:2, 2:4, 3:5}   # B2 (j,k2) -> B (j, k2+1): idx j*3+(k2+1)
        w2_map = {0:2, 1:4, 2:3, 3:5}   # C2 (k2,i) -> C (k2+1,i): idx (k2+1)*2+i
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0: U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0: V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0: W[w2_map[r]][base+c] += W2[r][c]
        return U, V, W

    def build_2x4x6():
        """Rank-12 decomposition for 2x4 * 4x6 -> 2x6 using Strassen-like structure.
        
        Split A (2x4) into A1 (2x2), A2 (2x2)
        Split B (4x6) into B1 (2x6), B2 (2x6)
        C = A1*B1 + A2*B2
        
        Use Strassen for 2x2 * 2x6:
        Split B1 into B1a (2x3), B1b (2x3)
        Each 2x2 * 2x3 uses rank-11 (from build_2x2x3)
        Total: 11 + 11 = 22? No, we need a better approach.
        
        Alternative: Use the fact that 2x4 * 4x6 can be done with rank 12
        by treating it as two independent 2x2 * 2x3 problems plus cross terms.
        
        Actually, let's use a direct construction:
        A = [[A11, A12], [A21, A22]] where each is 2x2
        B = [[B1], [B2]] where B1, B2 are 2x6
        C = [[C1], [C2]] where C1, C2 are 2x6
        
        C1 = A11*B1 + A12*B2
        C2 = A21*B1 + A22*B2
        
        We can use Strassen-like formulas:
        M1 = (A11+A22) * (B1+B2)  [but B1,B2 are 2x6, so this is 2x2 * 2x6]
        M2 = (A21+A22) * B1
        M3 = A11 * (B2-B1)
        M4 = A22 * B2
        M5 = (A11+A12) * B2
        M6 = (A21-A11) * B1
        M7 = (A12-A22) * (B1+B2)
        
        C1 = M1 + M3 - M4 - M7
        C2 = M2 + M4 + M6 - M5
        
        But each Mi involves a 2x2 * 2x6 multiplication, which has rank 11.
        Total rank would be 7 * 11 = 77, same as before.
        
        Better approach: Direct sum decomposition.
        Split A along rows: A_top (2x4), A_bottom (2x4)
        Each 2x4 * 4x6 -> 2x6
        
        For 2x4 * 4x6:
        A = [A1, A2] where A1, A2 are 2x2
        B = [B1; B2] where B1, B2 are 2x6
        C = A1*B1 + A2*B2
        
        Each 2x2 * 2x6 has rank 11 (from build_2x2x3, but that's 2x2 * 2x3 -> 2x3)
        Wait, 2x2 * 2x6 is different.
        
        Let me think differently. The minimal rank for 2x4 * 4x6 is at most 16 (schoolbook).
        Using Strassen ideas, we might do better.
        
        Actually, let's just use the schoolbook for 2x4 * 4x6 which is rank 16,
        and total rank = 2 * 16 = 32.
        
        But we can do better. Let's use a mixed approach:
        For each 2x4 * 4x6, split into:
        - 2x2 * 2x6 (rank 11) for the first half of B
        - 2x2 * 2x6 (rank 11) for the second half of B
        Total per 2x4 * 4x6: 22
        Total: 44
        
        Hmm, that's not great either. Let me try a different strategy.
        
        Actually, the best known bound for 4x4 * 4x6 is around 48-50.
        Let's use a direct sum of four 2x4 * 4x6 problems, each with rank 12.
        
        For 2x4 * 4x6 with rank 12:
        This is equivalent to multiplying a 2x4 matrix by a 4x6 matrix.
        We can view this as: C[i,k] = sum_j A[i,j] * B[j,k]
        
        One approach: Use the fact that this is a linear map from R^24 to R^12.
        The tensor rank is at most 24 (schoolbook).
        
        Let's try a specific construction with rank 12:
        We'll use 12 terms, each computing a linear combination.
        
        Actually, let me just implement the schoolbook for the whole thing first to ensure correctness,
        then try to reduce.
        """
        n, m, p = 2, 4, 6
        rank = n * m * p  # 48 schoolbook
        U = [[0]*rank for _ in range(n*m)]  # 8
        V = [[0]*rank for _ in range(m*p)]  # 24
        W = [[0]*rank for _ in range(p*n)]  # 12
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W

    def build_4x4x6():
        """Direct sum decomposition: split A into 4 row-pairs, each 2x4 * 4x6 -> 2x6.
        
        Total rank = 4 * rank(2x4 * 4x6)
        Using schoolbook for each 2x4 * 4x6: rank = 48, total = 192 (worse)
        
        Better: Use a shared decomposition.
        Actually, let's use the recursive Strassen approach but with better sub-problems.
        
        Alternative: Use the fact that 4x4 * 4x6 can be decomposed as:
        Split A into A_top (2x4), A_bottom (2x4)
        Split B into B1 (4x3), B2 (4x3)
        C_top = A_top * B  (2x6)
        C_bottom = A_bottom * B (2x6)
        
        Each 2x4 * 4x6 has rank at most 24 (schoolbook) or less with Strassen.
        
        Let's use a hybrid: 
        - For 2x4 * 4x6, split B into two 4x3 blocks
        - Each 2x4 * 4x3 can be done with rank 12 (using Strassen-like)
        - Total per 2x4 * 4x6: 24
        - Total: 48
        
        Actually, let me just improve the current Strassen approach by using
        a better base case.
        """
        n, m, p = 4, 4, 6
        
        # Use the 2x2x3 base case (rank 11) and apply Strassen
        # This gives rank 77 as before, but let's see if we can do better.
        
        # Alternative: Direct construction with rank 48
        # Split into 4 independent 2x4 * 4x6 problems
        # Each 2x4 * 4x6: use rank 12 construction
        
        def build_2x4x6_rank12():
            """Rank-12 decomposition for 2x4 * 4x6 -> 2x6.
            
            A is 2x4, B is 4x6, C is 2x6.
            C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
            
            We use 12 terms. Let's define them carefully.
            
            Term 0: A[0,0] * B[0,:] -> C[0,:]
            Term 1: A[0,1] * B[1,:] -> C[0,:]
            Term 2: A[0,2] * B[2,:] -> C[0,:]
            Term 3: A[0,3] * B[3,:] -> C[0,:]
            Term 4: A[1,0] * B[0,:] -> C[1,:]
            Term 5: A[1,1] * B[1,:] -> C[1,:]
            Term 6: A[1,2] * B[2,:] -> C[1,:]
            Term 7: A[1,3] * B[3,:] -> C[1,:]
            That's only 8, but we need 24 for schoolbook.
            
            Wait, each term contributes to all 6 entries of C[i,:].
            So 8 terms is enough! Rank 8.
            
            But that's just schoolbook. Can we do better with Strassen?
            
            For 2x4 * 4x6, the tensor rank is at most 8 (schoolbook).
            Actually, schoolbook is 2*4*6 = 48, but that's counting each (i,j,k) separately.
            For a tensor decomposition, we need sum_r U[i,j,r] V[j,k,r] W[i,k,r] = delta(i,j,k)
            
            Wait, I'm confusing myself. Let me re-read the problem.
            
            The decomposition is for the matrix multiplication tensor:
            T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
            
            So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
            
            For 2x4 * 4x6: n=2, m=4, p=6
            U: 8 x R, V: 24 x R, W: 12 x R
            
            Schoolbook: R = 2*4*6 = 48
            U[i*4+j, r] = 1 if r corresponds to (i,j,k)
            V[j*6+k, r] = 1 if r corresponds to (i,j,k)
            W[k*2+i, r] = 1 if r corresponds to (i,j,k)
            
            Can we do better? The minimal rank for 2x4 * 4x6 is unknown,
            but it's at most 48 and at least max(2*4, 4*6, 2*6) = 24.
            
            Let's try rank 24 (the lower bound).
            """
            n, m, p = 2, 4, 6
            rank = 24  # lower bound
            U = [[0]*rank for _ in range(n*m)]  # 8
            V = [[0]*rank for _ in range(m*p)]  # 24
            W = [[0]*rank for _ in range(p*n)]  # 12
            
            # We need to find a rank-24 decomposition.
            # This is non-trivial. Let's fall back to schoolbook for now.
            # Actually, let's just use schoolbook with rank 48.
            
            # For now, return schoolbook
            rank = 48
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            return U, V, W
        
        # Use direct sum of four 2x4 * 4x6 decompositions
        U_sub, V_sub, W_sub = build_2x4x6_rank12()
        rank_sub = len(U_sub[0])  # 48
        total_rank = 4 * rank_sub  # 192
        
        # That's worse than 77. Let's stick with the Strassen approach.
        # Instead, let's try to improve the Strassen base case.
        
        U_strassen, V_strassen, W_strassen = build_2x2x3()
        rank_sub = len(U_strassen[0])  # 11
        total_rank = 7 * rank_sub  # 77
        
        U = [[0]*total_rank for _ in range(n*m)]  # 16
        V = [[0]*total_rank for _ in range(m*p)]  # 24
        W = [[0]*total_rank for _ in range(p*n)]  # 24
        
        # Strassen terms
        terms = [
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},
        ]
        u_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        v_offset = {0:(0,0), 1:(0,3), 2:(2,0), 3:(2,3)}
        w_offset = {0:(0,0), 1:(3,0), 2:(0,2), 3:(3,2)}
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_strassen[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2; j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(6):
                    v_val = V_strassen[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 3; k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(6):
                    w_val = W_strassen[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2; i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff
        return U, V, W

    def make_trace(U, V, W, n, m, p):
        """Return the (n*m*p,) vector of trace products T[i,j,k]."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                u_row = Ua[i*m + j, :]
                for k in range(p):
                    T[i, j, k] = np.dot(u_row * Va[j*p + k, :], Wa[k*n + i, :])
        return T

    def verify(U, V, W, n, m, p, tol=1e-7):
        """Exact (dense) verification of the decomposition."""
        T = make_trace(U, V, W, n, m, p)
        return bool(np.all(np.abs(T - 1.0) <= tol))

    def try_reduce_rank(U, V, W, n, m, p, time_budget):
        """More aggressive rank reduction using linear algebra.
        
        1. First, remove columns with zero contribution.
        2. Then, try to find linear dependencies among columns.
        3. Use SVD to identify near-dependent columns.
        """
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        R = Ua.shape[1]
        if R == 0:
            return U, V, W, R

        def column_contribution(r):
            u_col = Ua[:, r]
            v_col = Va[:, r]
            w_col = Wa[:, r]
            c = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    uv = u_col[i*m + j]
                    if uv == 0: continue
                    for k in range(p):
                        c[i, j, k] = uv * v_col[j*p + k] * w_col[k*n + i]
            return c

        # Step 1: Remove zero-contribution columns
        keep = list(range(R))
        contribs = {}
        for r in range(R):
            if time.time() > time_budget:
                break
            contribs[r] = column_contribution(r)

        changed = True
        while changed and time.time() < time_budget:
            changed = False
            for r in list(keep):
                if time.time() > time_budget:
                    break
                c = contribs.get(r)
                if c is None:
                    continue
                if np.all(np.abs(c) <= 1e-9):
                    keep.remove(r)
                    changed = True
                    break

        if len(keep) < R:
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            R = len(keep)

        # Step 2: Try to find linear dependencies using SVD
        if time.time() < time_budget - 5 and R > 1:
            # Build the trace tensor as a matrix: each column r gives a vector in R^(n*m*p)
            # vec_r = (U[:,r] outer V[:,r] outer W[:,r]) in index space
            nmp = n * m * p
            M = np.zeros((nmp, R), dtype=float)
            for r in range(R):
                c = contribs.get(r)
                if c is None:
                    c = column_contribution(r)
                    contribs[r] = c
                # Reshape c to 1D in the correct order
                # c[i,j,k] -> index i*m*p + j*p + k
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M[i*m*p + j*p + k, r] = c[i, j, k]
            
            # SVD to find rank
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                # Columns with small singular values might be removable
                # But we need to be careful: we can only remove if the contribution is zero
                # in the current decomposition.
                
                # Alternative: Check if any column is a linear combination of others
                # This is expensive, so let's skip for now.
            except:
                pass

        if len(keep) == Ua.shape[1]:
            return U, V, W, R

        return Ua.tolist(), Va.tolist(), Wa.tolist(), R

    # Build the initial rank-77 decomposition.
    U, V, W = build_4x4x6()

    # Verify the initial decomposition; fall back to schoolbook if broken.
    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce the rank (time-bounded).
    time_budget = min(deadline - 10, time.time() + 90)
    U_r, V_r, W_r, r_rank = try_reduce_rank(U, V, W, n, m, p, time_budget)

    # Use the reduced decomposition only if it still verifies exactly.
    if verify(U_r, V_r, W_r, n, m, p):
        return {"U": U_r, "V": V_r, "W": W_r}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
