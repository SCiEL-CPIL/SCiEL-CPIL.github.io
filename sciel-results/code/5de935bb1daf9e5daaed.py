"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0]=1;U[3][0]=1;V[0][0]=1;V[3][0]=1;W[0][0]=1;W[3][0]=1
        U[2][1]=1;U[3][1]=1;V[0][1]=1;W[2][1]=1;W[3][1]=-1
        U[0][2]=1;V[1][2]=1;V[3][2]=-1;W[1][2]=1;W[3][2]=1
        U[3][3]=1;V[2][3]=1;V[0][3]=-1;W[0][3]=1;W[2][3]=1
        U[0][4]=1;U[1][4]=1;V[3][4]=1;W[0][4]=-1;W[1][4]=1
        U[2][5]=1;U[0][5]=-1;V[0][5]=1;V[1][5]=1;W[3][5]=1
        U[1][6]=1;U[3][6]=-1;V[2][6]=1;V[3][6]=1;W[0][6]=1
        return U, V, W

    def build_2x2x3_rank11():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3."""
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0: U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0: V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0: W[r][c] += W1[r][c]
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0: 1, 1: 2, 2: 4, 3: 5}
        w2_map = {0: 2, 1: 4, 2: 3, 3: 5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0: U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0: V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0: W[w2_map[r]][base+c] += W2[r][c]
        return U, V, W

    def build_2x2x3_rank10():
        """Attempt rank-10 decomposition for 2x2 * 2x3 -> 2x3 using numerical search.
        Returns None if not found within time budget."""
        if time.time() > deadline - 40:
            return None
        
        # We need to find U (4x10), V (6x10), W (6x10) such that
        # sum_t U[i*2+j, t] * V[j*3+k, t] * W[k*2+i, t] = 1 for all i,j,k
        # 
        # Use a structured ansatz with free parameters and solve.
        # Start from the rank-11 solution and try to merge two terms.
        
        # Get the rank-11 solution as starting point
        U11, V11, W11 = build_2x2x3_rank11()
        
        # Try to find a linear combination of two terms that can be replaced
        # by a single term, reducing rank by 1.
        
        # Alternative: use scipy to solve a system
        try:
            from scipy.optimize import fsolve
        except ImportError:
            return None
        
        # Parameterize 10 terms with structured coefficients
        # Use the following ansatz based on known rank-10 constructions:
        # Terms 0-2: (A00+A11) * (B0k+B1k) for k=0,1,2
        # Terms 3-5: (A10+A11) * B0k for k=0,1,2  
        # Terms 6-8: A00 * (B1k-B0k) for k=0,1,2
        # Term 9: (A01-A11) * B1k with appropriate W
        
        # Let me try: Term 9 = (A01) * (B10+B11+B12) with W contributing to C0k
        # C0k = A00*B0k + A01*B1k
        # From terms 0,3,6: C0k gets (A00+A11)(B0k+B1k) + (A10+A11)B0k + A00(B1k-B0k)
        #   = A00*B0k + A00*B1k + A11*B0k + A11*B1k + A10*B0k + A11*B0k + A00*B1k - A00*B0k
        #   = A00*B1k + A11*B0k + A11*B1k + A10*B0k + A11*B0k + A00*B1k
        #   = 2*A00*B1k + A11*B0k + A11*B1k + A10*B0k + A11*B0k
        #   = 2*A00*B1k + 2*A11*B0k + A11*B1k + A10*B0k
        # We want C0k = A00*B0k + A01*B1k
        # So we need: 2*A00*B1k + 2*A11*B0k + A11*B1k + A10*B0k + term9 = A00*B0k + A01*B1k
        # term9 = A00*B0k + A01*B1k - 2*A00*B1k - 2*A11*B0k - A11*B1k - A10*B0k
        # This is not a rank-1 term in general.
        
        # This approach isn't working cleanly. Let me try a direct numerical search.
        # Use a random initialization and gradient descent on the non-convex objective.
        
        if time.time() > deadline - 35:
            return None
        
        # Objective: minimize sum over all (i,j,k) of (sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] - 1)^2
        # with the constraint that entries are 0, 1, or -1 (for integer/half-integer requirement)
        
        # For now, allow real-valued entries and check if we get close enough.
        
        np.random.seed(42)
        best_loss = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        max_iters = 200
        for it in range(max_iters):
            if time.time() > deadline - 35:
                break
            
            # Random initialization with sparse structure
            U = np.zeros((4, 10))
            V = np.zeros((6, 10))
            W = np.zeros((6, 10))
            
            for t in range(10):
                # Each term picks 1-2 entries in U, 1-2 in V, 1-2 in W
                u_rows = np.random.choice(4, size=np.random.randint(1,3), replace=False)
                v_rows = np.random.choice(6, size=np.random.randint(1,3), replace=False)
                w_rows = np.random.choice(6, size=np.random.randint(1,3), replace=False)
                
                for r in u_rows:
                    U[r, t] = np.random.choice([-1, 1])
                for r in v_rows:
                    V[r, t] = np.random.choice([-1, 1])
                for r in w_rows:
                    W[r, t] = np.random.choice([-1, 1])
            
            # Compute loss
            loss = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = np.sum(U[i*2+j, :] * V[j*3+k, :] * W[k*2+i, :])
                        loss += (val - 1.0)**2
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_loss < 1e-9:
                break
        
        if best_loss > 1e-6:
            return None
        
        # Convert to lists
        U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in best_U]
        V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in best_V]
        W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in best_W]
        
        # Final verification
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = np.sum(U_arr[i*2+j, :] * V_arr[j*3+k, :] * W_arr[k*2+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return None
        
        return U_list, V_list, W_list

    def build_4x4x6():
        """Build 4x4x6 decomposition using best available 2x2x3 sub-decomposition."""
        n, m, p = 4, 4, 6
        
        # Try rank-10 first, fall back to rank-11
        sub_decomp = build_2x2x3_rank10()
        if sub_decomp is None:
            sub_decomp = build_2x2x3_rank11()
        
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub

        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        terms = [
            {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
            {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
            {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
            {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
            {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
            {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
        ]

        u_offset = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
        v_offset = {0: (0, 0), 1: (0, 3), 2: (2, 0), 3: (2, 3)}
        w_offset = {0: (0, 0), 1: (3, 0), 2: (0, 2), 3: (3, 2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 3
                    k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff

        return U, V, W

    U, V, W = build_4x4x6()

    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        n2, m2, p2 = 2, 2, 3

        # Get the 2x2x3 sub-decomposition
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])  # 11
        total_rank = 7 * rank_sub  # 77

        U = [[0]*total_rank for _ in range(n*m)]  # 16 rows
        V = [[0]*total_rank for _ in range(m*p)]  # 24 rows
        W = [[0]*total_rank for _ in range(p*n)]  # 24 rows

        # Strassen terms for 4x4 * 4x6:
        # A blocks: A11 (0:2, 0:2), A12 (0:2, 2:4), A21 (2:4, 0:2), A22 (2:4, 2:4)
        # B blocks: B11 (0:2, 0:3), B12 (0:2, 3:6), B21 (2:4, 0:3), B22 (2:4, 3:6)
        # C blocks: C11 (0:2, 0:3), C12 (0:2, 3:6), C21 (2:4, 0:3), C22 (2:4, 3:6)
        #
        # U index: i*m + j = i*4 + j
        # V index: j*p + k = j*6 + k
        # W index: k*n + i = k*4 + i
        #
        # Sub-problem local indices:
        # U_sub: i*m2 + j = i*2 + j, where i in 0..1, j in 0..1
        # V_sub: j*p2 + k = j*3 + k, where j in 0..1, k in 0..2
        # W_sub: k*n2 + i = k*2 + i, where k in 0..2, i in 0..1
        #
        # The 7 Strassen terms:
        terms = [
            # Term 0: M1 = (A11+A22)(B11+B22) -> C11+C22
            {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            # Term 1: M2 = (A21+A22)B11 -> C21-C22
            {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
            # Term 2: M3 = A11(B12-B22) -> C12+C22
            {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
            # Term 3: M4 = A22(B21-B11) -> C11+C21
            {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
            # Term 4: M5 = (A11+A12)B22 -> -C11+C12
            {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
            # Term 5: M6 = (A21-A11)(B11+B12) -> C22
            {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
            # Term 6: M7 = (A12-A22)(B21+B22) -> C11
            {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
        ]

        # Block offset functions:
        # Block number: 0=A11/B11/C11, 1=A12/B12/C12, 2=A21/B21/C21, 3=A22/B22/C22
        # For U (n x m = 4x4):
        # Block 0: rows 0-1, cols 0-1 -> offset (0, 0)
        # Block 1: rows 0-1, cols 2-3 -> offset (0, 2)
        # Block 2: rows 2-3, cols 0-1 -> offset (2, 0)
        # Block 3: rows 2-3, cols 2-3 -> offset (2, 2)
        u_offset = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}

        # For V (m x p = 4x6):
        # Block 0: rows 0-1, cols 0-2 -> offset (0, 0)
        # Block 1: rows 0-1, cols 3-5 -> offset (0, 3)
        # Block 2: rows 2-3, cols 0-2 -> offset (2, 0)
        # Block 3: rows 2-3, cols 3-5 -> offset (2, 3)
        v_offset = {0: (0, 0), 1: (0, 3), 2: (2, 0), 3: (2, 3)}

        # For W (p x n = 6x4):
        # Block 0: rows 0-2, cols 0-1 -> offset (0, 0)
        # Block 1: rows 3-5, cols 0-1 -> offset (3, 0)
        # Block 2: rows 0-2, cols 2-3 -> offset (0, 2)
        # Block 3: rows 3-5, cols 2-3 -> offset (3, 2)
        w_offset = {0: (0, 0), 1: (3, 0), 2: (0, 2), 3: (3, 2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub

            for c in range(rank_sub):
                # U part
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0:
                        continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff

                # V part
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0:
                        continue
                    j_local = v_local // 3
                    k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff

                # W part
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0:
                        continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff

        return U, V, W

    U, V, W = build_4x4x6()

    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
