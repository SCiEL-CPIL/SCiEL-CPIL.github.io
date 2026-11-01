import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin before 150s limit

    # Target: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # With n=2, m=2, p=9: U is (4, R), V is (18, R), W is (18, R)

    size_u = n * m  # 4
    size_v = m * p  # 18
    size_w = p * n  # 18
    n_tensors = n * m * p  # 36

    def build_result(U, V, W):
        """Build and validate the result. Returns dict or None if invalid."""
        R = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = max(err, abs(val - 1.0))
        if err > 1e-8:
            return None
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_schoolbook(rank_override=None):
        """Standard schoolbook decomposition with rank n*m*p = 36."""
        R = rank_override if rank_override is not None else n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def strassen_block_U(i, j, col, U):
        """Set U entries for Strassen block."""
        if i == 0 and j == 0:
            U[0, col] = 1
        elif i == 0 and j == 1:
            U[0, col] = 1
            U[1, col] = -1
        elif i == 1 and j == 0:
            U[2, col] = 1
        elif i == 1 and j == 1:
            U[2, col] = 1
            U[3, col] = 1

    def strassen_block_V(j, k, col, V):
        """Set V entries for Strassen block. j in {0,1}, k in {0,1}."""
        if j == 0 and k == 0:
            V[0, col] = 1
        elif j == 0 and k == 1:
            V[0, col] = 1
            V[9, col] = 1
        elif j == 1 and k == 0:
            V[9, col] = 1
        elif j == 1 and k == 1:
            V[9, col] = 1
            V[10, col] = -1

    def strassen_block_W(k, i, col, W):
        """Set W entries for Strassen block. k in {0,1}, i in {0,1}."""
        if k == 0 and i == 0:
            W[0, col] = 1
        elif k == 0 and i == 1:
            W[0, col] = 1
            W[1, col] = 1
        elif k == 1 and i == 0:
            W[2, col] = 1
        elif k == 1 and i == 1:
            W[2, col] = 1
            W[3, col] = -1

    # Build a hybrid: Strassen for k in {0,1}, schoolbook for k in {2..8}
    # This gives rank = 7 + 2*2*7 = 7 + 28 = 35
    def build_hybrid_strassen():
        R = 7 + 2 * 2 * 7  # 35
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen part for k in {0, 1}
        # 7 Strassen terms
        # P1 = A00 * B00  -> C00 += P1, C10 += P1, C01 += P1, C11 += P1
        # P2 = A10 * B00  -> C10 += P2
        # P3 = A01 * (B10 - B00)  -> C01 += P3
        # P4 = A11 * (B01 - B11)  -> C11 += P4
        # P5 = (A00 + A11) * (B00 + B11)  -> C00 += P5, C11 += P5, C01 -= P5, C11 -= P5
        # P6 = (A10 + A00) * (B01 - B00)  -> C11 += P6, C01 -= P6
        # P7 = (A01 - A11) * (B10 + B11)  -> C10 += P7, C00 -= P7
        
        # Let's define the 7 terms with their (i,j) for A, (j,k) for B, (k,i) for C
        # Actually, for 2x2*2x2, the standard Strassen decomposition:
        # M1 = (A00 + A11)(B00 + B11)  -> C00 += M1, C11 += M1, C01 -= M1, C11 -= M1
        # M2 = (A10 + A11)B00  -> C10 += M2, C11 += M2
        # M3 = A00(B10 - B00)  -> C00 += M3, C10 += M3
        # M4 = A11(B01 - B11)  -> C01 += M4, C11 += M4
        # M5 = (A00 + A01)B11  -> C01 += M5, C11 += M5
        # M6 = (A10 - A00)(B00 + B01)  -> C00 -= M6, C01 -= M6
        # M7 = (A01 - A11)(B10 + B11)  -> C10 += M7, C11 += M7
        
        # This is getting complicated. Let me just use the standard 7-term decomposition
        # where each term is a product of a linear combination of A elements, 
        # a linear combination of B elements, and a linear combination of C elements.
        
        # For multiplying A (2x2) by B (2x2) to get C (2x2):
        # C[i,k] = sum_j A[i,j] B[j,k]
        # T[i,j,k] = 1
        
        # The 7 terms for Strassen (one common formulation):
        # r=0: A00*B00 -> C00 += 1, C10 += 1 (from A10*B00... no)
        
        # Let me just hard-code the standard 7-term Strassen for 2x2*2x2.
        # Using the formulation where:
        # S1 = A00 + A11, S2 = A10 + A11, S3 = A00, S4 = A11
        # S5 = A00 + A01, S6 = A10 - A00, S7 = A01 - A11
        # T1 = B00 + B11, T2 = B00, T3 = B10 - B00, T4 = B01 - B11
        # T5 = B11, T6 = B00 + B01, T7 = B10 + B11
        # 
        # M1 = S1*T1 = (A00+A11)(B00+B11)
        #   C00 += M1, C11 += M1, C01 -= M1, C11 -= M1  -> C00+=M1, C01-=M1, C11+=M1-M1=0? 
        #   That doesn't work. Let me use a different standard set.
        
        # Standard Strassen (from Wikipedia):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M3 - M5 - M7
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M4 + M7
        
        # So the 7 terms and their contributions to (i,k):
        # Term 0 (M1): A: A00+A11, B: B00+B11, C: C00+=1, C01+=0, C10+=0, C11+=1... 
        # Wait, C00 = M1 + M3 - M5 - M7, so M1 contributes +1 to C00
        # C01 = M4 + M5, so M1 contributes 0 to C01
        # C10 = M2 + M3, so M1 contributes 0 to C10
        # C11 = M1 - M2 + M4 + M7, so M1 contributes +1 to C11
        
        # Let me define each term as (U_coeffs, V_coeffs, W_coeffs)
        # U is indexed by (i,j) -> i*2+j
        # V is indexed by (j,k) -> j*2+k (for the 2x2 block, but in our case V is (18, R))
        # W is indexed by (k,i) -> k*2+i
        
        # For the Strassen block (k in {0,1}):
        # V index: j*9+k, W index: k*2+i
        
        # Term 0 (M1): (A00+A11)(B00+B11) -> C00+=1, C11+=1
        # U: [1,0,0,1], V: B00+B11 -> V[0,0]=1, V[1,1]=1 (j=0,k=0 and j=1,k=1)
        #   In our V indexing: V[0*9+0]=V[0]=1, V[1*9+1]=V[10]=1
        # W: C00=1, C11=1 -> W[0*2+0]=W[0]=1, W[1*2+1]=W[3]=1
        # But wait, C00 means i=0,k=0, so W index = k*2+i = 0*2+0 = 0
        # C11 means i=1,k=1, so W index = 1*2+1 = 3
        
        # Let me just enumerate all 7 terms:
        strassen_terms = [
            # (U_coeffs for 4 entries [A00,A01,A10,A11], 
            #  V_coeffs for B entries (8 entries for 2x2: B00,B01,B10,B11 -> indices 0,1,9,10),
            #  W_coeffs for C entries (4 entries: C00,C01,C10,C11 -> indices 0,1,2,3))
            # M1 = (A00+A11)(B00+B11): C00+=1, C11+=1
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # C00=idx0, C01=idx1, C10=idx2, C11=idx3
            # M2 = (A10+A11)B00: C10+=1
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            # M3 = A00(B10-B00): C00+=1, C10+=1
            ([1,0,0,0], [0,0,1,-1], [1,0,1,0]),
            # M4 = A11(B01-B11): C01+=1, C11+=1
            ([0,0,0,1], [0,1,0,-1], [0,1,0,1]),
            # M5 = (A00+A01)B11: C01+=1, C11+=1... 
            # Wait, C01 = M4 + M5, C11 = M1 - M2 + M4 + M7
            # So M5 contributes to C01 only? Let me recheck.
            # C00 = M1 + M3 - M5 - M7
            # C01 = M4 + M5
            # C10 = M2 + M3
            # C11 = M1 - M2 + M4 + M7
            # So M5: C01+=1, C00-=1
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            # M6 = (A10-A00)(B00+B01): C00-=1, C01-=1
            # Wait, M6 is not in the standard 7. Let me re-examine.
            # The standard 7 are M1-M7 as above. M6 = (A10-A00)(B00+B01)
            # C00 = M1 + M3 - M5 - M7... where does M6 appear?
            # I think I have the wrong set. Let me use a different source.
        ]
        
        # Actually, let me just use a simpler approach. The rank of 2x2x2 is 7.
        # I'll use the known 7-term decomposition and verify it.
        
        # Known decomposition for 2x2*2x2 (rank 7):
        # Using the formulation where each term has at most 2 non-zero entries in each factor.
        
        # Let me just construct it carefully.
        # T[i,j,k] = 1 for i,j,k in {0,1}
        # We need 7 terms such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        
        # This is a well-known problem. The 7 terms can be:
        terms_2x2x2 = [
            # r, U entries (i,j), V entries (j,k), W entries (k,i)
            # Format: (list of (idx, val) for U, list of (idx, val) for V, list of (idx, val) for W)
            # idx for U: i*2+j, idx for V: j*2+k, idx for W: k*2+i
            # Term 0: U[0]=1, V[0]=1, W[0]=1, W[2]=1  (A00*B00 -> C00 and C10)
            #   T[0,0,0] += 1*1*1 = 1 ✓, T[1,0,0] += 0*1*1 = 0, T[0,0,1] += 1*1*0 = 0, T[1,0,1] += 0*1*0 = 0
            #   This only covers C00 for (i=0,k=0). Not enough.
        ]
        
        # This is getting too complicated. Let me use a different approach.
        # I'll use the fact that we can always do schoolbook for rank 36.
        # For the hybrid, let me just use Strassen for the first 2 columns of B.
        # The key insight is that for a 2x2 * 2x2 subproblem, we can use 7 multiplications.
        # Then for the remaining 7 columns, we use schoolbook (4 multiplications per column).
        # Total: 7 + 4*7 = 35.
        
        # Let me just hard-code the Strassen 7-term decomposition for 2x2*2x2.
        # I'll verify it numerically.
        
        # Standard Strassen decomposition (one common set):
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B00)
        # P4 = A11(B01-B11)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        #
        # C00 = P1 + P3 - P5 - P7
        # C01 = P4 + P5
        # C10 = P2 + P3
        # C11 = P1 - P2 + P4 + P7
        
        # Let me define each P as a tensor product of three vectors.
        # P1: (A00+A11) * (B00+B11) * (C00 + C11 - C01)
        #   U: [1,0,0,1] (A00=1, A01=0, A10=0, A11=1)
        #   V: [1,0,0,1] (B00=1, B01=0, B10=0, B11=1)
        #   W: [1,-1,0,1] (C00=1, C01=-1, C10=0, C11=1)
        # P2: (A10+A11) * B00 * (C10 + C11)
        #   U: [0,0,1,1]
        #   V: [1,0,0,0]
        #   W: [0,0,1,1]
        # P3: A00 * (B10-B00) * (C00 + C10)
        #   U: [1,0,0,0]
        #   V: [0,0,1,-1]  (B10=1, B00=-1)
        #   W: [1,0,1,0]
        # P4: A11 * (B01-B11) * (C01 + C11)
        #   U: [0,0,0,1]
        #   V: [0,1,0,-1]  (B01=1, B11=-1)
        #   W: [0,1,0,1]
        # P5: (A00+A01) * B11 * (-C00 + C01)
        #   U: [1,1,0,0]
        #   V: [0,0,0,1]
        #   W: [-1,1,0,0]
        # P6: (A10-A00) * (B00+B01) * ?
        #   Wait, P6 doesn't appear in the C formula. Let me re-examine.
        #   C00 = P1 + P3 - P5 - P7
        #   So P7 must be: (A01-A11)*(B10+B11)*(-C00 + C10)
        #   U: [0,1,0,-1]
        #   V: [0,0,1,1]
        #   W: [-1,0,1,0]
        
        # Let me verify: 
        # C00 = P1 + P3 - P5 - P7
        #   = (A00+A11)(B00+B11) + A00(B10-B00) - (A00+A01)B11 - (A01-A11)(B10+B11)
        #   = A00B00 + A00B11 + A11B00 + A11B11 + A00B10 - A00B00 - A00B11 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
        #   Wait, that's not right. Let me expand more carefully.
        
        # This is getting too complex. Let me just use a numerical approach to find a rank-7 decomposition
        # for 2x2x2, or just use schoolbook for everything.
        
        # Given the time constraints, let me just use the schoolbook (rank 36) as a reliable fallback,
        # and try to find a better decomposition using optimization.
        
        return build_schoolbook()

    # Try a hybrid approach: use known low-rank decompositions for blocks
    # For 2x2x2, the minimal rank is 7 (Strassen).
    # For the full 2x2x9, we can potentially use a Kronecker product or block approach.
    
    # Let's try: split B into two parts: B[:, 0:2] and B[:, 2:9]
    # For B[:, 0:2] (2x2 part), use Strassen (rank 7)
    # For B[:, 2:9] (2x7 part), use schoolbook (rank 4*7 = 28)
    # Total rank: 7 + 28 = 35
    
    def build_hybrid_rank35():
        R = 35
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen part for k in {0, 1}
        # We need 7 terms for the 2x2*2x2 subproblem
        # Let me use the following 7-term decomposition for 2x2x2:
        # I'll define them as (U_coeffs, V_coeffs, W_coeffs) for the 2x2 block
        
        # After careful thought, let me use a simpler but correct approach:
        # Just use schoolbook for all 36 terms. Rank 36.
        # The hybrid approach requires getting the Strassen decomposition right,
        # which is error-prone. Let me try optimization instead.
        
        return build_schoolbook()

    # Let's try to use optimization to find a rank < 36 decomposition
    from scipy.optimize import least_squares
    
    def residuals(params, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U_flat = params[:R * size_u].reshape(R, size_u)
        V_flat = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v)
        W_flat = params[R * (size_u + size_v) :].reshape(R, size_w)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    val = np.sum(U_flat[:, ui] * V_flat[:, vj] * W_flat[:, wk])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    best_result = build_schoolbook()
    best_rank = n * m * p

    # Try to find lower rank using optimization
    # Start from rank 20 and go up
    for R in range(20, 36):
        if time.time() > deadline - 10:
            break
        
        n_params = R * (size_u + size_v + size_w)
        np.random.seed(42 + R)
        x0 = np.random.randn(n_params) * 0.1
        
        try:
            result = least_squares(
                residuals, 
                x0, 
                args=(R,), 
                method='lm',
                max_nfev=500,
                x_scale='jac-1'
            )
            
            res = residuals(result.x, R)
            if np.max(np.abs(res)) < 1e-8:
                U_flat = result.x[:R * size_u].reshape(R, size_u)
                V_flat = result.x[R * size_u : R * (size_u + size_v)].reshape(R, size_v)
                W_flat = result.x[R * (size_u + size_v) :].reshape(R, size_w)
                
                U = U_flat.T
                V = V_flat.T
                W = W_flat.T
                
                res_check = build_result(U, V, W)
                if res_check is not None:
                    best_result = res_check
                    best_rank = R
                    break
        except Exception:
            pass

    return best_result
