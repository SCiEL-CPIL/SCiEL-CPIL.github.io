"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def make_strassen_4x4():
        """
        Strassen-like decomposition for 4x4x4 matrix multiplication.
        Uses the known rank-19 decomposition for 4x4x4 (from Winograd/Strassen extensions).
        Actually, the best known for 4x4x4 is rank 19 (or we use a simple block approach).
        
        Let's use a simpler approach: split 4x4 into four 2x2 blocks.
        Each 2x2x2 uses Strassen (rank 7), total = 4*7 = 28.
        But we can do better with known 4x4x4 decompositions.
        
        For now, use 4x(2x2x2 Strassen) = rank 28 for the 4x4x4 part.
        """
        # Strassen 2x2x2: 7 multiplications
        # For 4x4x4, split into 4 quadrants of 2x2
        # Each quadrant needs a 2x2x2 decomposition
        # Total rank = 4 * 7 = 28
        
        # Actually, let me use the direct schoolbook for 4x4x4 = rank 64
        # But that's worse. Let me think...
        
        # Better: use the known fact that 4x4x4 can be done in rank 19
        # But implementing that is complex. Let's use a block approach:
        # Split 4x4 into 2x2 blocks, use Strassen for each
        # This gives rank 4*7 = 28
        
        # Strassen 2x2x2 coefficients:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b21-b22)
        # m4 = a22(b12-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # For 4x4x4, we have 4 independent 2x2x2 problems
        # Each has 7 terms, total 28 terms
        
        rank = 28
        U = [[0] * rank for _ in range(16)]  # 4*4 = 16
        V = [[0] * rank for _ in range(16)]  # 4*4 = 16
        W = [[0] * rank for _ in range(16)]  # 4*4 = 16
        
        # Strassen coefficients for 2x2x2
        # For each 2x2 block, the U, V, W entries
        # Block (bi, bj) where bi, bj in {0,1}
        # Within block, indices are 2x2
        
        strassen_coeffs = [
            # (u_entries, v_entries, w_entries) for each of 7 multiplications
            # u: (i1, i2) pairs, v: (j1, j2) pairs, w: (k1, k2) pairs
            # m1 = (a11+a22)(b11+b22)
            {
                'u': [(0,1),(1,0)],  # a11, a22
                'v': [(0,1),(0,1)],  # b11, b22 -> actually b11 and b22
                'w': [(0,0),(0,0)]   # contributes to c11
            },
            # m2 = (a21+a22)b11
            {
                'u': [(1,1),(1,0)],  # a21, a22
                'v': [(0,1),(0,1)],  # b11
                'w': [(1,1),(1,1)]   # contributes to c21
            },
            # m3 = a11(b21-b22)
            {
                'u': [(0,1)],  # a11
                'v': [(1,1),(1,0)],  # b21, -b22
                'w': [(0,1),(1,1)]   # contributes to c12, c22
            },
            # m4 = a22(b12-b11)
            {
                'u': [(1,0)],  # a22
                'v': [(0,1),(0,0)],  # b12, -b11
                'w': [(0,0),(1,1)]   # contributes to c11, c22
            },
            # m5 = (a11+a12)b22
            {
                'u': [(0,1),(0,0)],  # a11, a12
                'v': [(1,0),(1,0)],  # b22
                'w': [(0,0),(0,1)]   # contributes to c11, c12
            },
            # m6 = (a21-a11)(b11+b12)
            {
                'u': [(1,1),(0,1)],  # a21, -a11
                'v': [(0,1),(0,1)],  # b11, b12
                'w': [(1,1)]         # contributes to c22
            },
            # m7 = (a12-a22)(b21+b22)
            {
                'u': [(0,0),(1,0)],  # a12, -a22
                'v': [(1,1),(1,1)],  # b21, b22
                'w': [(0,0)]         # contributes to c11
            },
        ]
        
        # This is getting complex. Let me use a simpler approach.
        # Just use schoolbook for 4x4x4 = rank 64, and schoolbook for 4x4x3 = rank 48
        # Total = 112 (same as full schoolbook)
        # 
        # Better: use the insight from inspiration 1.
        # Use Strassen for the 4x4x4 part (rank ~49 based on inspiration)
        # and schoolbook for 4x4x3 (rank 48)
        # Total = 97
        
        # Let me construct this properly.
        pass

    def make_block_decomp():
        """
        Block-structured decomposition:
        - Split 4x4x7 into 4x4x4 and 4x4x3 parts
        - Use Strassen-based rank-49 for 4x4x4 (from known results)
        - Use schoolbook rank-48 for 4x4x3
        - Total rank = 97
        
        Actually, let me use a cleaner approach with explicit Strassen for 2x2 blocks.
        """
        # For 4x4x4: split into 4 independent 2x2x2 problems
        # Each 2x2x2 uses Strassen (rank 7)
        # Total for 4x4x4: rank 28
        
        # For 4x4x3: schoolbook, rank = 4*4*3 = 48
        # Total: 28 + 48 = 76
        
        # But wait, the 4x4x4 part has 4 quadrants, each 2x2x2.
        # The 4x4x3 part also has 4 quadrants (in the last dimension sense).
        
        # Let me be more careful.
        # A is 4x4, B is 4x7, C = A*B is 4x7.
        # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
        # Split B into 2x2 blocks (with last dim 7, so split into 4 and 3):
        #   B = [[B11, B12], [B21, B22]] where B11 is 2x4, B12 is 2x3, etc.
        # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
        #             [A21*B11 + A22*B21, A21*B12 + A22*B22]]
        
        # Each 2x2x4 and 2x2x3 multiplication.
        # 2x2x4: can be done as 2x2x2 + 2x2x2, rank 14
        # 2x2x3: can be done as 2x2x2 + 2x2x1, rank 7+4=11
        
        # There are 4 such 2x2x4 terms and 4 such 2x2x3 terms.
        # Total rank = 4*14 + 4*11 = 56 + 44 = 100
        
        # Hmm, that's worse than 97. Let me try another split.
        
        # Alternative: split along the middle dimension (m=4).
        # A is 4x4, B is 4x7.
        # Split A's columns and B's rows into 2+2.
        # C = A1*B1 + A2*B2 where A1 is 4x2, B1 is 2x7, etc.
        # Each 4x2x7 multiplication.
        # 4x2x7: split further...
        
        # This is getting complicated. Let me just use the schoolbook (rank 112)
        # and then try to refine it, or use the known rank-97 from inspiration.
        
        # Based on inspiration 1, rank 97 is achievable.
        # Let me construct it using the block approach described there.
        
        # For now, return schoolbook and we'll refine.
        return make_schoolbook()

    def make_rank97():
        """
        Construct the rank-97 decomposition based on the approach from inspiration 1:
        - Use Strassen's algorithm (rank 49) for the first 4x4 block
        - Use schoolbook (rank 48) for the remaining 4x3 part
        """
        # This is a placeholder - the actual construction is complex.
        # Let me use a different strategy: start with schoolbook and try to
        # find a lower-rank decomposition via structured optimization.
        
        # For now, return schoolbook.
        return make_schoolbook()

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    def refine(U, V, W, r, max_iter=200):
        """Vectorized gradient descent refinement. Returns refined arrays."""
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            u_vals = Ua[u_idx, :]
            v_vals = Va[v_idx, :]
            w_vals = Wa[w_idx, :]
            
            prod = u_vals * v_vals * w_vals
            result = prod.sum(axis=1)
            error = result - 1.0
            err_norm = np.linalg.norm(error)
            
            if err_norm < 1e-10:
                break
            
            # Reshape to 4D: (n, m, p, r)
            err_4d = error.reshape(n, m, p, 1)
            u_4d = u_vals.reshape(n, m, p, r)
            v_4d = v_vals.reshape(n, m, p, r)
            w_4d = w_vals.reshape(n, m, p, r)
            
            # Vectorized gradient computations
            # grad_U[i*m+j, c] = sum_k 2*err[i,j,k]*V[j*p+k,c]*W[k*n+i,c]
            grad_U_4d = 2 * err_4d * v_4d * w_4d
            grad_U = np.sum(grad_U_4d, axis=2).reshape(n*m, r)
            
            # grad_V[j*p+k, c] = sum_i 2*err[i,j,k]*U[i*m+j,c]*W[k*n+i,c]
            grad_V_4d = 2 * err_4d * u_4d * w_4d
            # Sum over i axis (axis=0) to get (m, p, r), then reshape to (m*p, r)
            grad_V = np.sum(grad_V_4d, axis=0).reshape(m*p, r)
            
            # grad_W[k*n+i, c] = sum_j 2*err[i,j,k]*U[i*m+j,c]*V[j*p+k,c]
            grad_W_4d = 2 * err_4d * u_4d * v_4d
            # Sum over j axis (axis=1) to get (n, p, r), then transpose to (p, n, r) and reshape
            grad_W = np.sum(grad_W_4d, axis=1).transpose(1, 0, 2).reshape(p*n, r)
            
            # Adaptive learning rate with backtracking
            lr = 0.1 / (1.0 + it * 0.1)
            
            Ua -= lr * grad_U
            Va -= lr * grad_V
            Wa -= lr * grad_W
        
        return Ua, Va, Wa

    def try_rank(r, n_trials=10, max_iter=100):
        """Try to find a rank-r decomposition with random restarts."""
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(trial * 104729 + r * 31 + 7)
            
            # Random initialization with various scales
            scale = [0.5, 1.0, 2.0, 0.1, 5.0][trial % 5]
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) * scale
            
            U_ref, V_ref, W_ref = refine(U, V, W, r, max_iter=max_iter)
            
            err = verify(U_ref, V_ref, W_ref, r)
            if err < 1e-8:
                return {
                    "U": U_ref.tolist(),
                    "V": V_ref.tolist(),
                    "W": W_ref.tolist()
                }
        
        return None

    # Search for lower rank decompositions
    # Lower bound for 4x4x7: at least max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # Realistic lower bound is higher; try from ~50 upward
    search_ranks = list(range(50, min(best_rank, 112)))
    
    for r in search_ranks:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, n_trials=8, max_iter=80)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
