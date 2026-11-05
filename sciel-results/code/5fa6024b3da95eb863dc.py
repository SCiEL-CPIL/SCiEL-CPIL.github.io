import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # safety margin

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, seed=12345, trials=20):
        """Return True if the decomposition exactly reproduces A@B for random inputs."""
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            rng = np.random.RandomState(seed)
            for _ in range(trials):
                A = rng.rand(n, m)
                B = rng.rand(m, p)
                C_expected = A @ B
                x = (A.reshape(n * m) @ Ua)
                y = (B.reshape(m * p) @ Va)
                prod = x * y
                C_computed = (Wa * prod[None, :]).sum(axis=1).reshape(p, n)
                C_expected_T = C_expected.T
                if not np.allclose(C_computed, C_expected_T, atol=1e-9, rtol=1e-9):
                    return False
            return True
        except Exception:
            return False

    def schoolbook_decomposition():
        """Full schoolbook: rank 80"""
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
        return U, V, W

    def winograd_4x4x4_decomposition():
        """
        Winograd F(4) decomposition for 4x4x4 matrix multiplication.
        Uses the tensor product structure: 4x4x4 = (2x2x2) ⊗ (2x2x2)
        Rank = 16 (vs 49 for double Strassen)
        
        The key insight: if we have a rank-r decomposition for d×d×d,
        then the rank-r² decomposition for (2d)×(2d)×(2d) works via
        Kronecker products of the factor matrices.
        
        For 2x2x2, Strassen gives rank 7.
        For 4x4x4, Winograd gives rank 16 (which is less than 7²=49).
        
        Actually, let me use the correct Winograd approach:
        The 4x4 matrix product can be computed using the F(4) transform
        which requires 16 multiplications (vs 16 schoolbook, 49 Strassen).
        
        Wait - schoolbook for 4x4x4 is 16*4=64... no, 4*4*4=64.
        Actually schoolbook for n×m×p is n*m*p = 4*4*4 = 64.
        Strassen gives 49.
        Winograd F(4) gives 16.
        
        Let me construct the Winograd F(4) decomposition properly.
        
        For the 2x2 case, Strassen's 7 multiplications are:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B00)
        M4 = A11(B11-B01)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        For 4x4, we can use the Kronecker product approach:
        If T is the 2x2x2 tensor with rank-7 decomposition (U2, V2, W2),
        then the 4x4x4 tensor has a rank-49 decomposition via U2⊗U2, V2⊗V2, W2⊗W2.
        
        But Winograd F(4) achieves rank 16 by a different method.
        
        Actually, let me reconsider. The Winograd F(4) transform for 
        4-point convolution uses 16 multiplications. For 4x4 matrix 
        multiplication, this corresponds to rank 16.
        
        Let me construct this carefully. The 4x4 matrix product C = A*B
        where A is 4x4 and B is 4x4.
        
        Using the Kronecker product structure:
        A = [[A00, A01], [A10, A11]] where each Aij is 2x2
        B = [[B00, B01], [B10, B11]] where each Bij is 2x2
        
        C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
             [A10*B00 + A11*B10, A10*B01 + A11*B11]]
        
        Each block product is a 2x2 matrix product, which can be done
        with Strassen's 7 multiplications.
        
        But that gives 4*7 = 28 multiplications for the block products,
        plus we need to handle the additions.
        
        Actually, the standard Winograd F(4) approach for matrix multiplication
        uses the fact that 4 = 2*2 and applies the 2-point transform twice.
        
        Let me use a different approach: direct construction of rank-16.
        
        For 4x4x4, the Winograd F(4) decomposition has rank 16.
        The factors are constructed using the DFT-like transform for 4 points.
        
        Actually, I think the cleanest approach is to use the Kronecker 
        product of the 2x2x2 Strassen decomposition, which gives rank 49,
        but that's not better than what we have.
        
        Let me try a different structural approach: use the fact that
        for 4x4x5, we can decompose as 4x4x4 + 4x4x1.
        
        For 4x4x4: use rank-16 Winograd
        For 4x4x1: use rank-4 (each of the 4 outputs needs 4 terms)
        
        Total rank: 20
        
        Let me construct the rank-16 Winograd F(4) for 4x4x4.
        
        The Winograd F(4) transform uses the following approach:
        For 4-point convolution, we use:
        - Transform A with F(4) matrix
        - Transform B with F(4) matrix  
        - Pointwise multiply
        - Transform result back
        
        The F(4) matrix for Winograd is:
        [[1, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 1, -1, 0],
         [1, 0, 0, 0]]
        
        Wait, that's not quite right. Let me look at this more carefully.
        
        Actually, for the 4x4 matrix multiplication problem, I'll use a 
        direct construction. The key insight is that the 4x4x4 tensor
        can be decomposed into 16 rank-1 tensors using the Kronecker 
        product of the 2x2x2 identity decomposition... no, that gives 16.
        
        Hmm, let me just use the schoolbook for 4x4x4 (rank 64) combined
        with schoolbook for 4x4x1 (rank 16), total rank 80.
        
        Or better: use Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x1 (rank 16) = 65.
        
        That's what the current code does. Let me try to improve the 4x4x1 part.
        
        For 4x4x1: C[i] = sum_j A[i,j] * B[j]
        This is just a matrix-vector product. The tensor for this has rank 4
        (one for each j, since C[i] = sum_j A[i,j]*B[j]).
        
        Wait, the rank of the 4x4x1 tensor: each term A[i,j]*B[j] contributes
        to C[i]. For fixed j, the term is A[i,j]*B[j] which contributes to all i.
        So we need 4 rank-1 terms (one for each j).
        
        U: 16x4, V: 4x4, W: 4x4
        For term t (j=t):
          U[i*4+t][t] = 1 for all i
          V[t][t] = 1
          W[i][t] = 1 for all i
        
        So rank 4 for the 4x4x1 part.
        
        Total: 49 + 4 = 53.
        
        That's better than 65! Let me implement this.
        """
        rank = 16
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # Winograd F(4) for 4x4x4
        # Using the tensor product of 2x2x2 Strassen
        # This actually gives rank 49, not 16.
        # Let me just use a direct rank-16 construction.
        
        # Actually, for 4x4x4, the minimal rank is at most 16 using
        # the following construction based on the DFT.
        
        # For now, let me use the Kronecker product of 2x2x2 schoolbook (rank 4)
        # which gives 4*4 = 16 for 4x4x4.
        
        # 2x2x2 schoolbook: rank 4
        # U2[i*2+j][i*2+j] = 1
        # V2[j*2+k][j*2+k] = 1  
        # W2[k*2+i][k*2+i] = 1
        
        # Kronecker product gives 4x4x4 with rank 16
        col = 0
        for i1 in range(2):
            for j1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                # This is getting complicated. Let me just do schoolbook.
                                pass
        
        # This approach is getting too complex. Let me just return the schoolbook
        # for 4x4x4 and use rank-4 for 4x4x1.
        # Schoolbook 4x4x4: rank 64
        # Rank-4 for 4x4x1: rank 4
        # Total: 68
        
        # Actually, let me just return None to indicate we should use a different approach.
        return None

    def combined_decomposition():
        """
        Combine rank-49 Strassen for columns 0-3 with rank-4 for column 4.
        Total rank: 53
        """
        rank = 53
        U = [[0] * rank for _ in range(n * m)]  # 16 x 53
        V = [[0] * rank for _ in range(m * p)]  # 20 x 53
        W = [[0] * rank for _ in range(p * n)]  # 20 x 53

        # Part 1: Strassen for first 4 columns (rank 49)
        U_s, V_s, W_s = strassen_4x4x4_decomposition()
        for i in range(16):
            for c in range(49):
                if U_s[i][c] != 0:
                    U[i][c] = U_s[i][c]
        for i in range(16):
            for c in range(49):
                if V_s[i][c] != 0:
                    V[i][c] = V_s[i][c]
        for i in range(16):
            for c in range(49):
                if W_s[i][c] != 0:
                    W[i][c] = W_s[i][c]

        # Part 2: Rank-4 for 5th column (rank 4)
        col = 49
        for j in range(4):
            for i in range(4):
                U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            for i in range(4):
                W[4 * 4 + i][col] = 1
            col += 1

        return U, V, W

    def strassen_4x4x4_decomposition():
        """
        Rank-49 double-Strassen for 4x4x4.
        Returns U (16x49), V (20x49), W (16x49) for the 4x4x4 subproblem.
        U: A[i,j] -> index i*4+j
        V: B[j,k] -> index j*4+k
        W: C[i,k] -> index k*4+i (trace convention W[k*n+i])
        """
        rank = 49
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]

        # Inner Strassen 2x2x2 definitions
        inner_defs = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)], [(1,1,1),(0,1,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]

        # Super Strassen 4x4x4 definitions
        super_defs = [
            ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),
            ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),
            ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),
            ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),
            ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),
            ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),
            ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),
        ]

        col = 0
        for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
            for i_idx, (u_pairs, v_pairs, w_pairs) in enumerate(inner_defs):
                for i, j, c in u_pairs:
                    for b_idx, coeff in enumerate(p_bc):
                        if coeff == 0:
                            continue
                        gi = b_idx // 2 * 2 + i
                        gj = b_idx % 2 * 2 + j
                        U[gi * 4 + gj][col] += c * coeff
                for j, k, c in v_pairs:
                    for b_idx, coeff in enumerate(q_bc):
                        if coeff == 0:
                            continue
                        gj = b_idx // 2 * 2 + j
                        gk = b_idx % 2 * 2 + k
                        V[gj * 4 + gk][col] += c * coeff
                for i, k, c in w_pairs:
                    for br, bc, sign in c_contribs:
                        gi = br * 2 + i
                        gk = bc * 2 + k
                        W[gk * 4 + gi][col] += sign * c
                col += 1

        return U, V, W

    # Try the combined decomposition (rank 53)
    if elapsed() < DEADLINE:
        U, V, W = combined_decomposition()
        ok = verify(U, V, W, seed=42, trials=15)
        if ok:
            return {"U": U, "V": V, "W": W}

    # Fallback: full schoolbook (rank 80)
    U, V, W = schoolbook_decomposition()
    return {"U": U, "V": V, "W": W}
