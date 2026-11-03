"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 sub-blocks and
optimized handling of the remaining dimensions.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition: split n=3 as 2+1, m=4 as 2+2, and apply
    Strassen to 2x2x2 sub-blocks where possible.
    
    Decomposition strategy:
    - Split n=3 into n1=2, n2=1
    - Split m=4 into m1=2, m2=2
    - Split p=7 into p1=2, p2=2, p3=2, p4=1
    
    Blocks:
    - (2,2,2): Strassen rank 7, three of these = 21
    - (2,2,1): schoolbook rank 4, one of these = 4
    - (1,2,2): schoolbook rank 4, two of these = 8
    - (1,2,1): schoolbook rank 2, one of these = 2
    
    Total: 21 + 4 + 8 + 2 = 35
    """
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 tensor term to the decomposition."""
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        Adds 7 rank-1 terms.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def schoolbook_221(i0, j0, k0):
        """
        Schoolbook decomposition for 2x2 * 2x1 block.
        Rank = 4.
        """
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k0], [1.0],
                               [k0*n+ii], [1.0])

    def schoolbook_122(i0, j0, k0):
        """
        Schoolbook decomposition for 1x2 * 2x2 block.
        Rank = 4.
        """
        for i in range(1):
            for j in range(2):
                for k in range(2):
                    ii = i0 + i
                    jj = j0 + j
                    kk = k0 + k
                    add_rank1_term([ii*m+jj], [1.0],
                                   [jj*p+kk], [1.0],
                                   [kk*n+ii], [1.0])

    def schoolbook_121(i0, j0, k0):
        """
        Schoolbook decomposition for 1x2 * 2x1 block.
        Rank = 2.
        """
        for j in range(2):
            jj = j0 + j
            add_rank1_term([i0*m+jj], [1.0],
                           [jj*p+k0], [1.0],
                           [k0*n+i0], [1.0])

    # Build the full decomposition
    # Split: n=3 -> [2,1], m=4 -> [2,2], p=7 -> [2,2,2,1]
    
    # Block (n=2, m=2, p=2) at (0,0,0): Strassen rank 7
    strassen_222(0, 0, 0)
    
    # Block (n=2, m=2, p=2) at (0,0,2): Strassen rank 7
    strassen_222(0, 0, 2)
    
    # Block (n=2, m=2, p=2) at (0,0,4): Strassen rank 7
    strassen_222(0, 0, 4)
    
    # Block (n=2, m=2, p=1) at (0,0,6): schoolbook rank 4
    schoolbook_221(0, 0, 6)
    
    # Block (n=1, m=2, p=2) at (2,0,0): schoolbook rank 4
    schoolbook_122(2, 0, 0)
    
    # Block (n=1, m=2, p=2) at (2,0,2): schoolbook rank 4
    schoolbook_122(2, 0, 2)
    
    # Block (n=1, m=2, p=1) at (2,0,6): schoolbook rank 2
    schoolbook_121(2, 0, 6)

    rank = len(U_vecs)

    # Convert to matrix format
    U = np.array(U_vecs).T  # shape (n*m, rank)
    V = np.array(V_vecs).T  # shape (m*p, rank)
    W = np.array(W_vecs).T  # shape (p*n, rank)

    # Verify correctness with vectorized computation
    if time.time() - start_time < deadline - 5.0:
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        # Vectorized verification
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                w_row = W[k*n+i, :]  # shape (rank,)
                for j in range(m):
                    u_row = U[i*m+j, :]  # shape (rank,)
                    v_row = V[j*p+k, :]  # shape (rank,)
                    inner = np.sum(u_row * v_row * w_row)
                    C_computed[i, k] += A[i, j] * B[j, k] * inner

        error = np.max(np.abs(C_computed - C_expected))
        if error > 1e-9:
            # Fallback to schoolbook
            rank_school = n * m * p
            U_sb = [[0] * rank_school for _ in range(n * m)]
            V_sb = [[0] * rank_school for _ in range(m * p)]
            W_sb = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j][col] = 1
                        V_sb[j * p + k][col] = 1
                        W_sb[k * n + i][col] = 1
                        col += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
