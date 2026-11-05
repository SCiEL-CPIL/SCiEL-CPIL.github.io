"""Strassen + schoolbook hybrid decomposition for 3x3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    t0 = time.time()
    deadline = t0 + 110.0

    # Strategy: Split B's 8 columns into 4 pairs of 2.
    # For each pair (k0, k1), we compute a 2x2x2 block product:
    #   A_block (2x2) * B_block (2x2) -> C_block (2x2)
    # using Strassen (rank 7).
    # The remaining row/col j=2 is handled with schoolbook.
    #
    # Total rank = 4 * 7 (Strassen for 4 k-pairs) + 24 (schoolbook for j=2)
    #            = 28 + 24 = 52
    #
    # Wait, let me reconsider. For each k-pair, we use Strassen on the
    # (i in {0,1}, j in {0,1}) sub-block. That gives 7 terms per k-pair.
    # For j=2, we need schoolbook for all i in {0,1,2} and all k in {0,...,7}:
    # that's 3 * 8 = 24 terms.
    # Total = 28 + 24 = 52.
    
    # But wait - I need to verify the Strassen decomposition is correct.
    # Let me construct it carefully and verify.
    
    # Strassen decomposition for 2x2x2:
    # Given A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    #
    # 7 products:
    # M0 = (a00 + a11) * (b00 + b11)
    # M1 = (a10 + a11) * b00
    # M2 = a00 * (b01 - b11)
    # M3 = a11 * b10
    # M4 = (a00 + a01) * b10
    # M5 = (a10 - a00) * (b00 + b01)
    # M6 = (a01 - a11) * (b10 + b11)
    #
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # For the tensor decomposition, each M corresponds to a rank-1 term:
    # U vector has entries for A positions, V for B positions, W for C positions.
    #
    # For a specific k-pair (k0, k1), the indices are:
    # i in {0,1}, j in {0,1}, k in {k0, k1}
    # U index: i*m + j (m=3)
    # V index: j*p + k (p=8)
    # W index: k*n + i (n=3)
    
    # Let me build the decomposition column by column.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # For each k-pair (k0, k1), add 7 Strassen terms
    # A positions: (0,0)->idx 0, (0,1)->idx 1, (1,0)->idx 3, (1,1)->idx 4
    # B positions: (0,k0)->idx k0, (0,k1)->idx k1, (1,k0)->idx 8+k0, (1,k1)->idx 8+k1
    # C positions: (k0,0)->idx 0, (k0,1)->idx 1, (k1,0)->idx 3, (k1,1)->idx 4
    
    for pair_idx in range(4):
        k0 = pair_idx * 2
        k1 = pair_idx * 2 + 1
        
        # A indices
        A00 = 0  # i=0,j=0
        A01 = 1  # i=0,j=1
        A10 = 3  # i=1,j=0
        A11 = 4  # i=1,j=1
        
        # B indices
        B00 = 0 * p + k0      # j=0,k=k0
        B01 = 0 * p + k1      # j=0,k=k1
        B10 = 1 * p + k0      # j=1,k=k0
        B11 = 1 * p + k1      # j=1,k=k1
        
        # C indices
        C00 = k0 * n + 0      # k=k0,i=0
        C01 = k0 * n + 1      # k=k0,i=1
        C10 = k1 * n + 0      # k=k1,i=0
        C11 = k1 * n + 1      # k=k1,i=1
        
        def make_u(indices):
            vec = [0] * (n * m)
            for idx in indices:
                vec[idx] = 1
            return vec
        
        def make_v(indices):
            vec = [0] * (m * p)
            for idx in indices:
                vec[idx] = 1
            return vec
        
        def make_w(indices):
            vec = [0] * (p * n)
            for idx in indices:
                vec[idx] = 1
            return vec
        
        # M0 = (a00 + a11) * (b00 + b11)
        # U: A00, A11 ; V: B00, B11 ; W: C00, C01, C10, C11
        # But the signs in C reconstruction need to be encoded in W.
        # C00 += M0, C01 += M0, C10 += M0, C11 += M0
        # So W has 1 at all four C positions.
        add_col(make_u([A00, A11]), make_v([B00, B11]), make_w([C00, C01, C10, C11]))
        
        # M1 = (a10 + a11) * b00
        # C10 += M1, C11 -= M1
        # W: C10=1, C11=-1
        vec_w = [0] * (p * n)
        vec_w[C10] = 1
        vec_w[C11] = -1
        add_col(make_u([A10, A11]), make_v([B00]), vec_w)
        
        # M2 = a00 * (b01 - b11)
        # C01 += M2, C11 += M2
        vec_v = [0] * (m * p)
        vec_v[B01] = 1
        vec_v[B11] = -1
        vec_w = [0] * (p * n)
        vec_w[C01] = 1
        vec_w[C11] = 1
        add_col(make_u([A00]), vec_v, vec_w)
        
        # M3 = a11 * b10
        # C00 += M3, C10 += M3
        vec_w = [0] * (p * n)
        vec_w[C00] = 1
        vec_w[C10] = 1
        add_col(make_u([A11]), make_v([B10]), vec_w)
        
        # M4 = (a00 + a01) * b10
        # C00 -= M4, C01 += M4
        vec_w = [0] * (p * n)
        vec_w[C00] = -1
        vec_w[C01] = 1
        add_col(make_u([A00, A01]), make_v([B10]), vec_w)
        
        # M5 = (a10 - a00) * (b00 + b01)
        # C11 += M5
        vec_u = [0] * (n * m)
        vec_u[A10] = 1
        vec_u[A00] = -1
        vec_w = [0] * (p * n)
        vec_w[C11] = 1
        add_col(vec_u, make_v([B00, B01]), vec_w)
        
        # M6 = (a01 - a11) * (b10 + b11)
        # C00 -= M6, C11 += M6
        vec_u = [0] * (n * m)
        vec_u[A01] = 1
        vec_u[A11] = -1
        vec_w = [0] * (p * n)
        vec_w[C00] = -1
        vec_w[C11] = 1
        add_col(vec_u, make_v([B10, B11]), vec_w)
    
    # Now handle j=2 with schoolbook for all i in {0,1,2} and k in {0,...,7}
    for i in range(3):
        for k in range(8):
            u = [0] * (n * m)
            u[i * m + 2] = 1  # j=2
            v = [0] * (m * p)
            v[2 * p + k] = 1  # j=2, k
            w = [0] * (p * n)
            w[k * n + i] = 1  # k, i
            add_col(u, v, w)
    
    total_rank = len(U_cols)
    
    # Build the matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for c in range(total_rank):
        for r in range(n * m):
            U[r][c] = U_cols[c][r]
        for r in range(m * p):
            V[r][c] = V_cols[c][r]
        for r in range(p * n):
            W[r][c] = W_cols[c][r]
    
    # Verify correctness
    if time.time() < deadline - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1 for all i,j,k
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(total_rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    max_err = max(max_err, abs(val - 1))
        
        if max_err > 1e-10:
            # Fallback to schoolbook if verification fails
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
