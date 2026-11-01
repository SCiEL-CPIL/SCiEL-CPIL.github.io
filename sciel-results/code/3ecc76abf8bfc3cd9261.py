<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
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
    
    # Verify the decomposition is correct
    # The tensor T[a, b, c] should equal sum_r U[a,r] * V[b,r] * W[c,r]
    # where a indexes (i,j) as i*m+j, b indexes (j,k) as j*p+k, c indexes (k,i) as k*n+i
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check a few entries to verify correctness
    # T[i*m+j, j*p+k, k*n+i] should be 1, all other entries should be 0
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Compute T[a,b,c] from the decomposition
                val = 0.0
                for r in range(rank):
                    val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                if abs(val - 1.0) > 1e-9:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    # If verification fails, return the schoolbook anyway (it should always be correct)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # We'll build a Strassen-based decomposition for the 2x5x5 matrix multiplication.
    # Split j-index into {0,1} (size 2) and {2,3,4} (size 3)
    # Split k-index into {0,1} (size 2) and {2,3,4} (size 3)
    # For the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}), use Strassen (7 terms)
    # For remaining blocks, use schoolbook.
    
    U_terms = []  # list of (index_a, value) pairs
    V_terms = []  # list of (index_b, value) pairs
    W_terms = []  # list of (index_c, value) pairs
    
    def add_term(a_idx, b_idx, c_idx, ua, vb, wc):
        """Add a rank-1 term with given indices and coefficients."""
        U_terms.append((a_idx, ua))
        V_terms.append((b_idx, vb))
        W_terms.append((c_idx, wc))
    
    # Strassen's 7 products for 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1, 1, 1)       # A00*B00 contributes to C00
    add_term(1, 1, 0, 1, 1, 1)       # A11*B11 contributes to C00 (but wait, C index is k*n+i)
    
    # Let me reconsider the indexing carefully.
    # U index: i*m + j where i in {0,1}, j in {0,1,2,3,4}
    # V index: j*p + k where j in {0,1,2,3,4}, k in {0,1,2,3,4}  
    # W index: k*n + i where k in {0,1,2,3,4}, i in {0,1}
    # 
    # For the 2x2x2 Strassen block: i in {0,1}, j in {0,1}, k in {0,1}
    # A[i,j] has U-index i*5+j
    # B[j,k] has V-index j*5+k
    # C[i,k] has W-index k*2+i
    
    # Strassen products for computing C[i,k] = sum_j A[i,j]*B[j,k] for i,k in {0,1}, j in {0,1}:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[0,0])
    # M4 = A[1,1]*(B[1,1]-B[0,1])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # In the tensor decomposition, each "product" M is a rank-1 term.
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) means:
    #   U has +1 at index 0*5+0=0 and +1 at index 1*5+1=6
    #   V has +1 at index 0*5+0=0 and +1 at index 1*5+1=6
    #   W: this contributes to C[0,0] and C[1,1]
    #     C[0,0] has W-index 0*2+0=0, coefficient +1
    #     C[1,1] has W-index 1*2+1=3, coefficient +1
    
    # This approach is getting complex. Let me use a cleaner formulation.
    # Each rank-1 term is (u_vector, v_vector, w_vector) where:
    #   u_vector has entries in R^{n*m} = R^{10}
    #   v_vector has entries in R^{m*p} = R^{25}
    #   w_vector has entries in R^{p*n} = R^{10}
    
    terms = []  # list of (u_dict, v_dict, w_dict) where each is {index: coeff}
    
    def add_rank1(u_coeffs, v_coeffs, w_coeffs):
        """Add a rank-1 term. Each is a dict {index: coefficient}."""
        terms.append((u_coeffs, v_coeffs, w_coeffs))
    
    # Strassen 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # Contributes: +1 to C[0,0], +1 to C[1,1]
    add_rank1(
        {0*5+0: 1, 1*5+1: 1},  # A[0,0], A[1,1]
        {0*5+0: 1, 1*5+1: 1},  # B[0,0], B[1,1]
        {0*2+0: 1, 1*2+1: 1}   # C[0,0], C[1,1]
    )
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # Contributes: +1 to C[1,0]
    add_rank1(
        {1*5+0: 1, 1*5+1: 1},  # A[1,0], A[1,1]
        {0*5+0: 1},             # B[0,0]
        {0*2+1: 1}              # C[1,0]
    )
    
    # M3 = A[0,0] * (B[1,0]-B[0,0])
    # Contributes: +1 to C[0,1]
    add_rank1(
        {0*5+0: 1},             # A[0,0]
        {1*5+0: 1, 0*5+0: -1}, # B[1,0], -B[0,0]
        {1*2+0: 1}              # C[0,1]
    )
    
    # M4 = A[1,1] * (B[1,1]-B[0,1])
    # Contributes: +1 to C[0,0], +1 to C[1,0]
    add_rank1(
        {1*5+1: 1},             # A[1,1]
        {1*5+1: 1, 0*5+1: -1}, # B[1,1], -B[0,1]
        {0*2+0: 1, 0*2+1: 1}   # C[0,0], C[1,0]
    )
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # Contributes: -1 to C[0,0], +1 to C[0,1]
    add_rank1(
        {0*5+0: 1, 0*5+1: 1},  # A[0,0], A[0,1]
        {1*5+1: 1},             # B[1,1]
        {0*2+0: -1, 1*2+0: 1}  # -C[0,0], +C[0,1]
    )
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # Contributes: +1 to C[1,1]
    add_rank1(
        {1*5+0: 1, 0*5+0: -1}, # A[1,0], -A[0,0]
        {0*5+0: 1, 0*5+1: 1},  # B[0,0], B[0,1]
        {1*2+1: 1}              # C[1,1]
    )
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # Contributes: +1 to C[0,0]
    add_rank1(
        {0*5+1: 1, 1*5+1: -1}, # A[0,1], -A[1,1]
        {1*5+0: 1, 1*5+1: 1},  # B[1,0], B[1,1]
        {0*2+0: 1}              # C[0,0]
    )
    
    # Now add schoolbook terms for the remaining blocks:
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3,4} (2x2x3)
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_rank1(
                    {i*5+j: 1},
                    {j*5+k: 1},
                    {k*2+i: 1}
                )
    
    # Block 3: i in {0,1}, j in {2,3,4}, k in {0,1} (2x3x2)
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_rank1(
                    {i*5+j: 1},
                    {j*5+k: 1},
                    {k*2+i: 1}
                )
    
    # Block 4: i in {0,1}, j in {2,3,4}, k in {2,3,4} (2x3x3)
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_rank1(
                    {i*5+j: 1},
                    {j*5+k: 1},
                    {k*2+i: 1}
                )
    
    rank = len(terms)
    # Build U, V, W matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u_coeffs, v_coeffs, w_coeffs) in enumerate(terms):
        for idx, coeff in u_coeffs.items():
            U[idx][r] = coeff
        for idx, coeff in v_coeffs.items():
            V[idx][r] = coeff
        for idx, coeff in w_coeffs.items():
            W[idx][r] = coeff
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_np[a] * V_np[b] * W_np[c])
                if abs(val - 1.0) > 1e-9:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    # Also verify that non-matching entries are zero
    if verified:
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if this is a valid (i,j,k) triple
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        continue  # Should be 1, already verified
                    else:
                        val = np.sum(U_np[a] * V_np[b] * W_np[c])
                        if abs(val) > 1e-9:
                            verified = False
                            break
                if not verified:
                    break
            if not verified:
                break
        if not verified:
            break
    
    if not verified:
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
>>>>>>> REPLACE

Wait, I need to verify my Strassen coefficients are correct. Let me trace through:

For C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]:
- M1 contributes +1 (A[0,0]*B[0,0] part)
- M4 contributes +1 (A[1,1]*(B[1,1]-B[0,1])... wait this doesn't have A[0,0]*B[0,0])

I think I made an error in the Strassen formulas. Let me redo this carefully.

Standard Strassen for 2×2 matrices:
C[0,0] = M1 + M4 - M5 + M7
C[0,1] = M3 + M5
C[1,0] = M2 + M4
C[1,1] = M1 - M2 + M3 + M6

Where:
M1 = (A00+A11)(B00+B11)
M2 = (A10+A11)B00
M3 = A00(B10-B00)
M4 = A11(B11-B01)
M5 = (A00+A01)B11
M6 = (A10-A00)(B00+B01)
M7 = (A01-A11)(B10+B11)

Let me verify C[0,0]:
M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
M4 = A11*B11 - A11*B01
M5 = A00*B11 + A01*B11
M7 = A01*B10 + A01*B11 - A11*B10 - A11*B11

M1+M4-M5+M7 = (A00*B00 + A00*B11 + A11*B00 + A11*B11) + (A11*B11 - A11*B01) - (A00*B11 + A01*B11) + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
= A00*B00 + A00*B11 - A00*B11 + A11*B00 + A11*B11 + A11*B11 - A11*B11 - A11*B01 - A01*B11 + A01*B10 + A01*B11 - A11*B10
= A00*B00 + A11*B00 - A11*B01 + A01*B10 - A11*B10
= A00*B00 + A11*(B00-B01) + A01*B10 - A11*B10

That's not equal to A00*B00 + A01*B10. I'm making errors. Let me look up the correct Strassen formula.

The correct Strassen algorithm:
M1 = (A00 + A11) * (B00 + B11)
M2 = (A10 + A11) * B00
M3 = A00 * (B10 - B00)
M4 = A11 * (B11 - B01)
M5 = (A00 + A01) * B11
M6 = (A10 - A00) * (B00 + B01)
M7 = (A01 - A11) * (B10 + B11)

C00 = M1 + M4 - M5 + M7
C01 = M3 + M5
C10 = M2 + M4
C11 = M1 - M2 + M3 + M6

Let me verify C00 = A00*B00 + A01*B10:
M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
M4 = A11*B11 - A11*B01
M5 = A00*B11 + A01*B11
M7 = A01*B10 + A01*B11 - A11*B10 - A11*B11

M1+M4-M5+M7:
A00 terms: A00*B00 + A00*B11 - A00*B11 = A00*B00 ✓
A01 terms: -A01*B11 + A01*B10 + A01*B11 = A01*B10 ✓
A11 terms: A11*B00 + A11*B11 + A11*B11 - A11*B01 - A11*B10 - A11*B11 = A11*B00 + A11*B11 - A11*B01 - A11*B10

Hmm, there are extra A11 terms. That means my formula is wrong. Let me reconsider.

Actually, I think the issue is that Strassen's algorithm computes C = A*B where C[i,j] = sum_k A[i,k]*B[k,j]. In our problem, C[i,k] = sum_j A[i,j]*B[j,k]. So the roles are slightly different.

In standard matrix multiplication C = A*B: C[i,j] = sum_k A[i,k]*B[k,j]
In our problem: C[i,k] = sum_j A[i,j]*B[j,k]

So if I set A_strassen = A (2x2), B_strassen = B^T (2x2), then C_strassen = A*B^T gives C_strassen[i,j] = sum_k A[i,k]*B^T[k,j] = sum_k A[i,k]*B[j,k]. So C_strassen[i,j] = C[i,j] in our notation where the second index of C is k.

Wait, I'm getting confused. Let me just directly verify that my tensor decomposition is correct by checking that the sum of rank-1 terms gives the right tensor.

Let me just fix the code to properly verify and use the correct Strassen formulas. The key insight is that each M in Strassen is a bilinear form, and in the tensor decomposition, each M corresponds to a rank-1 term where the U, V, W vectors encode the linear combinations.

Let me rewrite this more carefully:

