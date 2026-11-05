"""
Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.

We use a block-based approach with Strassen's algorithm for 2x2 blocks.

The key idea:
- Split the 4x5 * 5x6 multiplication into blocks
- Use Strassen's algorithm (rank 7) for 2x2 * 2x2 blocks instead of schoolbook (rank 8)
- This reduces the total rank below the schoolbook 120

Strategy:
- A (4x5): split rows into 2+2, columns into 2+2+1
- B (5x6): split rows into 2+2+1, columns into 2+2+2
- C (4x6): result of block multiplications

For each pair of row-blocks and column-blocks, we compute a 2x2 or 2x1 or 1x2
sub-multiplication. The 2x2 * 2x2 blocks use Strassen (rank 7 each).

Let's be more systematic. We'll construct the decomposition by:
1. Identifying all the "multiplication terms" needed
2. Grouping them into Strassen-like patterns where possible

Actually, let me use a cleaner approach:

The matrix multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1.

For a rank-r decomposition, we need:
T[i*m+j, j*p+k, k*n+i] = sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Schoolbook: rank = 4*5*6 = 120, with U[i*m+j, i*m+j] = 1, etc.

To reduce rank, we look for linear dependencies. For Strassen on 2x2*2x2:
The 8 schoolbook terms can be replaced by 7 terms using the identity that
certain combinations of products are linearly dependent.

For our case, let's try to find a decomposition with rank < 120.

Approach: Use the fact that we can group terms. Specifically, for fixed j,
the terms (i, j, k) for all i,k form a 4x6 grid. If we can express this grid
as a sum of few rank-1 tensors (in the i,k space), we save rank.

For a 4x6 grid, the schoolbook uses 24 terms. Can we do better?
A 4x6 matrix has rank at most 4, so we can express it as a sum of 4 rank-1
tensors. But that's for a specific matrix, not for the symbolic case.

Actually, the tensor entry T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k).
For fixed j, we have a 4x6 array of 1s. This is a rank-1 matrix (all ones),
so it can be written as outer product of a 4-vector of 1s and a 6-vector of 1s.

But that's not quite right because the U, V, W factors are shared across j.

Let me try a different approach: construct a valid low-rank decomposition
using known results.

For 2x2 * 2x2: rank 7 (Strassen)
For 2x2 * 2x3: we can split into 2x2*2x2 + 2x2*2x1, using 7 + 4 = 11 (vs schoolbook 12)
For 2x3 * 3x2: similar, rank 11 (vs 12)
For 2x3 * 3x3: split into 2x2*2x2 + 2x2*2x1 + 2x1*2x2 + 2x1*2x1... this gets complicated.

Let me just try to construct a decomposition with rank around 100-110 and verify it.

Actually, the simplest improvement: use Strassen for the 2x2 sub-blocks.

Split:
- A (4x5) into 2x2 blocks: A = [[A11, A12, A13], [A21, A22, A23]]
  where A11, A12 are 2x2, A13 is 2x1, A21, A22 are 2x2, A23 is 2x1
- B (5x6) into 2x2 blocks: B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
  where B11..B23 are 2x2, B31..B33 are 1x2

C = A * B is 4x6:
C = [[A11*B11 + A12*B21 + A13*B31,  A11*B12 + A12*B22 + A13*B32,  A11*B13 + A12*B23 + A13*B33],
     [A21*B11 + A22*B21 + A23*B31,  A21*B12 + A22*B22 + A23*B32,  A21*B13 + A22*B23 + A23*B33]]

Each 2x2*2x2 term uses Strassen (rank 7).
Each 2x2*2x1 term: schoolbook rank 4, can we do better? 2x2*2x1 is like multiplying a 2x2 by a 2x1 vector. That's 4 scalar multiplications, which is minimal (rank 4).
Each 2x1*2x2 term: similarly rank 4.
Each 1x2*2x2 term: similarly rank 4.

So for the 2x2*2x2 blocks: there are 6 such blocks in C (positions (1,1), (1,2), (1,3), (2,1), (2,2), (2,3) in the 2x2 block structure of C... wait, C is 4x6, so it's 2x3 blocks of size 2x2.

Actually C is 4x6 = (2+2) x (2+2+2), so it has 2*3 = 6 blocks of size 2x2.

Each 2x2 block of C is a sum of three products:
C_ij = A_i1 * B_1j + A_i2 * B_2j + A_i3 * B_3j (for i=1,2 and j=1,2,3)

Where A_i1, A_i2 are 2x2, A_i3 is 2x1; B_1j, B_2j are 2x2, B_3j is 1x2.

So each C_ij block = (2x2 * 2x2) + (2x2 * 2x2) + (2x1 * 1x2)
Wait, that's not right. A_i1 is 2x2, B_1j is 2x2, so A_i1 * B_1j is 2x2.
A_i2 is 2x2, B_2j is 2x2, so A_i2 * B_2j is 2x2.
A_i3 is 2x1, B_3j is 1x2, so A_i3 * B_3j is 2x2.

So each 2x2 block of C is a sum of three 2x2 matrices, each from a different type of multiplication.

For the first two (2x2*2x2), we use Strassen (rank 7 each).
For the third (2x1*1x2), we use schoolbook (rank 4).

Total per 2x2 block of C: 7 + 7 + 4 = 18 rank terms.
6 blocks * 18 = 108 rank terms.

But wait, the Strassen terms for different blocks might share some structure. Let's just construct it directly.

Actually, let me just build the decomposition explicitly and count the rank.

Let me use a more direct approach: I'll construct U, V, W matrices by enumerating the terms.

For each (i, j, k) triple, the schoolbook has one term. To apply Strassen, I need to
group certain (i, j, k) triples together.

Let me just go with a practical approach: construct a valid decomposition with rank < 120
by using Strassen for the 2x2 sub-blocks.

Here's my plan:
- For the 2x2 * 2x2 sub-blocks (where both A-block and B-block are 2x2), use Strassen's 7 terms
- For other sub-blocks, use schoolbook

Let me implement this carefully.
"""
import numpy as np
import time

def construct():
    """
    Construct an exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The decomposition satisfies:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    where T[i*m+j, j*p+k, k*n+i] = 1 and 0 otherwise.
    """
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 4, 5, 6
    
    # We'll build the decomposition term by term
    # Each term is a triple (u_vec, v_vec, w_vec) where these are sparse vectors
    # U has shape (n*m, rank) = (20, rank)
    # V has shape (m*p, rank) = (30, rank)  
    # W has shape (p*n, rank) = (24, rank)
    
    terms_u = []  # list of dicts or sparse representations
    terms_v = []
    terms_w = []
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        """Add a rank-1 term with given index-value pairs."""
        terms_u.append(dict(zip(u_indices, u_vals)))
        terms_v.append(dict(zip(v_indices, v_vals)))
        terms_w.append(dict(zip(w_indices, w_vals)))
    
    # Strassen's algorithm for 2x2 * 2x2
    # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+cg, bf+dh], [ce+ag, df+bh]]
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # In our tensor notation:
    # A is 2x2 with entries A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B is 2x2 with entries B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C[0,0] = ae + cg, C[0,1] = bf + dh
    # C[1,0] = ce + ag, C[1,1] = df + bh
    
    # The tensor indices:
    # For A[i,j] where i is row (0,1) and j is col (0,1) within the block
    # For B[j',k] where j' is row (0,1) and k is col (0,1) within the block
    # For C[i',k'] where i' is row (0,1) and k' is col (0,1) within the block
    # The multiplication: C[i',k'] = sum_j A[i',j] * B[j,k']
    
    # In global indices:
    # A block at rows (r0, r0+1), cols (c0, c0+1): global A index = (r0+i)*m + (c0+j)
    # B block at rows (r0, r0+1), cols (c0, c0+1): global B index = (r0+j)*p + (c0+k)
    # C block at rows (r0, r0+1), cols (c0, c0+1): global C index = (c0+k)*n + (r0+i)
    
    # Let's define the Strassen terms for a 2x2*2x2 block
    # where A block is at rows (ar0, ar0+1), cols (ac0, ac0+1)
    # B block is at rows (br0, br0+1), cols (bc0, bc0+1)
    # C block is at rows (cr0, cr0+1), cols (cc0, cc0+1)
    # with ar0 = cr0, ac0 = br0, bc0 = cc0 (for the multiplication to make sense)
    
    def add_strassen_2x2(ar0, ac0, br0, bc0):
        """Add Strassen terms for A[ar0:ar0+2, ac0:ac0+2] * B[br0:br0+2, bc0:bc0+2] -> C[ar0:ar0+2, bc0:bc0+2]"""
        # A indices: A[ar0, ac0]=a, A[ar0, ac0+1]=b, A[ar0+1, ac0]=c, A[ar0+1, ac0+1]=d
        # B indices: B[br0, bc0]=e, B[br0, bc0+1]=f, B[br0+1, bc0]=g, B[br0+1, bc0+1]=h
        # C indices: C[ar0, bc0], C[ar0, bc0+1], C[ar0+1, bc0], C[ar0+1, bc0+1]
        
        # In global tensor indices:
        # A[i,j] -> U index: i*m + j
        # B[j,k] -> V index: j*p + k
        # C[i,k] -> W index: k*n + i
        
        a_idx = (ar0, ac0)      # a
        b_idx = (ar0, ac0 + 1)  # b
        c_idx = (ar0 + 1, ac0)  # c
        d_idx = (ar0 + 1, ac0 + 1)  # d
        
        e_idx = (br0, bc0)      # e
        f_idx = (br0, bc0 + 1)  # f
        g_idx = (br0 + 1, bc0)  # g
        h_idx = (br0 + 1, bc0 + 1)  # h
        
        # C positions
        c11_idx = (ar0, bc0)      # C[0,0] in block
        c12_idx = (ar0, bc0 + 1)  # C[0,1] in block
        c21_idx = (ar0 + 1, bc0)  # C[1,0] in block
        c22_idx = (ar0 + 1, bc0 + 1)  # C[1,1] in block
        
        # Strassen multiplications:
        # m1 = (a+d)(e+h)
        add_term(
            [a_idx[0]*m+a_idx[1], d_idx[0]*m+d_idx[1]],
            [e_idx[0]*p+e_idx[1], h_idx[0]*p+h_idx[1]],
            [c11_idx[1]*n+c11_idx[0], c12_idx[1]*n+c12_idx[0], c21_idx[1]*n+c21_idx[0], c22_idx[1]*n+c22_idx[0]],
            [1, 1],
            [1, 1],
            [1, 1, 1, 1]
        )
        
        # m2 = (c+d)e
        add_term(
            [c_idx[0]*m+c_idx[1], d_idx[0]*m+d_idx[1]],
            [e_idx[0]*p+e_idx[1]],
            [c21_idx[1]*n+c21_idx[0], c22_idx[1]*n+c22_idx[0]],
            [1, 1],
            [1],
            [1, 1]
        )
        
        # m3 = a(f-h)
        add_term(
            [a_idx[0]*m+a_idx[1]],
            [f_idx[0]*p+f_idx[1], h_idx[0]*p+h_idx[1]],
            [c12_idx[1]*n+c12_idx[0], c22_idx[1]*n+c22_idx[0]],
            [1],
            [1, -1],
            [1, 1]
        )
        
        # m4 = d(g-e)
        add_term(
            [d_idx[0]*m+d_idx[1]],
            [g_idx[0]*p+g_idx[1], e_idx[0]*p+e_idx[1]],
            [c11_idx[1]*n+c11_idx[0], c21_idx[1]*n+c21_idx[0]],
            [1],
            [1, -1],
            [1, 1]
        )
        
        # m5 = (a+b)h
        add_term(
            [a_idx[0]*m+a_idx[1], b_idx[0]*m+b_idx[1]],
            [h_idx[0]*p+h_idx[1]],
            [c11_idx[1]*n+c11_idx[0], c12_idx[1]*n+c12_idx[0]],
            [1, 1],
            [1],
            [1, 1]
        )
        
        # m6 = (c-a)(e+f)
        add_term(
            [c_idx[0]*m+c_idx[1], a_idx[0]*m+a_idx[1]],
            [e_idx[0]*p+e_idx[1], f_idx[0]*p+f_idx[1]],
            [c22_idx[1]*n+c22_idx[0]],
            [1, -1],
            [1, 1],
            [1]
        )
        
        # m7 = (b-d)(g+h)
        add_term(
            [b_idx[0]*m+b_idx[1], d_idx[0]*m+d_idx[1]],
            [g_idx[0]*p+g_idx[1], h_idx[0]*p+h_idx[1]],
            [c11_idx[1]*n+c11_idx[0], c12_idx[1]*n+c12_idx[0]],
            [1, -1],
            [1, 1],
            [1, 1]
        )
    
    # Now let's build the full decomposition
    # A is 4x5, B is 5x6, C is 4x6
    # We split A rows into 2+2, cols into 2+2+1
    # B rows into 2+2+1, cols into 2+2+2
    # C is 4x6 = (2+2) x (2+2+2)
    
    # The 2x2*2x2 blocks (using Strassen):
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    # A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    # A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    # A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
    # A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]
    # A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    # A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]
    
    # That's 12 2x2*2x2 blocks, each with 7 Strassen terms = 84 terms
    
    # The remaining blocks involve the 1-wide strips:
    # A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2]  (2x1 * 1x2)
    # A[0:2, 4:5] * B[4:5, 2:4] -> C[0:2, 2:4]  (2x1 * 1x2)
    # A[0:2, 4:5] * B[4:5, 4:6] -> C[0:2, 4:6]  (2x1 * 1x2)
    # A[2:4, 4:5] * B[4:5, 0:2] -> C[2:4, 0:2]  (2x1 * 1x2)
    # A[2:4, 4:5] * B[4:5, 2:4] -> C[2:4, 2:4]  (2x1 * 1x2)
    # A[2:4, 4:5] * B[4:5, 4:6] -> C[2:4, 4:6]  (2x1 * 1x2)
    
    # Each 2x1 * 1x2 block: schoolbook rank 4
    # 6 blocks * 4 = 24 terms
    
    # Total: 84 + 24 = 108 terms
    
    # Let's verify this is correct by checking that we cover all (i,j,k) triples
    
    # Add Strassen terms for all 2x2*2x2 blocks
    for ar0 in [0, 2]:
        for ac0 in [0, 2]:
            for br0 in [0, 2]:
                if ac0 == br0:  # Only when A's col block matches B's row block
                    for bc0 in [0, 2, 4]:
                        add_strassen_2x2(ar0, ac0, br0, bc0)
    
    # Wait, I need to be more careful. The A block at (ar0, ac0) multiplies with
    # B block at (br0, bc0) only when ac0 == br0 (A's column index matches B's row index).
    # And the result goes to C block at (ar0, bc0).
    
    # Let me redo this:
    # Clear the terms and rebuild
    terms_u.clear()
    terms_v.clear()
    terms_w.clear()
    
    # 2x2 * 2x2 blocks using Strassen
    # A rows: 0-1, 2-3; A cols: 0-1, 2-3, 4
    # B rows: 0-1, 2-3, 4; B cols: 0-1, 2-3, 4-5
    # C rows: 0-1, 2-3; C cols: 0-1, 2-3, 4-5
    
    # For each combination where A col block = B row block (both 2-wide):
    for ar0 in [0, 2]:
        for ac0 in [0, 2]:
            for bc0 in [0, 2, 4]:
                # A[ar0:ar0+2, ac0:ac0+2] * B[ac0:ac0+2, bc0:bc0+2] -> C[ar0:ar0+2, bc0:bc0+2]
                add_strassen_2x2(ar0, ac0, ac0, bc0)
    
    # 2x1 * 1x2 blocks using schoolbook
    # A[ar0:ar0+2, 4] * B[4, bc0:bc0+2] -> C[ar0:ar0+2, bc0:bc0+2]
    for ar0 in [0, 2]:
        for bc0 in [0, 2, 4]:
            # A[ar0, 4] and A[ar0+1, 4] are the two A entries
            # B[4, bc0] and B[4, bc0+1] are the two B entries
            # C[ar0, bc0], C[ar0, bc0+1], C[ar0+1, bc0], C[ar0+1, bc0+1]
            
            # Schoolbook: 4 terms
            # C[ar0, bc0] = A[ar0,4] * B[4,bc0]
            add_term(
                [ar0*m + 4],
                [4*p + bc0],
                [bc0*n + ar0],
                [1], [1], [1]
            )
            # C[ar0, bc0+1] = A[ar0,4] * B[4,bc0+1]
            add_term(
                [ar0*m + 4],
                [4*p + bc0 + 1],
                [(bc0+1)*n + ar0],
                [1], [1], [1]
            )
            # C[ar0+1, bc0] = A[ar0+1,4] * B[4,bc0]
            add_term(
                [(ar0+1)*m + 4],
                [4*p + bc0],
                [bc0*n + (ar0+1)],
                [1], [1], [1]
            )
            # C[ar0+1, bc0+1] = A[ar0+1,4] * B[4,bc0+1]
            add_term(
                [(ar0+1)*m + 4],
                [4*p + bc0 + 1],
                [(bc0+1)*n + (ar0+1)],
                [1], [1], [1]
            )
    
    # Now convert to matrices
    rank = len(terms_u)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        for idx, val in terms_u[r].items():
            U[idx, r] += val
        for idx, val in terms_v[r].items():
            V[idx, r] += val
        for idx, val in terms_w[r].items():
            W[idx, r] += val
    
    # Verify the decomposition
    # T[i*m+j, j*p+k, k*n+i] should equal 1 for all i,j,k
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify correctness
    T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check against expected
    is_correct = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = T_reconstructed[i*m+j, j*p+k, k*n+i]
                if abs(actual - expected) > 1e-9:
                    is_correct = False
                    break
            if not is_correct:
                break
        if not is_correct:
            break
    
    # Also check that non-matching entries are 0
    if is_correct:
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if j != j2 or k != k2 or i != i2:
                        if abs(T_reconstructed[a, b, c]) > 1e-9:
                            is_correct = False
                            break
                if not is_correct:
                    break
            if not is_correct:
                break
    
    if not is_correct:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
