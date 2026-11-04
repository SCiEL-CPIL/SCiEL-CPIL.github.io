"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Tensor convention: T[i*2+j, j*2+k, k*2+i] = 1
    """
    # Strassen's 7 products for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # U: maps A entries to Strassen inputs
    # A indices: 0=a, 1=b, 2=c, 3=d
    U = np.zeros((4, 7))
    # V: maps B entries to Strassen inputs  
    # B indices: 0=e, 1=f, 2=g, 3=h
    V = np.zeros((4, 7))
    # W: maps C entries from Strassen outputs
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h): U has 1 at a,d; V has 1 at e,h
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1  # M1 contributes to C00 and C11
    
    # M2 = (c+d)e
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1  # M2 contributes to C10
    
    # M3 = a(f-h)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1  # M3 contributes to C01
    
    # M4 = d(g-e)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1  # M4 contributes to C00 and C10
    
    # M5 = (a+b)h
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1  # M5 contributes to C00 (negative) and C01
    
    # M6 = (c-a)(e+f)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1  # M6 contributes to C11
    
    # M7 = (b-d)(g+h)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1  # M7 contributes to C00
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Returns the schoolbook decomposition for n x m times m x p.
    U shape: (n*m, n*m*p), V shape: (m*p, n*m*p), W shape: (p*n, n*m*p)
    """
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
    return U, V, W

def _combine_decompositions(decs):
    """
    Combine multiple tensor decompositions via direct sum.
    Each dec is a tuple (U, V, W) where U is (n_i*m_i, r_i), etc.
    Returns (U, V, W) for the direct sum.
    """
    total_rank = sum(U.shape[1] for U, V, W in decs)
    # Determine total sizes
    total_nm = sum(U.shape[0] for U, V, W in decs)
    total_mp = sum(V.shape[0] for U, V, W in decs)
    total_pn = sum(W.shape[0] for U, V, W in decs)
    
    U_out = np.zeros((total_nm, total_rank))
    V_out = np.zeros((total_mp, total_rank))
    W_out = np.zeros((total_pn, total_rank))
    
    nm_offset = 0
    mp_offset = 0
    pn_offset = 0
    rank_offset = 0
    for U, V, W in decs:
        n_i, m_i = U.shape[0] // max(1, V.shape[0] // max(1, W.shape[0] // 1))  # tricky
        # Actually, let's just track offsets properly
        U_out[nm_offset:nm_offset+U.shape[0], rank_offset:rank_offset+U.shape[1]] = U
        V_out[mp_offset:mp_offset+V.shape[0], rank_offset:rank_offset+V.shape[1]] = V
        W_out[pn_offset:pn_offset+W.shape[0], rank_offset:rank_offset+W.shape[1]] = W
        nm_offset += U.shape[0]
        mp_offset += V.shape[0]
        pn_offset += W.shape[0]
        rank_offset += U.shape[1]
    
    return U_out, V_out, W_out

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly represents the matrix multiplication tensor.
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] should equal 1 if a=i*m+j, b=j*p+k, c=k*n+i
    and 0 otherwise.
    """
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute tensor from decomposition
    # Use einsum for efficiency
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    error = np.max(np.abs(T_computed - T_target))
    return error < tol, error

def _strassen_4x4x4():
    """
    Build a decomposition for 4x4 x 4x4 using Strassen recursively.
    4x4x4 can be done with 4 blocks of 2x2x2, each using Strassen rank 7.
    Total: 4 * 7 = 28 rank terms? No, that's not right.
    
    Actually, to multiply two 4x4 matrices, we split into 2x2 blocks:
    A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    Each Cij is a 2x2 matrix computed from 2x2 block multiplications.
    Using Strassen for the 4x4x4: we need 7 products of 2x2 matrices,
    each requiring rank 7, so total rank = 7 * 7 = 49.
    
    Let me implement this properly.
    """
    # For 4x4 x 4x4 using Strassen:
    # Split A into 4 2x2 blocks, B into 4 2x2 blocks
    # A indices in U: 0..3 for A00, 4..7 for A01, 8..11 for A10, 12..15 for A11
    # B indices in V: 0..3 for B00, 4..7 for B01, 8..11 for B10, 12..15 for B11
    # C indices in W: 0..3 for C00, 4..7 for C01, 8..11 for C10, 12..15 for C11
    
    # Strassen for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each Mj is a 2x2 matrix product, which we compute with Strassen rank 7.
    # So we need 7 * 7 = 49 rank terms total.
    
    S_U, S_V, S_W = _strassen_2x2x2()  # Each is (4,7)
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Block offsets for 4x4 matrices
    # A: A00=[0:4], A01=[4:8], A10=[8:12], A11=[12:16]
    # B: B00=[0:4], B01=[4:8], B10=[8:12], B11=[12:16]
    # C: C00=[0:4], C01=[4:8], C10=[8:12], C11=[12:16]
    
    # For each Strassen product Mj, we need to:
    # 1. Form the appropriate linear combinations of A and B blocks
    # 2. Apply Strassen 2x2x2 to compute the 2x2 product
    # 3. Add the result to the appropriate C blocks
    
    # M1 = (A00+A11)(B00+B11)
    # U contribution: A00 and A11 with +1
    # V contribution: B00 and B11 with +1
    # W contribution: C00 and C11 with +1
    
    def add_strassen_block(U, V, W, base_rank, 
                           a_blocks, b_blocks, c_blocks,
                           a_scales, b_scales, c_scales):
        """
        Add a Strassen 2x2x2 block to the decomposition.
        a_blocks: list of block indices for A (each 0-3)
        b_blocks: list of block indices for B
        c_blocks: list of block indices for C
        a_scales: list of scales for each A block
        b_scales: list of scales for each B block
        c_scales: list of scales for each C block
        """
        S_U, S_V, S_W = _strassen_2x2x2()
        for s in range(7):
            r = base_rank + s
            # U: sum over a_blocks
            for bi, scale in zip(a_blocks, a_scales):
                U[bi*4:(bi+1)*4, r] += scale * S_U[:, s]
            # V: sum over b_blocks
            for bi, scale in zip(b_blocks, b_scales):
                V[bi*4:(bi+1)*4, r] += scale * S_V[:, s]
            # W: sum over c_blocks
            for bi, scale in zip(c_blocks, c_scales):
                W[bi*4:(bi+1)*4, r] += scale * S_W[:, s]
    
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    add_strassen_block(U, V, W, 0, [0, 3], [0, 3], [0, 3],
                       [1, 1], [1, 1], [1, 1])
    
    # M2 = (A10+A11)B00 -> contributes to C10
    add_strassen_block(U, V, W, 7, [2, 3], [0], [2],
                       [1, 1], [1], [1])
    
    # M3 = A00(B01-B11) -> contributes to C01
    add_strassen_block(U, V, W, 14, [0], [1, 3], [1],
                       [1], [1, -1], [1])
    
    # M4 = A11(B10-B00) -> contributes to C00 and C10
    add_strassen_block(U, V, W, 21, [3], [2, 0], [0, 2],
                       [1], [1, -1], [1, 1])
    
    # M5 = (A00+A01)B11 -> contributes to C00 (neg) and C01
    add_strassen_block(U, V, W, 28, [0, 1], [3], [0, 1],
                       [1, 1], [1], [-1, 1])
    
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    add_strassen_block(U, V, W, 35, [2, 0], [0, 1], [3],
                       [1, -1], [1, 1], [1])
    
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    add_strassen_block(U, V, W, 42, [1, 3], [2, 3], [0],
                       [1, -1], [1, 1], [1])
    
    return U, V, W

def construct():
    """
    Returns a low-rank exact tensor decomposition for 4x4 x 4x7 matrix multiplication.
    
    Strategy: Decompose M_{4,4,7} = M_{4,4,4} ⊕ M_{4,4,3}
    - M_{4,4,4}: Use Strassen-based decomposition (rank 49)
    - M_{4,4,3}: Use schoolbook (rank 48) or try to improve
    
    Total rank: 49 + 48 = 97
    
    We can potentially do better for M_{4,4,3} by decomposing it as
    M_{4,4,3} = M_{4,4,2} ⊕ M_{4,4,1}
    M_{4,4,2}: Can use 2 Strassen 2x2x2 blocks = rank 14? No...
    Actually M_{4,4,2} schoolbook is 32. Can we do better?
    M_{4,4,1} schoolbook is 16.
    So 32 + 16 = 48, same as before.
    
    Alternative: M_{4,4,3} = M_{2,4,3} ⊕ M_{2,4,3}
    M_{2,4,3} schoolbook = 24. Two of them = 48. Same.
    
    Let me try: M_{4,4,3} using block structure.
    Split A into two 2x4 matrices, B into 4x3.
    Or split differently.
    
    For now, let's use 49 + 48 = 97 and verify.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 7
    
    # Part 1: M_{4,4,4} with rank 49
    U1, V1, W1 = _strassen_4x4x4()
    
    # Part 2: M_{4,4,3} with schoolbook rank 48
    U2, V2, W2 = _schoolbook_nm_p(4, 4, 3)
    
    # Combine via direct sum
    total_rank = U1.shape[1] + U2.shape[1]
    U = np.zeros((16, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((28, total_rank))
    
    # M_{4,4,4} occupies first 16 rows of U, first 16 rows of V, first 16 rows of W
    U[0:16, 0:U1.shape[1]] = U1
    V[0:16, 0:V1.shape[1]] = V1
    W[0:16, 0:W1.shape[1]] = W1
    
    # M_{4,4,3} occupies last 16 rows of U, rows 16:28 of V, rows 16:28 of W
    U[0:16, U1.shape[1]:] = U2
    V[16:28, U1.shape[1]:] = V2
    W[16:28, U1.shape[1]:] = W2
    
    # Verify
    valid, error = _verify_decomposition(U, V, W, 4, 4, 7)
    
    if not valid:
        # Fallback to schoolbook
        rank = 4 * 4 * 7
        U = np.zeros((16, rank))
        V = np.zeros((28, rank))
        W = np.zeros((28, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i*4+j, col] = 1
                    V[j*7+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Round to half-integers if close
    def round_half(x):
        x_rounded = np.round(x * 2) / 2.0
        if np.max(np.abs(x - x_rounded)) < 1e-9:
            return x_rounded
        return x
    
    U = round_half(U)
    V = round_half(V)
    W = round_half(W)
    
    # Final verification after rounding
    valid, error = _verify_decomposition(U, V, W, 4, 4, 7)
    if not valid:
        # Use unrounded if rounding broke it
        U, V, W = _strassen_4x4x4()
        U2, V2, W2 = _schoolbook_nm_p(4, 4, 3)
        total_rank = U.shape[1] + U2.shape[1]
        U = np.zeros((16, total_rank))
        V = np.zeros((28, total_rank))
        W = np.zeros((28, total_rank))
        U[0:16, 0:U1.shape[1]] = U1
        V[0:16, 0:V1.shape[1]] = V1
        W[0:16, 0:W1.shape[1]] = W1
        U[0:16, U1.shape[1]:] = U2
        V[16:28, U1.shape[1]:] = V2
        W[16:28, U1.shape[1]:] = W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
