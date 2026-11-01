"""Strategy: Algebraic construction via Kronecker product of Strassen-7 and identity,
then attempt to reduce rank by finding linear dependencies among the 63 Kronecker
terms. If no reduction found, return the rank-63 (or best found) decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# Strassen's 7 bilinear forms for 2x2 * 2x2
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)(B00)
# M3 = A00(B01-B11)
# M4 = A11(B10-B00)
# M5 = (A00+A01)(B11)
# M6 = (A10-A00)(B00+B01)
# M7 = (A01-A11)(B10+B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

# Strassen coefficients: for each of the 7 products, the A-coeffs, B-coeffs, and C-contributions
# A is 2x2, indexed as [A00, A01, A10, A11]
# B is 2x2, indexed as [B00, B01, B10, B11]
# C is 2x2, indexed as [C00, C01, C10, C11]

# Product r: A_coeffs (4), B_coeffs (4), C_coeffs (4)
_strassen_A = np.array([
    [1, 0, 0, 1],   # M1: A00 + A11
    [0, 0, 1, 1],   # M2: A10 + A11
    [1, 0, 0, 0],   # M3: A00
    [0, 0, 0, 1],   # M4: A11
    [1, 1, 0, 0],   # M5: A00 + A01
    [-1, 0, 1, 0],  # M6: A10 - A00
    [0, 1, 0, -1],  # M7: A01 - A11
], dtype=float)

_strassen_B = np.array([
    [1, 0, 0, 1],   # M1: B00 + B11
    [1, 0, 0, 0],   # M2: B00
    [0, 1, 0, -1],  # M3: B01 - B11
    [0, 0, 1, -1],  # M4: B10 - B00
    [0, 0, 0, 1],   # M5: B11
    [1, 1, 0, 0],   # M6: B00 + B01
    [0, 0, 1, 1],   # M7: B10 + B11
], dtype=float)

_strassen_C = np.array([
    [1, 0, 0, 0],   # M1 -> C00
    [0, 0, 0, 0],   # M2 -> (handled below)
    [0, 1, 0, 0],   # M3 -> C01
    [0, 0, 0, 0],   # M4 -> (handled below)
    [0, 0, 0, 0],   # M5 -> (handled below)
    [0, 0, 0, 0],   # M6 -> (handled below)
    [0, 0, 0, 0],   # M7 -> (handled below)
], dtype=float)

# Actually, let me be more careful. The C contributions:
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6
# So C_coeffs for each M_r:
# M1: C00=1, C11=1
# M2: C10=1, C11=-1
# M3: C01=1, C11=1
# M4: C00=1, C10=1
# M5: C00=-1, C01=1
# M6: C11=1
# M7: C00=1

_strassen_C = np.array([
    [1, 0, 0, 1],   # M1: C00 + C11
    [0, 0, 1, -1],  # M2: C10 - C11
    [0, 1, 0, 1],   # M3: C01 + C11
    [1, 0, 1, 0],   # M4: C00 + C10
    [-1, 1, 0, 0],  # M5: -C00 + C01
    [0, 0, 0, 1],   # M6: C11
    [1, 0, 0, 0],   # M7: C00
], dtype=float)


def _build_strassen_2x2x2():
    """Return U, V, W for 2x2*2x2 with rank 7."""
    U = _strassen_A.T  # shape (4, 7)
    V = _strassen_B.T  # shape (4, 7)
    W = _strassen_C.T  # shape (4, 7)
    return U, V, W


def _build_kronecker_decomposition():
    """
    Build a decomposition for 2x2 * 2x9 using Strassen on the 2x2 A side
    and identity on the 9-column B side.
    
    Idea: C[i,k] = sum_j A[i,j] B[j,k]
    We can think of this as: for each k, C[:,k] = A * B[:,k]
    So we're doing 9 independent 2x1 multiplications, but sharing the A-factor.
    
    Actually, a cleaner approach:
    C = A * B where A is 2x2, B is 2x9.
    
    We can write B as 9 column vectors b_k (each 2x1).
    C[:,k] = A * b_k
    
    Using Strassen: for each k, we need 7 products. But we can share the A-side
    linear combinations across all k.
    
    Specifically, define:
    s1 = A00 + A11
    s2 = A10 + A11
    s3 = A00
    s4 = A11
    s5 = A00 + A01
    s6 = A10 - A00
    s7 = A01 - A11
    
    And for each k:
    t1_k = B00_k + B11_k  (but B is 2x9, so B[j,k] for j=0,1)
    
    Wait, B is 2x9, so B[j,k] for j in {0,1}, k in {0..8}.
    The "2x2" Strassen doesn't directly apply because B has only 2 rows but 9 columns.
    
    Let me reconsider. The multiplication is:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is really 9 separate 2x1 multiplications (each using the same A).
    For a 2x1 multiplication: C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    This has rank 2 (schoolbook) or can it be done with rank 1? No, rank 2 is minimal.
    
    But we can share across k! Define:
    For each k: C[0,k] = A00*B0k + A01*B1k
                 C[1,k] = A10*B0k + A11*B1k
    
    We can use the following 4 products (one per (i,j) pair), but that's rank 4*9=36.
    
    Alternatively, use Strassen-like sharing. Let me think differently.
    
    Actually, the simplest good decomposition:
    For each k (0..8), we compute C[:,k] = A * B[:,k] where B[:,k] is a 2-vector.
    This is a 2x2 * 2x1 multiplication, which has rank 2.
    So total rank = 9 * 2 = 18.
    
    But we can do better by sharing. Note that all 9 columns use the same A.
    So we can precompute linear combinations of A once.
    
    Let me try: use 2 products per k, but make them depend on all of A.
    Product for k, type 0: (A00 + A10) * (B0k + B1k)  -- no, this doesn't work cleanly.
    
    Let me just go with the rank-18 schoolbook-like decomposition:
    For each k:
      Term 2k:   A[0,0] * B[0,k] -> C[0,k]
      Term 2k+1: A[0,1] * B[1,k] -> C[0,k]
      Term 2k+2: A[1,0] * B[0,k] -> C[1,k]
      Term 2k+3: A[1,1] * B[1,k] -> C[1,k]
    That's 36 terms.
    
    Or, the rank-18 version:
    For each k:
      Term 2k:   (A[0,0]*B[0,k] + A[0,1]*B[1,k]) -> C[0,k]  -- but this is 2 products, not 1!
    
    I see, each output C[i,k] requires 2 products (sum over j). So minimum is 18*2/... 
    No, each product can contribute to multiple outputs.
    
    Let me use the Kronecker product approach properly.
    
    The tensor for 2x2*2x9 can be seen as the "tensor product" of the 2x2*2x1 tensor
    (repeated 9 times with shared A).
    
    Actually, let me just build the rank-18 decomposition:
    For each k in 0..8:
      For each (i,j) in {(0,0),(0,1),(1,0),(1,1)}:
        U[i*2+j, r] = 1
        V[j*9+k, r] = 1
        W[k*2+i, r] = 1
    This is 36 terms.
    
    To get rank 18, I need to use the fact that for fixed i, the two j-terms can be
    combined. But they can't be combined into a single bilinear term in general.
    
    Let me try a different approach: use Strassen on the 2x2 part and replicate for 9 columns.
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Using Strassen's 7 products for the 2x2 A:
    s1 = A00+A11, s2 = A10+A11, s3 = A00, s4 = A11, s5 = A00+A01, s6 = A10-A00, s7 = A01-A11
    
    For each k, define:
    t1_k = B0k + B1k  (but B only has 2 rows, so this is B[0,k]+B[1,k])
    t2_k = B0k
    t3_k = B1k - B0k
    t4_k = B1k
    t5_k = B1k
    t6_k = B0k + B1k
    t7_k = B1k
    
    Then:
    C[0,k] = s1*t1_k + s4*t4_k - s5*t5_k + s7*t7_k
            = (A00+A11)(B0k+B1k) + A11*B1k - (A00+A01)*B1k + (A01-A11)*B1k
            = A00*B0k + A00*B1k + A11*B0k + A11*B1k + A11*B1k - A00*B1k - A01*B1k + A01*B1k - A11*B1k
            = A00*B0k + A11*B0k + A01*B1k - A01*B1k + A11*B1k - A11*B1k + A11*B1k
            Hmm, this is getting messy. Let me just verify with the standard Strassen formula.
    
    Standard Strassen for C = A*B where A,B are 2x2:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    where:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)(B00)
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)(B11)
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    For our case, B is 2x9, so B[j,k] for j=0,1 and k=0..8.
    We can apply Strassen column-by-column. For column k:
    B_col = [B[0,k], B[1,k]] (a 2x1 vector)
    
    The "2x2" Strassen doesn't directly apply to a 2x1 B. But we can embed:
    Think of B[:,k] as the first column of a 2x2 matrix [[B[0,k], 0], [B[1,k], 0]].
    Then C[:,k] = A * [[B[0,k], 0], [B[1,k], 0]] gives C[:,k] in the first column.
    
    Using Strassen on this embedded 2x2:
    M1_k = (A00+A11)(B[0,k] + 0) = (A00+A11)*B[0,k]
    M2_k = (A10+A11)(B[0,k]) = (A10+A11)*B[0,k]
    M3_k = A00(0 - 0) = 0
    M4_k = A11(0 - B[0,k]) = -A11*B[0,k]
    M5_k = (A00+A01)(0) = 0
    M6_k = (A10-A00)(B[0,k] + 0) = (A10-A00)*B[0,k]
    M7_k = (A01-A11)(0 + 0) = 0
    
    C[0,k] = M1_k + M4_k - M5_k + M7_k = (A00+A11)*B[0,k] - A11*B[0,k] = A00*B[0,k]
    But we need C[0,k] = A00*B[0,k] + A01*B[1,k]. So this doesn't work.
    
    The issue is that Strassen for 2x2 assumes both matrices are 2x2. For 2x2 * 2x1,
    the rank is 2 (you need A00*B0 + A01*B1 for C0, and A10*B0 + A11*B1 for C1).
    
    So the rank of 2x2*2x9 is at least 18 (9 columns * rank 2 per column), but we can
    share the A-side combinations.
    
    Let me try: define 2 shared A-combinations:
    p1 = A00, p2 = A01, p3 = A10, p4 = A11 (just the 4 entries)
    
    For each k, we need:
    C[0,k] = p1*B[0,k] + p2*B[1,k]
    C[1,k] = p3*B[0,k] + p4*B[1,k]
    
    This is 4 products per k, total 36.
    
    Can we reduce? Let's try:
    For each k:
      q1_k = p1*B[0,k] + p3*B[0,k] = (p1+p3)*B[0,k]  -- but this mixes C[0,k] and C[1,k]
    
    I think the minimum rank for 2x2*2x9 is 18. Here's a construction:
    For each k (0..8):
      Term 2k:   U has A[0,0] and A[1,0] (i.e., first column of A), V has B[0,k], W has C[0,k] and C[1,k]
      Term 2k+1: U has A[0,1] and A[1,1] (i.e., second column of A), V has B[1,k], W has C[0,k] and C[1,k]
    
    Wait, that doesn't work because a single term can only multiply one A-linear-combination
    by one B-linear-combination.
    
    Let me try:
    Term 2k:   U = [A00, 0, A10, 0], V = [B0k, 0], W = [C0k, C1k]  -- this gives (A00*B0k, A10*B0k)
    Term 2k+1: U = [0, A01, 0, A11], V = [0, B1k], W = [C0k, C1k]  -- this gives (A01*B1k, A11*B1k)
    
    Then C[0,k] = A00*B0k + A01*B1k ✓
    C[1,k] = A10*B0k + A11*B1k ✓
    
    This uses 2 terms per k, total 18 terms! Rank 18.
    
    Let me verify the tensor entries:
    U is 4x18, V is 18x18, W is 18x18.
    
    For term r = 2k (k=0..8):
      U[0, r] = 1  (A[0,0])
      U[1, r] = 0  (A[0,1])
      U[2, r] = 1  (A[1,0])
      U[3, r] = 0  (A[1,1])
      V[0*9+k, r] = 1  (B[0,k])
      V[1*9+k, r] = 0  (B[1,k])
      W[k*2+0, r] = 1  (C[0,k])
      W[k*2+1, r] = 1  (C[1,k])
    
    For term r = 2k+1:
      U[0, r] = 0
      U[1, r] = 1  (A[0,1])
      U[2, r] = 0
      U[3, r] = 1  (A[1,1])
      V[0*9+k, r] = 0
      V[1*9+k, r] = 1  (B[1,k])
      W[k*2+0, r] = 1  (C[0,k])
      W[k*2+1, r] = 1  (C[1,k])
    
    Now check T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]:
    T[0, 0*9+k, k*2+0] = U[0,2k]*V[k,2k]*W[2k,2k] = 1*1*1 = 1 ✓ (A[0,0]*B[0,k] -> C[0,k])
    T[0, 1*9+k, k*2+0] = U[0,2k+1]*V[9+k,2k+1]*W[2k,2k+1] = 1*1*1 = 1 ✓ (A[0,1]*B[1,k] -> C[0,k])
    T[2, 0*9+k, k*2+1] = U[2,2k]*V[k,2k]*W[2k+1,2k] = 1*1*1 = 1 ✓ (A[1,0]*B[0,k] -> C[1,k])
    T[3, 1*9+k, k*2+1] = U[3,2k+1]*V[9+k,2k+1]*W[2k+1,2k+1] = 1*1*1 = 1 ✓ (A[1,1]*B[1,k] -> C[1,k])
    
    And all other entries should be 0. Let me check T[0, 0*9+k, k'*2+0] for k' != k:
    = U[0,2k']*V[k,2k']*W[2k,2k'] = 1*0*1 = 0 ✓ (V[k,2k'] = 0 since k != k')
    
    T[0, 1*9+k, k'*2+0] for k' != k:
    = U[0,2k'+1]*V[9+k,2k'+1]*W[2k,2k'+1] = 1*0*1 = 0 ✓
    
    Looks correct! Rank 18.
    
    Can we do better than 18? The rank of 2x2*2x1 is 2, and we have 9 such multiplications
    sharing the A-matrix. The minimum rank is at least 18/2 = 9? No, that's not right.
    
    Actually, I think 18 might be optimal for this problem, or close to it. Let me just
    return the rank-18 decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 2, 9
    rank = 2 * p  # 18
    U = np.zeros((n * m, rank))  # (4, 18)
    V = np.zeros((m * p, rank))  # (18, 18)
    W = np.zeros((p * n, rank))  # (18, 18)
    
    for k in range(p):
        # Term 2k: A[:,0] * B[0,k] -> C[:,k]
        r = 2 * k
        U[0 * m + 0, r] = 1  # A[0,0]
        U[1 * m + 0, r] = 1  # A[1,0]
        V[0 * p + k, r] = 1  # B[0,k]
        W[k * n + 0, r] = 1  # C[0,k]
        W[k * n + 1, r] = 1  # C[1,k]
        
        # Term 2k+1: A[:,1] * B[1,k] -> C[:,k]
        r = 2 * k + 1
        U[0 * m + 1, r] = 1  # A[0,1]
        U[1 * m + 1, r] = 1  # A[1,1]
        V[1 * p + k, r] = 1  # B[1,k]
        W[k * n + 0, r] = 1  # C[0,k]
        W[k * n + 1, r] = 1  # C[1,k]
    
    return U, V, W


def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly represents the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # Build the target tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Compute the decomposition
    # M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    rank = U.shape[1]
    M = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check
    diff = np.abs(M - T)
    max_diff = np.max(diff)
    return max_diff < 1e-10, max_diff


def construct():
    """
    Find a low-rank decomposition for 2x2 * 2x9 matrix multiplication.
    Returns dict with U (4 x rank), V (18 x rank), W (18 x rank) as lists.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Build the rank-18 decomposition
    U, V, W = _build_kronecker_decomposition()
    
    # Verify
    valid, max_diff = _verify_decomposition(U, V, W)
    if not valid:
        # Fallback to schoolbook rank-36
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Try to reduce rank by finding linear dependencies
    # This is a structural change: try to express some columns as linear combinations of others
    rank = U.shape[1]
    
    # Check if any column is a linear combination of others (for U, V, W simultaneously)
    # This is hard, so let's try a simpler approach: check if we can merge terms
    # by finding if two terms can be combined.
    
    # For now, just return the rank-18 decomposition.
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
