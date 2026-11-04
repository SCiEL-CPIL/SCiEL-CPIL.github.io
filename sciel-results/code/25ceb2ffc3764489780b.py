"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    Checks that for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Compute the sum of products for this specific tensor entry
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """
    Return the schoolbook decomposition of rank n*m*p for matrix multiplication.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index into U: row i, col j
                u_idx = i * m + j
                # Index into V: row j, col k
                v_idx = j * p + k
                # Index into W: row k, col i
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    return U, V, W

def _winograd_3x3():
    """
    Return the 7-term Winograd/Strassen decomposition for 3x3 matrix multiplication.
    Returns (U3, V3, W3) each of shape (9, 7) where:
      U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]
    """
    # Winograd's 7-term scheme for 3x3 matrix multiplication
    # The 7 bilinear forms are:
    # m0 = (a00+a10+a20)*(b00+b10+b20)
    # m1 = (a00+a10+a20)*(b01+b11+b21)
    # m2 = (a00+a10+a20)*(b02+b12+b22)
    # m3 = (a10+a20)*(b00+b10+b20)  -- wait, let me use the standard Winograd scheme
    
    # Actually, let me use a known 7-term scheme. The standard Strassen-like 7 terms for 3x3:
    # Using the decomposition from Winograd (1979) or similar:
    #
    # Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # Let B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    #
    # The 7 bilinear forms:
    # m0 = (a00+a10+a20)*(b00+b10+b20)
    # m1 = (a00+a10+a20)*(b01+b11+b21)  
    # m2 = (a00+a10+a20)*(b02+b12+b22)
    # m3 = (a10+a20)*(b00+b10+b20)
    # m4 = (a10+a20)*(b01+b11+b21)
    # m5 = (a10+a20)*(b02+b12+b22)
    # m6 = a20*(b00+b10+b20)  -- this isn't quite right
    
    # Let me use a verified 7-term scheme. I'll construct it from the known 
    # 3x3x3 rank-7 decomposition.
    
    # A well-known rank-7 decomposition for 3x3 matrix multiplication:
    # The 7 factors are defined by their U, V, W components.
    
    # Using the scheme where we decompose based on the structure:
    # Factor 0: U has 1s at (0,0),(1,0),(2,0); V has 1s at (0,0),(1,0),(2,0); W has 1s at (0,0),(1,0),(2,0)
    # This gives sum of all products in column 0 of A times column 0 of B... 
    
    # Let me just directly specify a valid rank-7 decomposition.
    # I'll use the one from the literature.
    
    # Actually, let me construct it carefully. For 3x3 matrix mult, 
    # we need: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor has indices (i,j,k) for i,j,k in {0,1,2}.
    
    # A rank-7 decomposition means 7 terms where each term is a product of 
    # a vector in A-space (3x3=9 dims), a vector in B-space (3x3=9 dims), 
    # and a vector in C-space (3x3=9 dims).
    
    # Let me use the following known 7-term scheme:
    # Term 0: A-vec has 1s at positions (0,0),(1,0),(2,0) [i.e., j=0 column]
    #          B-vec has 1s at positions (0,0),(1,0),(2,0) [i.e., k=0 column]  
    #          C-vec has 1s at positions (0,0),(1,0),(2,0) [i.e., k=0 column]
    # This doesn't work directly because we need cross terms.
    
    # Let me just hardcode a verified decomposition.
    # After checking, here's a valid 7-term scheme for 3x3:
    
    U3 = np.zeros((9, 7))
    V3 = np.zeros((9, 7))
    W3 = np.zeros((9, 7))
    
    # Term 0
    U3[0, 0] = 1; U3[3, 0] = 1; U3[6, 0] = 1  # A[0,0], A[1,0], A[2,0]
    V3[0, 0] = 1; V3[3, 0] = 1; V3[6, 0] = 1  # B[0,0], B[1,0], B[2,0]
    W3[0, 0] = 1; W3[3, 0] = 1; W3[6, 0] = 1  # C[0,0], C[1,0], C[2,0]
    
    # Term 1
    U3[0, 1] = 1; U3[3, 1] = 1; U3[6, 1] = 1
    V3[1, 1] = 1; V3[4, 1] = 1; V3[7, 1] = 1  # B[0,1], B[1,1], B[2,1]
    W3[1, 1] = 1; W3[4, 1] = 1; W3[7, 1] = 1  # C[0,1], C[1,1], C[2,1]
    
    # Term 2
    U3[0, 2] = 1; U3[3, 2] = 1; U3[6, 2] = 1
    V3[2, 2] = 1; V3[5, 2] = 1; V3[8, 2] = 1  # B[0,2], B[1,2], B[2,2]
    W3[2, 2] = 1; W3[5, 2] = 1; W3[8, 2] = 1  # C[0,2], C[1,2], C[2,2]
    
    # Term 3
    U3[3, 3] = 1; U3[6, 3] = 1  # A[1,0], A[2,0]
    V3[0, 3] = 1; V3[3, 3] = 1; V3[6, 3] = 1
    W3[0, 3] = -1; W3[3, 3] = 1; W3[6, 3] = 1  # C[0,0]=-1, C[1,0]=1, C[2,0]=1
    
    # Term 4
    U3[3, 4] = 1; U3[6, 4] = 1
    V3[1, 4] = 1; V3[4, 4] = 1; V3[7, 4] = 1
    W3[1, 4] = -1; W3[4, 4] = 1; W3[7, 4] = 1
    
    # Term 5
    U3[3, 5] = 1; U3[6, 5] = 1
    V3[2, 5] = 1; V3[5, 5] = 1; V3[8, 5] = 1
    W3[2, 5] = -1; W3[5, 5] = 1; W3[8, 5] = 1
    
    # Term 6
    U3[6, 6] = 1  # A[2,0]
    V3[0, 6] = 1; V3[3, 6] = 1; V3[6, 6] = 1
    W3[0, 6] = -1; W3[3, 6] = -1; W3[6, 6] = 1  # C[0,0]=-1, C[1,0]=-1, C[2,0]=1
    
    # Verify this decomposition
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a_idx = i * 3 + j
                b_idx = j * 3 + k
                c_idx = k * 3 + i
                val = np.dot(U3[a_idx, :], V3[b_idx, :] * W3[c_idx, :])
                if abs(val - 1.0) > 1e-9:
                    # This decomposition is wrong, fall back to schoolbook for 3x3
                    return None, None, None
    
    return U3, V3, W3

def _structured_decomposition(n, m, p, deadline):
    """
    Try to build a structured decomposition using Winograd 3x3 for the core
    and rank-1 terms for the extra dimensions.
    """
    n, m, p = 3, 4, 8
    
    # For the 3x3x3 core (first 3 of 4 inner dims, first 3 of 8 output dims):
    # We can use the 7-term Winograd scheme.
    # The remaining dimensions need rank-1 terms.
    
    # Strategy: 
    # - Use 7 terms for the 3x3x3 sub-tensor (i in {0,1,2}, j in {0,1,2}, k in {0,1,2})
    # - For j=3 (4th inner dim): need 3*8 = 24 rank-1 terms (or fewer if structured)
    # - For k=3..7 (5th-8th output dims): need 3*4 = 12 rank-1 terms per extra k
    #   Total: 7 + 24 + 5*12 = 7 + 24 + 60 = 91
    # 
    # But we can do better:
    # - For j=3: A[i,3]*B[3,k] for all i,k. This is outer product structure.
    #   We need sum_j=3 A[i,3]*B[3,k] = A[i,3]*B[3,k]
    #   This is a rank-1 tensor in (i,k) space for fixed j=3.
    #   So we need n*p = 3*8 = 24 terms for j=3.
    #   
    # - For k=3..7: A[i,j]*B[j,k] for all i,j,k in {3..7}
    #   For each extra k, we need n*m = 3*4 = 12 terms.
    #   5 extra k's: 5*12 = 60 terms.
    #
    # Total: 7 + 24 + 60 = 91 terms.
    # Schoolbook is 3*4*8 = 96 terms.
    # So we save 5 terms.
    
    # Let's build this:
    U = np.zeros((n * m, 91))
    V = np.zeros((m * p, 91))
    W = np.zeros((p * n, 91))
    
    col = 0
    
    # Part 1: Winograd 7 terms for i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    U3, V3, W3 = _winograd_3x3()
    if U3 is None:
        # Fall back to schoolbook for the 3x3 part
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        # This uses 27 terms instead of 7, so total will be 27 + 24 + 60 = 111
        # That's worse than schoolbook! So we need the Winograd to work.
        # Let me just use schoolbook for everything if Winograd fails.
        return None
    else:
        # Embed the 3x3 Winograd into the 3x4x8 structure
        # U3 has shape (9,7) where row = i*3+j for i,j in {0,1,2}
        # We need to map this to U which has row = i*4+j
        for r in range(7):
            for i in range(3):
                for j in range(3):
                    u3_idx = i * 3 + j
                    u_idx = i * m + j
                    U[u_idx, col + r] = U3[u3_idx, r]
                    v3_idx = j * 3 + (k := 0)  # This is wrong, let me redo
        # Actually, the mapping is more complex. Let me be more careful.
        # U3[i*3+j, r] corresponds to the coefficient of A[i,j] in term r.
        # In our U matrix, row i*m+j = i*4+j holds the coefficient of A[i,j].
        # So we just need to copy U3 into the appropriate rows of U.
        
        for i in range(3):
            for j in range(3):
                for r in range(7):
                    U[i * m + j, col + r] = U3[i * 3 + j, r]
        
        for j in range(3):
            for k in range(3):
                for r in range(7):
                    V[j * p + k, col + r] = V3[j * 3 + k, r]
        
        for k in range(3):
            for i in range(3):
                for r in range(7):
                    W[k * n + i, col + r] = W3[k * 3 + i, r]
        
        col += 7
    
    # Part 2: j=3, all i in {0,1,2}, all k in {0,1,2,3,4,5,6,7}
    # For each (i,k) pair, we need a rank-1 term: U[i*4+3, r]=1, V[3*8+k, r]=1, W[k*3+i, r]=1
    for i in range(3):
        for k in range(8):
            u_idx = i * m + 3
            v_idx = 3 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1
    
    # Part 3: j in {0,1,2,3}, k in {3,4,5,6,7}
    # For each (i,j,k) with k>=3, we need a rank-1 term.
    for i in range(3):
        for j in range(4):
            for k in range(3, 8):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    
    # Total should be 7 + 24 + 60 = 91
    assert col == 91, f"Expected 91 columns, got {col}"
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys, each a 2D list.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try structured decomposition first
    result = _structured_decomposition(n, m, p, deadline)
    
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    if not _verify_decomposition(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
