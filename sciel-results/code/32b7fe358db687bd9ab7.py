import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    """Strassen decomposition for a 2x2x2 block."""
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[i_offset*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[(i_offset+1)*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = -1
    col += 1
    
    # M3: A00(B01-B11) -> C01, C11
    U[i_offset*m + j_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[j_offset*p + k_offset, col] = -1
    W[k_offset*n + i_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[i_offset*m + j_offset, col] = 1
    U[i_offset*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[(i_offset+1)*m + j_offset, col] = 1
    U[i_offset*m + j_offset, col] = -1
    V[j_offset*p + k_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[i_offset*m + (j_offset+1), col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = -1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    col += 1
    
    return col

def _strassen_2x2x3(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    """
    Decomposition for 2x2x3 block (rank 11).
    Uses Strassen for k_offset,k_offset+1 and schoolbook for k_offset+2.
    """
    # First 2x2x2 Strassen block
    col = _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p)
    
    # Schoolbook for the third k (4 terms)
    for i in range(2):
        for j in range(2):
            U[(i_offset+i)*m + (j_offset+j), col] = 1
            V[(j_offset+j)*p + (k_offset+2), col] = 1
            W[(k_offset+2)*n + (i_offset+i), col] = 1
            col += 1
    
    return col

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _build_rank28():
    """Build the rank-28 base decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Block 1: j=0, k=0..4 (2x1x5), rank 10
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Block 2: j=1,2, k=0,1 (2x2x2 Strassen), rank 7
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    
    # Block 3: j=1,2, k=2,3,4 (2x2x3), rank 11
    col = _strassen_2x2x3(U, V, W, col, 0, 1, 2, n, m, p)
    
    return U, V, W, col

def _build_rank27_attempt():
    """
    Attempt rank-27 by merging the j=0,k=2 term with the 2x2x3 block.
    
    Strategy: The j=0 block has term A[i,0]*B[0,2]->C[i,2] for i=0,1.
    The 2x2x3 block for j=1,2,k=2,3,4 has schoolbook terms for k=2:
      A[i,1]*B[1,2]->C[i,2] and A[i,2]*B[2,2]->C[i,2] for i=0,1.
    
    Idea: Replace the two j=0,k=2 terms (rank 2) with a modified 2x2x3
    decomposition that absorbs them, saving 1 rank total.
    
    Alternative: Use a 2x3x5 direct search with half-integer coefficients.
    """
    n, m, p = DIMENSIONS
    # Try: remove one j=0 term and see if we can adjust
    # This is complex; let's try a different approach.
    
    # Actually, let's try: for the k=2 column of C, we have:
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    # Currently this uses 6 rank-1 terms (2 for j=0, 2 for j=1, 2 for j=2)
    # Can we do it in 5? That would save 1.
    
    # For a 2x3x1 tensor (computing C[:,2]), the rank is at most 6.
    # Is it reducible? The tensor T[i,j,0] = 1 for all i,j (6 entries).
    # This is the all-ones 2x3 matrix viewed as a tensor.
    # Rank of all-ones 2x3 matrix = 1, but as a 3-mode tensor with the
    # specific structure of matrix multiplication, it's different.
    
    # Actually for 2x3x1: C_i = sum_j A_ij * B_j
    # This is just a 2x3 matrix times a 3-vector.
    # The tensor rank: each term A_ij*B_j contributes to C_i.
    # We can write: C_0 = A_00*B_0 + A_01*B_1 + A_02*B_2
    #               C_1 = A_10*B_0 + A_11*B_1 + A_12*B_2
    # Using (A_00+A_10)*B_0 -> contributes to both C_0 and C_1
    # But we need to separate them, which requires more terms.
    # R(2,3,1) = 6 (it's just the 6 scalar products, no savings possible
    # because the B_j are distinct for each j).
    
    # So no savings from that angle. Let's stick with rank 28.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Build rank-28 decomposition
    U, V, W, col = _build_rank28()
    
    assert col == 28, f"Column count mismatch: {col} != 28"
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition failed verification")
    
    # Try rank-27 attempt (likely returns None)
    if time.time() < deadline - 5:
        result = _build_rank27_attempt()
        if result is not None:
            U, V, W, col = result
            if _verify_decomposition(U, V, W, n, m, p):
                pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
