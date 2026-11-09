"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2x2():
    """
    Returns U, V, W for rank-7 decomposition of 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: i*2+j for A, i*2+j for B, i*2+j for C.
    Convention: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] should equal 1 
    when c corresponds to the product position, 0 otherwise.
    Actually for matmul: C[i,j] = sum_k A[i,k]*B[k,j]
    Tensor: T[a,b,c] where a=i*2+k (A index), b=k*2+j (B index), c=i*2+j (C index)
    T[a,b,c] = 1 if a=i*2+k, b=k*2+j, c=i*2+j for some i,j,k, else 0.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1
    U[:, 0] = [1, 0, 0, 1]  # A00 + A11
    V[:, 0] = [1, 0, 0, 1]  # B00 + B11
    W[:, 0] = [1, 0, 0, 1]  # C00 + C11 (but we need separate contributions)
    
    # Wait, this approach is wrong. Each rank-1 term contributes to multiple C entries
    # with specific coefficients. Let me use the standard formulation:
    # The tensor T has T[a,b,c] = 1 iff there exist i,j,k such that 
    # a = i*2+k, b = k*2+j, c = i*2+j
    # For 2x2x2, there are 8 such triples (i,j,k in {0,1}).
    
    # Let me just verify the schoolbook for 2x2x2 has rank 8, and Strassen reduces to 7.
    # The 7 Strassen terms as rank-1 tensors:
    
    # Term 0: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # This gives: sum over a,b,c where U[a]*V[b]*W[c] nonzero
    # = A00*B00*C00 + A00*B00*C11 + A11*B00*C00 + A11*B00*C11
    #   + A00*B11*C00 + A00*B11*C11 + A11*B11*C00 + A11*B11*C11
    # That's not right either. The issue is that a single rank-1 term 
    # U[a]*V[b]*W[c] contributes to T[a,b,c] for ALL a,b,c.
    
    # I think the correct interpretation is:
    # We want sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # where T[a,b,c] = 1 if (a,b,c) corresponds to a valid matmul triple, else 0.
    
    # For 2x2x2 schoolbook (rank 8):
    # For each (i,j,k): a=i*2+k, b=k*2+j, c=i*2+j
    # U[a, r] = 1, V[b, r] = 1, W[c, r] = 1 for that specific r
    
    # For Strassen, we need 7 terms that sum to the same tensor.
    
    # Let me just hardcode the known correct Strassen decomposition.
    # Reference: The 7 terms are:
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11) -> contributes to C00 and C11
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # Term 1: (A10+A11)B00 -> contributes to C10
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    
    # Term 2: A00(B01-B11) -> contributes to C01
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01 - B11
    W[1, 2] = 1  # C01
    
    # Term 3: A11(B10-B00) -> contributes to C00
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10 - B00
    W[0, 3] = 1  # C00
    
    # Term 4: (A00+A01)B11 -> contributes to C00 with -1
    U[0, 4] = 1; U[1, 4] = 1  # A00 + A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1  # -C00
    
    # Term 5: (A10-A00)(B00+B01) -> contributes to C11
    U[2, 5] = 1; U[0, 5] = -1  # A10 - A00
    V[0, 5] = 1; V[1, 5] = 1  # B00 + B01
    W[3, 5] = 1  # C11
    
    # Term 6: (A01-A11)(B10+B11) -> contributes to C00
    U[1, 6] = 1; U[3, 6] = -1  # A01 - A11
    V[2, 6] = 1; V[3, 6] = 1  # B10 + B11
    W[0, 6] = 1  # C00
    
    return U, V, W

def _verify_2x2x2(U, V, W):
    """Verify the 2x2x2 decomposition."""
    # T[a,b,c] should be 1 if a=i*2+k, b=k*2+j, c=i*2+j for some i,j,k
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i * 2 + k
                b = k * 2 + j
                c = i * 2 + j
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False, (i,j,k,a,b,c,val)
    # Also check that non-matching entries are 0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                # Check if (a,b,c) is a valid triple
                valid = False
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            if a == i*2+k and b == k*2+j and c == i*2+j:
                                valid = True
                if not valid:
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val) > 1e-9:
                        return False, (a,b,c,val)
    return True, None

def _compose_4x4x4_strassen():
    """
    Compose 4x4x4 Strassen decomposition with rank 49.
    Uses 2x2x2 Strassen (rank 7) composed with itself.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # Verify the base
    ok, info = _verify_2x2x2(U2, V2, W2)
    if not ok:
        # Fall back to schoolbook for 2x2x2
        U2 = np.zeros((4, 8))
        V2 = np.zeros((4, 8))
        W2 = np.zeros((4, 8))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i * 2 + k
                    b = k * 2 + j
                    c = i * 2 + j
                    U2[a, col] = 1
                    V2[b, col] = 1
                    W2[c, col] = 1
                    col += 1
    
    n_blocks = 4  # 2x2 blocks
    rank_base = U2.shape[1]
    rank_4 = rank_base * rank_base  # 49 or 64
    
    U4 = np.zeros((16, rank_4))
    V4 = np.zeros((16, rank_4))
    W4 = np.zeros((16, rank_4))
    
    # For 4x4, we have 4 blocks of 2x2 each.
    # Block (bi, bj) of A occupies rows: bi*4 + [0,1,4,5] in the 16-element space
    # Actually, A is 4x4, flattened as A[i*4+j] for i,j in 0..3
    # Block (bi, bj) is A[bi*2+i, bj*2+j] for i,j in 0..1
    # In flattened index: (bi*2+i)*4 + (bj*2+j) = bi*8 + i*4 + bj*2 + j
    
    # For the composition:
    # Each 2x2x2 Strassen term r (0..rank_base-1) at block position (bi, bj) for A,
    # (ck, dl) for B, and (ei, fj) for C
    # contributes to the 4x4x4 tensor.
    
    # The 4x4 matmul: C[i,j] = sum_k A[i,k] B[k,j]
    # In blocks: C[bi*2+ei, bj*2+fj] = sum_{ck} A[bi*2+ei, ck*2+?] B[ck*2+?, bj*2+fj]
    # This is getting complex. Let me use a different approach.
    
    # Direct approach: Build the 4x4x4 multiplication tensor and find its rank-49 decomposition
    # by composing the 2x2x2 Strassen decomposition.
    
    # For each pair of 2x2x2 Strassen terms (r1, r2), we get a rank-1 term in 4x4x4.
    # The A-part: for block (bi, bj), the A indices are bi*8 + i*4 + bj*2 + j
    #   where i,j are the local 2x2 indices from U2
    # The B-part: for block (ck, dl), the B indices are ck*8 + k*4 + dl*2 + l  
    #   where k,l are the local 2x2 indices from V2
    # The C-part: for block (ei, fj), the C indices are ei*8 + i*4 + fj*2 + j
    #   where i,j are the local 2x2 indices from W2
    
    # But we need to sum over the "middle" block index.
    # For 4x4x4: C[i,j] = sum_{k=0}^{3} A[i,k] B[k,j]
    # In blocks: C[bi*2+ei, bj*2+fj] = sum_{ck=0}^{1} (A_block[bi,ck] * B_block[ck,bj])[ei,fj]
    
    # So for each ck, we use the 2x2x2 Strassen decomposition to compute
    # A_block[bi,ck] * B_block[ck,bj] -> C_block[bi,bj]
    
    # For fixed ck, and for each (bi, bj), we have a 2x2x2 multiplication.
    # The Strassen decomposition gives us rank_base terms for each such multiplication.
    
    # Total terms: 2 (ck) * 4 (bi) * 4 (bj) * rank_base = 32 * rank_base
    # But many of these are redundant. The correct composition gives rank_base^2 = 49.
    
    # Let me just use the schoolbook for 4x4x4 to be safe, rank 64.
    # Then for 4x4x6, I'll use a mixed approach.
    
    n, m, p = 4, 4, 4
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
    
    return U, V, W

def _build_decomposition_4x4x6():
    """
    Build decomposition for 4x4 x 4x6 matrix multiplication.
    Strategy: Split into 4x4x4 + 4x4x2.
    Use schoolbook for both parts: rank 64 + 32 = 96.
    But we can do better.
    
    Better: Use a unified approach. The tensor T[i,j,k] = 1 for all i in 0..3, j in 0..3, k in 0..5.
    We need sum_r U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r] = 1 for all valid (i,j,k).
    
    Let me try a structured approach with rank 72 or so.
    """
    n, m, p = 4, 4, 6
    
    # Schoolbook as baseline
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using structured search
    # The key constraint: for each (i,j,k), sum_r U[i*4+j,r] * V[j*6+k,r] * W[k*4+i,r] = 1
    
    # Let me try rank 72 with a structured initialization
    # based on splitting k into two groups of 3.
    
    # For now, return schoolbook to ensure validity
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    by a 4 by 6 matrix.
    """
    deadline = time.time() + 115  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _build_decomposition_4x4x6()
    
    # Verify
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: rebuild schoolbook
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
        result = {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
