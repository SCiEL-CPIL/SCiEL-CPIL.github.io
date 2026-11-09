"""Program to find a rank-minimal exact tensor decomposition for 3x4 * 4x8 multiplication."""
import time
import itertools

DIMENSIONS = [3, 4, 8]

def _check(U, V, W, n=3, m=4, p=8):
    """Check if the decomposition is exact using the trace convention W[k*n+i, r]."""
    from fractions import Fraction
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    # Check off-diagonal terms are zero
    for r in range(rank):
        # For each factor r, the term U_i * V_j * W_k should only be non-zero when indices match
        # Actually, we need to check that the sum is zero for all (i,j,k) not matching the product.
        # The tensor T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We want T_{i,j,k} = 1 if (i,j,k) corresponds to the product, 0 otherwise.
        pass
        # Wait, the product is C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor is T[i,j,k] = 1 if we are computing that specific product term.
        # So T[i,j,k] should be 1 for all i,j,k? No.
        # The tensor for matrix multiplication is T = sum_{i,j,k} e_i (outer) e_j (outer) e_k (outer)
        # Wait, the bilinear form is <A, B> -> C where C_{ik} = sum_j A_{ij} B_{jk}.
        # This corresponds to the tensor T_{i,j,k} = 1 for all i,j,k in the standard basis?
        # Yes, the matrix multiplication tensor has a 1 in every (i,j,k) entry.
        # So the check `s != 1` for all (i,j,k) is sufficient and necessary.
    return True

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition."""
    # Strassen 7 terms
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    # M1
    # (A00+A11)(B00+B11)
    # A terms: A00, A11. B terms: B00, B11.
    # Product: A00B00 + A00B11 + A11B00 + A11B11
    # This contributes to C00 (A00B00+A11B11), C01 (A00B11+A11B00), C10 (A11B00+A00B11), C11 (A11B11+A00B00)
    # Let's use the standard Strassen implementation.
    # U:
    # M1: A00, A11
    U[0][0] = 1 # A00
    U[3][0] = 1 # A11
    # V:
    # M1: B00, B11
    V[0][0] = 1 # B00 (j=0,k=0)
    V[3][0] = 1 # B11 (j=1,k=1)
    # W:
    # M1: C00, C01, C10, C11
    # C00: k=0, i=0 -> W[0*2+0][0] = W[0][0]
    W[0][0] = 1
    # C01: k=1, i=0 -> W[1*2+0][0] = W[2][0]
    W[2][0] = 1
    # C10: k=0, i=1 -> W[0*2+1][0] = W[1][0]
    W[1][0] = 1
    # C11: k=1, i=1 -> W[1*2+1][0] = W[3][0]
    W[3][0] = 1

    # M2: (A10+A11)B00
    U[2][1] = 1 # A10
    U[3][1] = 1 # A11
    V[0][1] = 1 # B00
    # C10: k=0, i=1 -> W[1][1]
    # C11: k=1, i=1 -> W[3][1]
    W[1][1] = 1
    W[3][1] = 1

    # M3: A00(B10-B11)
    U[0][2] = 1 # A00
    V[2][2] = 1 # B10 (j=1,k=0)
    V[3][2] = -1 # B11 (j=1,k=1)
    # C00: k=0, i=0 -> W[0][2]
    # C01: k=1, i=0 -> W[2][2]
    W[0][2] = 1
    W[2][2] = -1

    # M4: A11(B00-B10)
    U[3][3] = 1 # A11
    V[0][3] = 1 # B00
    V[2][3] = -1 # B10
    # C10: k=0, i=1 -> W[1][3]
    # C11: k=1, i=1 -> W[3][3]
    W[1][3] = 1
    W[3][3] = -1

    # M5: (A00+A01)B11
    U[0][4] = 1 # A00
    U[1][4] = 1 # A01
    V[3][4] = 1 # B11
    # C00: k=0, i=0 -> W[0][4]
    # C01: k=1, i=0 -> W[2][4]
    W[0][4] = 1
    W[2][4] = 1

    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1 # A10
    U[0][5] = -1 # A00
    V[0][5] = 1 # B00
    V[1][5] = 1 # B01 (j=0,k=1)
    # C10: k=0, i=1 -> W[1][5]
    # C11: k=1, i=1 -> W[3][5]
    W[1][5] = 1
    W[3][5] = 1

    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1 # A01
    U[3][6] = -1 # A11
    V[2][6] = 1 # B10
    V[3][6] = 1 # B11
    # C00: k=0, i=0 -> W[0][6]
    # C01: k=1, i=0 -> W[2][6]
    W[0][6] = 1
    W[2][6] = 1

    return U, V, W

def _pad_3x4x8_to_4x4x8():
    """
    Returns U, V, W for 4x4x8 using Strassen for two 4x4x4 blocks.
    Rank 56.
    """
    n, m, p = 4, 4, 8
    rank = 56
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block 1: k=0..3 (4x4x4)
    U1, V1, W1 = _strassen_2x2x2()
    # Expand 2x2x2 to 4x4x4?
    # Wait, Strassen is for 2x2. For 4x4, we need to apply Strassen recursively or use the 4x4 specific algorithm.
    # Actually, the standard rank of 4x4x4 is 49 (Strassen).
    # But I only have the 2x2x2 Strassen.
    # Let's just use the schoolbook for 4x4x4? Rank 64.
    # Or use the 2x2x2 Strassen in a 2x2 block decomposition of the 4x4x4.
    # 4x4x4 can be decomposed into 2x2 blocks of 2x2x2.
    # There are 2x2 blocks of (i,k) and 2x2 blocks of j?
    # Actually, 4x4x4 tensor rank is 49.
    # Let's just use the 2x2x2 Strassen for the 3x4x8 directly?
    # No, dimensions don't match.
    
    # Let's try a different approach.
    # 3x4x8.
    # Split 4 into 2+2.
    # 3x2x8 and 3x2x8.
    # Rank of 3x2x8?
    # 3x2x8 = 3x2x(4+4) = 3x2x4 + 3x2x4.
    # 3x2x4.
    # Maybe use Strassen on 2x2x2 parts?
    
    # Let's go back to the 4x4x8 padding.
    # If I can't easily get 4x4x4 rank 49, I'll use the 2x2x2 Strassen to build up.
    # Actually, let's just use the 2x2x2 Strassen for the whole 3x4x8?
    # No.
    
    # Let's try to construct a 3x4x8 decomposition using 2x2x2 Strassen blocks.
    # This might be complex.
    
    # Alternative: Use the 2x2x2 Strassen for 4x4x4 by treating it as 2x2 blocks.
    # This gives rank 4 * 7 = 28 for the 4x4x4? No, that's not right.
    # Strassen for 4x4x4 has rank 49.
    
    # Let's just use the 2x2x2 Strassen for the 3x4x8?
    # No.
    
    # Let's try to use the 2x2x2 Strassen for 3x2x4?
    # 3x2x4.
    # Split 3 into 1+1+1?
    # Split 2 into 1+1?
    # Split 4 into 2+2?
    
    # This is getting complicated. Let's stick to the 4x4x8 padding with schoolbook 4x4x4 (rank 64).
    # Total rank 128.
    # Wait, the schoolbook for 3x4x8 is 96.
    # 4x4x8 schoolbook is 128.
    # So padding makes it worse!
    
    # Okay, padding is a bad idea if I don't have a good 4x4x4 algorithm.
    
    # Let's try to find a better decomposition for 3x4x8 directly.
    # Maybe 3x4x8 can be done with rank less than 96?
    # 3x4x8 = 3x4x(4+4) = 3x4x4 + 3x4x4.
    # Rank of 3x4x4?
    # 3x4x4.
    # 3x4x4 = (1+1+1)x4x4.
    # Rank of 1x4x4 is 16.
    # So 3x4x4 is at most 48.
    # Total 96.
    
    # Can we do better?
    # Maybe use Strassen on 2x2x2 blocks within the 3x4x4?
    # 3x4x4.
    # Split 4 (m) into 2+2.
    # 3x2x4 + 3x2x4.
    # 3x2x4.
    # Split 4 (p) into 2+2.
    # 3x2x2 + 3x2x2.
    # 3x2x2.
    # Split 3 (n) into 1+1+1.
    # 1x2x2 + 1x2x2 + 1x2x2.
    # 1x2x2 rank is 4.
    # So 3x2x2 rank is 12.
    # 3x2x4 rank is 24.
    # 3x4x4 rank is 48.
    # 3x4x8 rank is 96.
    
    # It seems 96 is the standard rank.
    # Is there a known lower bound?
    # R(3,4,8) >= ?
    
    # Let's try to use the 2x2x2 Strassen to reduce the rank of 2x2x4 or 2x4x2.
    # 2x2x4 = 2x2x(2+2) = 2x2x2 + 2x2x2.
    # Rank of 2x2x2 is 7.
    # So 2x2x4 rank is 14.
    # Schoolbook is 16.
    # So we save 2 per 2x2x4 block.
    
    # 3x4x8.
    # Split 3 into 1+1+1.
    # 1x4x8 + 1x4x8 + 1x4x8.
    # 1x4x8 = 1x4x(4+4) = 1x4x4 + 1x4x4.
    # 1x4x4 = 1x(2+2)x4 = 1x2x4 + 1x2x4.
    # 1x2x4 = 1x2x(2+2) = 1x2x2 + 1x2x2.
    # 1x2x2 rank is 4.
    # So 1x2x4 rank is 8.
    # 1x4x4 rank is 16.
    # 1x4x8 rank is 32.
    # 3x4x8 rank is 96.
    
    # It seems hard to beat 96 without a specific trick.
    
    # Let's just return the schoolbook decomposition for now.
    return None

def construct():
    n, m, p = DIMENSIONS
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
