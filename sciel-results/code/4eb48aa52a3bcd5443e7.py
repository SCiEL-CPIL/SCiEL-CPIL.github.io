import numpy as np
import time

DEADLINE_SECONDS = 140.0


def _verify_decomposition(U, V, W, n, m, p, checks=500):
    """Verify that the decomposition correctly represents matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    # Build the full multiplication tensor T[a, b, c]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if the indices are consistent, else 0
    # The decomposition gives T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]

    # Check all valid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False

    # Check that off-diagonal terms are zero
    # For random (a,b,c) where the indices don't match
    for _ in range(checks):
        a = np.random.randint(n * m)
        b = np.random.randint(m * p)
        c = np.random.randint(p * n)
        
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k1 = b % p
        k2 = c // n
        i2 = c % n
        
        if j1 != j2 or k1 != k2 or i1 != i2:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val) > 1e-9:
                return False

    return True


def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U (4,7), V (4,7), W (4,7) where:
    U[i*2+j, r], V[i*2+j, r], W[i*2+j, r] for i,j in {0,1}
    and the multiplication tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    Actually for 2x2x2: T[a, b, c] where a=i*2+j, b=j*2+k, c=k*2+i
    """
    # Strassen's 7 bilinear forms for 2x2 multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U[i*2+j, r] corresponds to coefficient of A[i,j] in M_r
    # V[i*2+j, r] corresponds to coefficient of B[i,j] in M_r
    # W[i*2+j, r] corresponds to coefficient of C[i,j] in M_r (with sign)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00: W[0,0]=1, C11: W[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W


def _rank2_2x2x1():
    """Rank-2 decomposition for 2x2x1 (2x2 matrix times 2x1 vector).
    
    A is 2x2, B is 2x1, C = A*B is 2x1.
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    
    U has shape (4, 2), V has shape (2, 2), W has shape (2, 2)
    where U[i*2+j, r] for A[i,j], V[j, r] for B[j], W[i, r] for C[i]
    """
    # M1 = A00*B0 + A11*B1
    # M2 = A01*B1 + A10*B0
    # C0 = M1 + (A01*B1 - A11*B1) ... no that's not right
    
    # Let me think again.
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Rank 2:
    # M1 = A00*B0 + A10*B0 = (A00+A10)*B0
    # M2 = A01*B1 + A11*B1 = (A01+A11)*B1
    # C0 = M1_part0 + M2_part0 = A00*B0 + A01*B1
    # C1 = M1_part1 + M2_part1 = A10*B0 + A11*B1
    
    U = np.zeros((4, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    
    # M1 = (A00+A10)*B0
    U[0, 0] = 1  # A00
    U[2, 0] = 1  # A10
    V[0, 0] = 1  # B0
    W[0, 0] = 1  # contributes to C0
    W[1, 0] = 1  # contributes to C1
    
    # M2 = (A01+A11)*B1
    U[1, 1] = 1  # A01
    U[3, 1] = 1  # A11
    V[1, 1] = 1  # B1
    W[0, 1] = 1  # contributes to C0
    W[1, 1] = 1  # contributes to C1
    
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = 4, 4, 6
    
    # Strategy: Split the 4x6 B matrix into two 4x3 blocks
    # Use Strassen-like block decomposition
    
    # Approach: Decompose 4x4x6 as combination of smaller problems
    # We'll use a direct construction with known rank bounds
    
    # Known: rank(4x4x4) <= 49 (Strassen)
    # rank(4x4x2) <= 14 (7 blocks of 2x2x1, each rank 2)
    # But we need 4x4x6, so split B into 4x3 + 4x3 or 4x2+4x2+4x2
    
    # Let's use: 4x4x6 = 4x4x2 + 4x4x2 + 4x4x2
    # Each 4x4x2 has rank <= 8 (4 blocks of 2x2x1, each rank 2)
    # Total rank = 3 * 8 = 24? That seems too optimistic.
    
    # Actually, let me be more careful. For 4x4x2:
    # A is 4x4, B is 4x2, C is 4x2
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01], [B10, B11]] (each 2x1)
    # C = [[C00, C01], [C10, C11]] (each 2x1)
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each Cij is 2x1, computed as sum of two 2x2 * 2x1 products
    # Each 2x2*2x1 has rank 2
    # So C00 needs rank 4 (two products of rank 2 each)
    # Total for 4x4x2: 4 * 4 = 16
    
    # Hmm, let me just use the schoolbook for now and try to optimize.
    # Schoolbook rank = 4*4*6 = 96
    
    # Let's try a smarter approach.
    # We'll construct the decomposition using block Strassen where possible.
    
    # For 4x4x4, we can achieve rank 49 with Strassen.
    # For the remaining 4x4x2, we need rank <= 16.
    # Total: 49 + 16 = 65
    
    # But can we do better? Let's just implement the known best and verify.
    
    # Start with schoolbook to ensure correctness, then optimize.
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
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        raise TimeoutError("Exceeded deadline")
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
