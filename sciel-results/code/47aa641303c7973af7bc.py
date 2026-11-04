import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 multiplications
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U rows correspond to A entries (i,j) -> index i*2+j
    # V rows correspond to B entries (j,k) -> index j*2+k
    # W rows correspond to C entries (k,i) -> index k*2+i (Official trace convention W[k*n+i,r] for n=p=2)
    
    U = np.array([
        # M1: A11, A22
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        # M2: A21, A22
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        # M3: A11
        [1, 0, 0, 0, 0, 0, 0],
        # M4: A22
        [0, 0, 0, 0, 1, 0, 0],
        # M5: A11, A12
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        # M6: A21, A11
        [0, 0, 0, 1, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0],
        # M7: A12, A22
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0],
    ], dtype=int).T # 4x7? No, this is 12x7. 
    # Wait, the standard Strassen decomposition has 7 terms.
    # Each term r has a vector u_r (size 4), v_r (size 4), w_r (size 4).
    # U is 4x7.
    
    # Let's construct U, V, W columns directly.
    # r=0: M1
    u0 = np.array([1, 0, 0, 1]) # A11, A12, A21, A22? No. A is 2x2.
    # Indexing: 0:00, 1:01, 2:10, 3:11
    u0 = np.array([1, 0, 0, 1])
    v0 = np.array([1, 0, 0, 1]) # B11, B12, B21, B22? No. B is 2x2.
    # Indexing: 0:00, 1:01, 2:10, 3:11
    v0 = np.array([1, 0, 0, 1])
    w0 = np.array([1, 0, 0, 1]) # C11, C12, C21, C22?
    # C = [[C11, C12], [C21, C22]]
    # Indexing: 0:00, 1:01, 2:10, 3:11
    # M1 contributes to C11, C12, C21, C22?
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M3 + M6
    
    # Let's use the standard coefficients.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U columns (A coeffs for M1..M7)
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ], dtype=int).T # 4x7

    # V columns (B coeffs for M1..M7)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0],  # B12
        [0, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ], dtype=int).T # 4x7

    # W columns (C coeffs for M1..M7)
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M3 + M6
    W = np.array([
        [1, 0, 0, 0, 0, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0], # C12
        [0, 1, 0, 1, 0, 0, 0], # C21
        [1, -1, -1, 0, 0, 1, 0] # C22
    ], dtype=int).T # 4x7
    
    return U, V, W

def _rect_224():
    """
    Decomposition for 2x2x4 with rank 14.
    """
    Us, Vs, Ws = _strassen_222()
    Ur = np.hstack([Us, Us]) # 4x14
    Vr = np.zeros((8, 14), dtype=int)
    # B is 2x4. Rows j*4+k.
    # Left 2x2: j in 0..1, k in 0..1 -> indices 0..3
    # Right 2x2: j in 0..1, k in 2..3 -> indices 4..7
    for j in range(2):
        for k in range(2):
            # Left block B[j, k]
            Vr[j * 4 + k, 0:7] = Vs[j * 2 + k, :]
            # Right block B[j, k+2]
            Vr[j * 4 + k + 2, 7:14] = Vs[j * 2 + k, :]
    
    Wr = np.zeros((8, 14), dtype=int)
    # C is 2x4. Rows k*2+i.
    # Left 2x2: k in 0..1, i in 0..1 -> indices 0..3
    # Right 2x2: k in 2..3, i in 0..1 -> indices 4..7
    for k in range(2):
        for i in range(2):
            Wr[k * 2 + i, 0:7] = Ws[k * 2 + i, :]
            Wr[k * 2 + i, 7:14] = Ws[k * 2 + i, :]
            
    return Ur, Vr, Wr

def _build_448():
    """
    Decomposition for 4x4x8 with rank 98.
    Uses Kronecker product of 2x2x4 decomposition.
    """
    Us, Vs, Ws = _rect_224() # 4x14, 8x14, 8x14
    # We need 4x4x8.
    # A is 4x4. B is 4x8. C is 4x8.
    # We can view A as 2x2 blocks of 2x2.
    # B as 2x4 blocks of 2x2? No.
    # Let's use the Kronecker product of the 2x2x4 decomposition with itself?
    # No, 2x2x4 is not a Kronecker factor of 4x4x8.
    
    # Correct approach:
    # 4x4x8 = (2x2x4) x (2x1x1)? No.
    # 4x4x8 can be built from 2x2x4 by Kronecker product with 2x2x1?
    # No.
    
    # Let's use the 4x4x4 rank 49 decomposition and double it.
    # 4x4x4 rank 49:
    # U444, V444, W444 = kron(Us222, Us222) etc?
    # No, as discussed, kron of 222 is not valid for general 444.
    
    # However, the current program's _rect_224 logic was extended in the seed.
    # The seed used:
    # U = kron(Us, Ur) where Us is 4x7 (222) and Ur is 4x14 (224).
    # This implies a specific block structure.
    
    # Let's stick to the seed's successful logic which achieved 1.3061.
    # The seed achieved rank 98.
    
    # I will replicate the seed's rank-98 construction exactly but with cleaner code.
    Us222, Vs222, Ws222 = _strassen_222()
    Ur224, Vr224, Wr224 = _rect_224()
    
    # The seed used:
    # U = np.kron(Us, Ur) # 16x98
    # V = np.kron(Vs, Vr) # 32x98
    # W = np.kron(Ws, Wr) # 32x98
    # And then permutations.
    
    U = np.kron(Us222, Ur224)
    V = np.kron(Vs222, Vr224)
    W = np.kron(Ws222, Wr224)
    
    # Permutations to official index convention.
    # A: 4x4. Index i*4+j.
    # U rows: i1*8 + i2*4 + j1*2 + j2 (from kron) -> i1*8 + j1*4 + i2*2 + j2 (official)
    permU = {}
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    permU[i1 * 8 + i2 * 4 + j1 * 2 + j2] = i1 * 8 + j1 * 4 + i2 * 2 + j2
    U = U[[permU[y] for y in range(16)]]
    
    # B: 4x8. Index j*8+k.
    # V rows: j1*16 + j2*8 + k1*2 + k2 (from kron) -> j1*16 + k1*8 + j2*4 + k2 (official)
    permV = {}
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    permV[j1 * 16 + j2 * 8 + k1 * 2 + k2] = j1 * 16 + k1 * 8 + j2 * 4 + k2
    V = V[[permV[y] for y in range(32)]]
    
    # C: 4x8. Index k*4+i.
    # W rows: k1*8 + k2*4 + i1*2 + i2 (from kron) -> k1*16 + i1*8 + k2*4 + i2 (official)
    permW = {}
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    permW[k1 * 8 + k2 * 4 + i1 * 2 + i2] = k1 * 16 + i1 * 8 + k2 * 4 + i2
    W = W[[permW[y] for y in range(32)]]
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    rng = np.random.default_rng(12345)
    for _ in range(3):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C = np.zeros((n, p))
        for c in range(U.shape[1]):
            a = U[:, c].reshape(n, m) @ A
            b = V[:, c].reshape(m, p) @ B
            C += a * b * W[:, c].reshape(p, n).T
        if not np.allclose(C, C_true, atol=1e-8):
            return False
    return True

def _schoolbook(n, m, p):
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

def construct():
    import time
    t0 = time.time()
    n, m, p = DIMENSIONS
    try:
        U, V, W = _build_448()
        if time.time() - t0 < 110 and _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
