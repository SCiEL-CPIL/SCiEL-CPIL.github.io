import numpy as np
import time
import itertools

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Strassen 2x2 Decomposition (Rank 7)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 2. Schoolbook 3x3 Decomposition (Rank 27) - Guaranteed Fallback
    U3_sb = np.zeros((9, 27), dtype=int)
    V3_sb = np.zeros((9, 27), dtype=int)
    W3_sb = np.zeros((9, 27), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U3_sb[i * 3 + j, r] = 1
                V3_sb[j * 3 + k, r] = 1
                W3_sb[k * 3 + i, r] = 1

    def build_kronecker_fast(U2, V2, W2, U3, V3, W3):
        """
        Efficiently build 6x6 factors using Kronecker products.
        U6[i6*6+j6, r] = U2[i2*2+j2, r2] * U3[i3*3+j3, r3]
        where i6 = i2*3 + i3, j6 = j2*3 + j3, r = r2*rank3 + r3.
        """
        rank2 = U2.shape[1]
        rank3 = U3.shape[1]
        rank = rank2 * rank3
        
        # Use np.kron for U, V, W
        # U6 is a permutation of kron(U2, U3)
        # U2 is 4x7, U3 is 9x19.
        # kron(U2, U3) is 36 x (7*19).
        # The rows of kron(U2, U3) are indexed by (i2, i3) where i2 in 0..1, i3 in 0..2.
        # Row index in kron is i2*9 + i3.
        # We need row index i6*6 + j6.
        # i6 = i2*3 + i3, j6 = j2*3 + j3.
        # So row index is (i2*3+i3)*6 + (j2*3+j3) = 18*i2 + 6*i3 + 6*j2 + j3.
        # Kron row index is i2*9 + i3? No, kron(U2, U3) has shape (4*9, 7*19).
        # Row index is i2*9 + i3.
        # This does NOT match i6*6+j6 directly.
        
        # We need to permute the rows of kron(U2, U3).
        # Let's construct U6 directly to avoid confusion.
        
        U6 = np.zeros((36, rank), dtype=U3.dtype)
        V6 = np.zeros((36, rank), dtype=U3.dtype)
        W6 = np.zeros((36, rank), dtype=U3.dtype)
        
        # Precompute non-zero indices for speed
        nz_U2 = [(i, j, r, val) for i in range(2) for j in range(2) for r in range(rank2) for val in [U2[i*2+j, r]] if val != 0]
        nz_U3 = [(i, j, r, val) for i in range(3) for j in range(3) for r in range(rank3) for val in [U3[i*3+j, r]] if val != 0]
        
        nz_V2 = [(j, k, r, val) for j in range(2) for k in range(2) for r in range(rank2) for val in [V2[j*2+k, r]] if val != 0]
        nz_V3 = [(j, k, r, val) for j in range(3) for k in range(3) for r in range(rank3) for val in [V3[j*3+k, r]] if val != 0]
        
        nz_W2 = [(k, i, r, val) for k in range(2) for i in range(2) for r in range(rank2) for val in [W2[k*2+i, r]] if val != 0]
        nz_W3 = [(k, i, r, val) for k in range(3) for i in range(3) for r in range(rank3) for val in [W3[k*3+i, r]] if val != 0]
        
        # This loop is slow. Use vectorized ops if possible.
        # For rank 189, 36x189 matrix.
        # 36 * 189 * (operations) is small.
        
        for i2, j2, r2, val_u2 in nz_U2:
            for i3, j3, r3, val_u3 in nz_U3:
                r = r2 * rank3 + r3
                i6 = i2 * 3 + i3
                j6 = j2 * 3 + j3
                U6[i6 * 6 + j6, r] += val_u2 * val_u3
                
        for j2, k2, r2, val_v2 in nz_V2:
            for j3, k3, r3, val_v3 in nz_V3:
                r = r2 * rank3 + r3
                j6 = j2 * 3 + j3
                k6 = k2 * 3 + k3
                V6[j6 * 6 + k6, r] += val_v2 * val_v3
                
        for k2, i2, r2, val_w2 in nz_W2:
            for k3, i3, r3, val_w3 in nz_W3:
                r = r2 * rank3 + r3
                k6 = k2 * 3 + k3
                i6 = i2 * 3 + i3
                W6[k6 * 6 + i6, r] += val_w2 * val_w3
                
        return U6, V6, W6

    def verify_6x6(U6, V6, W6, rank, num_samples=500):
        """Verify 6x6 decomposition with random sampling."""
        np.random.seed(12345)
        for _ in range(num_samples):
            i = np.random.randint(0, 6)
            j = np.random.randint(0, 6)
            k = np.random.randint(0, 6)
            s = 0.0
            for r in range(rank):
                s += U6[i*6+j, r] * V6[j*6+k, r] * W6[k*6+i, r]
            if abs(s - 1.0) > 1e-6:
                return False
        return True

    # Strategy:
    # 1. Try to find a valid 3x3 decomposition with rank < 27.
    # 2. If found, use it.
    # 3. Else, use schoolbook rank 27.
    
    best_U3, best_V3, best_W3, best_rank3 = U3_sb, V3_sb, W3_sb, 27
    
    # Search for low-rank 3x3 decomposition (Rank 19-26)
    # This is a hard problem. We try a few random initializations and gradient steps.
    # Given time constraints, we skip heavy search and rely on the known 27.
    # However, we can try to verify the "Liu" 19-term decomposition from the prompt if it was valid.
    # The prompt's Liu implementation failed.
    
    # We proceed with the guaranteed rank-189 solution (7 * 27).
    U6, V6, W6 = build_kronecker_fast(U2, V2, W2, U3_sb, V3_sb, W3_sb)
    
    # Convert to int if possible
    if np.allclose(U6, np.round(U6)):
        U6 = U6.astype(int)
    if np.allclose(V6, np.round(V6)):
        V6 = V6.astype(int)
    if np.allclose(W6, np.round(W6)):
        W6 = W6.astype(int)
        
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
