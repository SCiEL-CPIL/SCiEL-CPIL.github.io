"""
FTW-inspired decomposition for 2x4x6 matrix multiplication.
Uses Kronecker product of (2x2x2 Strassen) and (1x2x3 Trivial).
Rank = 7 * 6 = 42.
"""
import numpy as np

def construct():
    # Dimensions
    n, m, p = 2, 4, 6
    
    # Core 1: 2x2x2 Strassen Tensor
    # Indices: i (2), j (2), k (2)
    # Rank 7
    # We represent the tensor as U1, V1, W1 of shape (2*2, 7), (2*2, 7), (2*2, 7)
    # U1 row index: i*2 + j
    # V1 row index: j*2 + k
    # W1 row index: k*2 + i (Trace convention W[k*n+i])
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U1 (A side)
    U1[0, 0] = 1; U1[3, 0] = 1  # M1: A00, A11
    U1[2, 1] = 1; U1[3, 1] = 1  # M2: A10, A11
    U1[0, 2] = 1                # M3: A00
    U1[3, 3] = 1                # M4: A11
    U1[0, 4] = 1; U1[1, 4] = 1  # M5: A00, A01
    U1[2, 5] = 1; U1[0, 5] = -1 # M6: A10, -A00
    U1[1, 6] = 1; U1[3, 6] = -1 # M7: A01, -A11
    
    # V1 (B side)
    V1[0, 0] = 1; V1[3, 0] = 1  # M1: B00, B11
    V1[0, 1] = 1                # M2: B00
    V1[1, 2] = 1; V1[3, 2] = -1 # M3: B01, -B11
    V1[2, 3] = 1; V1[0, 3] = -1 # M4: B10, -B00
    V1[3, 4] = 1                # M5: B11
    V1[0, 5] = 1; V1[1, 5] = 1  # M6: B00, B01
    V1[2, 6] = 1; V1[3, 6] = 1  # M7: B10, B11
    
    # W1 (Output side)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # W index: k*2 + i
    # C00 (i=0, k=0) -> idx 0
    # C01 (i=0, k=1) -> idx 2
    # C10 (i=1, k=0) -> idx 1
    # C11 (i=1, k=1) -> idx 3
    
    W1[0, 0] = 1; W1[0, 3] = 1; W1[0, 4] = -1; W1[0, 6] = 1 # C00
    W1[2, 2] = 1; W1[2, 4] = 1                               # C01
    W1[1, 1] = 1; W1[1, 3] = 1                               # C10
    W1[3, 0] = 1; W1[3, 1] = -1; W1[3, 2] = 1; W1[3, 5] = 1  # C11
    
    # Core 2: 1x2x3 Trivial Tensor
    # Indices: i (1), j (2), k (3)
    # Rank 6
    # U2 shape (1*2, 6) = (2, 6)
    # V2 shape (2*3, 6) = (6, 6)
    # W2 shape (3*1, 6) = (3, 6)
    
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    
    # Multiplications:
    # M0: A0 * B00
    # M1: A0 * B01
    # M2: A0 * B02
    # M3: A1 * B10
    # M4: A1 * B11
    # M5: A1 * B12
    
    # U2 (A side, size 1x2)
    # A0 is index 0, A1 is index 1
    U2[0, 0] = 1
    U2[0, 1] = 1
    U2[0, 2] = 1
    U2[1, 3] = 1
    U2[1, 4] = 1
    U2[1, 5] = 1
    
    # V2 (B side, size 2x3)
    # B[j, k] index j*3 + k
    # B00: idx 0, B01: idx 1, B02: idx 2
    # B10: idx 3, B11: idx 4, B12: idx 5
    V2[0, 0] = 1
    V2[1, 1] = 1
    V2[2, 2] = 1
    V2[3, 3] = 1
    V2[4, 4] = 1
    V2[5, 5] = 1
    
    # W2 (Output side, size 3x1)
    # C[k, 0] index k*1 + 0 = k
    # C0 = M0 + M3
    # C1 = M1 + M4
    # C2 = M2 + M5
    W2[0, 0] = 1; W2[0, 3] = 1
    W2[1, 1] = 1; W2[1, 4] = 1
    W2[2, 2] = 1; W2[2, 5] = 1
    
    # Kronecker Product
    # Tensor T = T1 (2,2,2) x T2 (1,2,3)
    # Dimensions:
    # n: 2 * 1 = 2
    # m: 2 * 2 = 4
    # p: 2 * 3 = 6
    # Rank: 7 * 6 = 42
    
    # U (n*m, rank) = (8, 42)
    # U index: (i1, j1, i2, j2) -> (i1*2 + i2) * (j1*2 + j2) ??
    # A is 2x4.
    # A[i, a] where i in {0,1}, a in {0,1,2,3}.
    # Decomposition: i = i1 (from T1), a = a1*2 + a2 (from T2)?
    # T1 uses A1[i1, j1] where i1, j1 in {0,1}.
    # T2 uses A2[i2, j2] where i2=0, j2 in {0,1}.
    # Combined A index: i = i1, a = j1*2 + j2.
    # Global A row index: i * 4 + a = i1 * 4 + j1 * 2 + j2.
    
    U = np.zeros((8, 42))
    V = np.zeros((24, 42))
    W = np.zeros((12, 42))
    
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            
            # U
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U1[i1*2 + j1, r1]
                    if val1 != 0:
                        for i2 in range(1):
                            for j2 in range(2):
                                val2 = U2[i2*2 + j2, r2]
                                if val2 != 0:
                                    i = i1
                                    a = j1 * 2 + j2
                                    idx_u = i * 4 + a
                                    U[idx_u, r] += val1 * val2
            
            # V
            # B is 4x6.
            # B[a, b] where a in {0,1,2,3}, b in {0,1,2,3,4,5}.
            # Decomposition: a = a1*2 + a2? No.
            # T1 uses B1[j1, k1] where j1, k1 in {0,1}.
            # T2 uses B2[j2, k2] where j2 in {0,1}, k2 in {0,1,2}.
            # Combined B index:
            # The "m" dimension of T1 is j1. The "m" dimension of T2 is j2.
            # In Kronecker product of matrices A (n x m1) and B (m1 x p1) ...
            # Here we are multiplying Tensors.
            # T1: A1 (2x2), B1 (2x2), C1 (2x2).
            # T2: A2 (1x2), B2 (2x3), C2 (3x1).
            # Combined:
            # A (2x4): A[i, a] = A1[i, j1] * A2[0, j2] ??
            # Actually, the Kronecker product of tensors corresponds to:
            # A[i, a] where i = i1, a = j1 * 2 + j2.
            # B[a, b] where a = j1 * 2 + j2, b = k1 * 3 + k2.
            # C[i, b] where i = i1, b = k1 * 3 + k2.
            
            # Check V index: a * 6 + b = (j1*2 + j2) * 6 + (k1*3 + k2).
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V1[j1*2 + k1, r1]
                    if val1 != 0:
                        for j2 in range(2):
                            for k2 in range(3):
                                val2 = V2[j2*3 + k2, r2]
                                if val2 != 0:
                                    a = j1 * 2 + j2
                                    b = k1 * 3 + k2
                                    idx_v = a * 6 + b
                                    V[idx_v, r] += val1 * val2
            
            # W
            # C[i, b] where i = i1, b = k1 * 3 + k2.
            # W index: b * n + i = (k1*3 + k2) * 2 + i1.
            for i1 in range(2):
                for k1 in range(2):
                    val1 = W1[k1*2 + i1, r1]
                    if val1 != 0:
                        for i2 in range(1):
                            for k2 in range(3):
                                val2 = W2[k2*1 + i2, r2]
                                if val2 != 0:
                                    i = i1
                                    b = k1 * 3 + k2
                                    idx_w = b * 2 + i
                                    W[idx_w, r] += val1 * val2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
