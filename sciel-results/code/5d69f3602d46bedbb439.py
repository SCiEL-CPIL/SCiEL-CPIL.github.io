"""
Rank-63 exact tensor decomposition for 6x6 matrix multiplication via Kronecker product
of the 2x2 Strassen tensor (rank 7) and the 3x3 schoolbook tensor (rank 9).
"""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # The 6x6 multiplication tensor is the tensor product of the 2x2 and 3x3 multiplication tensors.
    # Rank(T_2 ⊗ T_3) = Rank(T_2) * Rank(T_3) = 7 * 9 = 63.
    
    rank = 63
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 2x2 Strassen decomposition terms
    # Each term is a triple (u, v, w) where u, v, w are 2x2 matrices (flattened)
    # U acts on A, V acts on B, W acts on C.
    # Indices: row * dim + col for U/V, col * dim + row for W (trace convention)
    
    strassen_2x2 = [
        # M1 = (A00+A11)(B00+B11)
        (
            np.array([[1, 0], [0, 1]]).flatten(),
            np.array([[1, 0], [0, 1]]).flatten(),
            np.array([[1, 0], [0, 1]]).flatten()
        ),
        # M2 = (A10+A11)B00
        (
            np.array([[0, 0], [1, 1]]).flatten(),
            np.array([[1, 0], [0, 0]]).flatten(),
            np.array([[0, 0], [1, -1]]).flatten()
        ),
        # M3 = A00(B01-B11)
        (
            np.array([[1, 0], [0, 0]]).flatten(),
            np.array([[0, 1], [0, -1]]).flatten(),
            np.array([[0, 1], [0, 1]]).flatten()
        ),
        # M4 = A11(B10-B00)
        (
            np.array([[0, 0], [0, 1]]).flatten(),
            np.array([[0, 0], [1, -1]]).flatten(),
            np.array([[1, 0], [1, 0]]).flatten()
        ),
        # M5 = (A00+A01)B11
        (
            np.array([[1, 1], [0, 0]]).flatten(),
            np.array([[0, 0], [0, 1]]).flatten(),
            np.array([[-1, 1], [0, 0]]).flatten()
        ),
        # M6 = (A10-A00)(B00+B01)
        (
            np.array([[-1, 0], [1, 0]]).flatten(),
            np.array([[1, 1], [0, 0]]).flatten(),
            np.array([[0, 0], [0, 1]]).flatten()
        ),
        # M7 = (A01-A11)(B10+B11)
        (
            np.array([[0, 1], [0, -1]]).flatten(),
            np.array([[0, 0], [1, 1]]).flatten(),
            np.array([[1, 0], [0, 0]]).flatten()
        )
    ]
    
    # Define the 3x3 schoolbook decomposition terms
    # There are 9 terms, one for each (i, j, k) triple where k is the summation index.
    # A[i, k] * B[k, j] -> C[i, j]
    schoolbook_3x3 = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                u = np.zeros(9)
                v = np.zeros(9)
                w = np.zeros(9)
                u[i * 3 + k] = 1
                v[k * 3 + j] = 1
                # W index: col * n + row = j * 3 + i
                w[j * 3 + i] = 1
                schoolbook_3x3.append((u, v, w))
    
    # Construct the full 6x6 decomposition using the Kronecker product structure
    # The 6x6 matrices are indexed as (s, r) where s is the 3x3 block index and r is the 2x2 offset.
    # A is 6x6: A[(3*s1 + r1), (3*s2 + r2)]
    # B is 6x6: B[(3*s2 + r2), (3*s3 + r3)]
    # C is 6x6: C[(3*s1 + r1), (3*s3 + r3)]
    
    # The multiplication tensor is the tensor product T_3 ⊗ T_2.
    # The rank decomposition terms are indexed by (i3, i2) where i3 in 0..8, i2 in 0..6.
    # Total rank = 9 * 7 = 63.
    
    # For each term in the decomposition:
    # U_6[(3*s1+r1), (3*s2+r2)] = U_3[i3][(s1, s2)] * U_2[i2][(r1, r2)]
    # V_6[(3*s2+r2), (3*s3+r3)] = V_3[i3][(s2, s3)] * V_2[i2][(r2, r3)]
    # W_6[(3*s3+r3), (3*s1+r1)] = W_3[i3][(s3, s1)] * W_2[i2][(r3, r1)]
    
    # Note: The W index convention is W[k*n+i, r] where k is col, i is row.
    # In our local 2x2 Strassen terms, W was defined with index col*2 + row.
    # In our local 3x3 schoolbook terms, W was defined with index col*3 + row.
    
    # Let's iterate over all combinations of 3x3 term index i3 and 2x2 term index i2.
    
    for i3 in range(9):
        for i2 in range(7):
            col_idx = i3 * 7 + i2
            
            u3, v3, w3 = schoolbook_3x3[i3]
            u2, v2, w2 = strassen_2x2[i2]
            
            # Reshape to matrices for clarity
            U3 = u3.reshape(3, 3)
            V3 = v3.reshape(3, 3)
            W3 = w3.reshape(3, 3) # Indices: [col, row]
            
            U2 = u2.reshape(2, 2)
            V2 = v2.reshape(2, 2)
            W2 = w2.reshape(2, 2) # Indices: [col, row]
            
            # Fill U
            # U_6[row, col] where row = 3*s1 + r1, col = 3*s2 + r2
            for s1 in range(3):
                for r1 in range(2):
                    for s2 in range(3):
                        for r2 in range(2):
                            val = U3[s1, s2] * U2[r1, r2]
                            if val != 0:
                                row = 3 * s1 + r1
                                col = 3 * s2 + r2
                                U[row * 6 + col, col_idx] += val
            
            # Fill V
            # V_6[row, col] where row = 3*s2 + r2, col = 3*s3 + r3
            for s2 in range(3):
                for r2 in range(2):
                    for s3 in range(3):
                        for r3 in range(2):
                            val = V3[s2, s3] * V2[r2, r3]
                            if val != 0:
                                row = 3 * s2 + r2
                                col = 3 * s3 + r3
                                V[row * 6 + col, col_idx] += val
            
            # Fill W
            # W_6[col * 6 + row] where col = 3*s3 + r3, row = 3*s1 + r1
            for s3 in range(3):
                for r3 in range(2):
                    for s1 in range(3):
                        for r1 in range(2):
                            val = W3[s3, s1] * W2[r3, r1]
                            if val != 0:
                                row = 3 * s1 + r1
                                col = 3 * s3 + r3
                                W[col * 6 + row, col_idx] += val
    
    # Convert to lists
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
