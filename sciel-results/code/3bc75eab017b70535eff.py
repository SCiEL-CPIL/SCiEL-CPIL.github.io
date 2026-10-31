import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = DIMENSIONS

    # Recursive Strassen 4x4x4 construction (rank 49).
    # This is the best known explicit rank for 4x4 matrix multiplication.
    # We construct the decomposition by applying Strassen's 7-term algorithm
    # to the 2x2 block structure, where each block is a 2x2 matrix.
    # The resulting rank is 7*7 = 49.
    
    # Strassen coefficients for 2x2 matrices:
    # a_coeffs[s, i, j] is coefficient of A[i, j] in the s-th linear form for A
    # b_coeffs[s, j, k] is coefficient of B[j, k] in the s-th linear form for B
    # c_coeffs[s, i, k] is coefficient of C[i, k] in the s-th linear form for C
    
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: A00 + A11
        [[0, 0], [1, 1]],   # s=1: A10 + A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00 + A01
        [[-1, 0], [1, 0]],  # s=5: A10 - A00
        [[0, 1], [0, -1]],  # s=6: A01 - A11
    ], dtype=int)

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: B00 + B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01 - B11
        [[-1, 0], [1, 0]],  # s=3: B10 - B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00 + B01
        [[0, 0], [1, 1]],   # s=6: B10 + B11
    ], dtype=int)

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: C00 + C11
        [[0, 0], [1, -1]],  # s=1: C10 - C11
        [[0, 1], [0, 1]],   # s=2: C01 + C11
        [[1, 0], [1, 0]],   # s=3: C00 + C10
        [[-1, 1], [0, 0]],  # s=4: -C00 + C01
        [[0, 0], [0, 1]],   # s=5: C11
        [[1, 0], [0, 0]],   # s=6: C00
    ], dtype=int)

    # Construct U, V, W of shape (16, 49)
    # U[row, r] where row = i*4+j, r = 7*s+t
    # U[i*4+j, 7*s+t] = kron(a_coeffs[s], a_coeffs[t])[i, j]
    
    U_4d = np.zeros((7, 7, 4, 4), dtype=int)
    V_4d = np.zeros((7, 7, 4, 4), dtype=int)
    W_4d = np.zeros((7, 7, 4, 4), dtype=int)
    
    for s in range(7):
        for t in range(7):
            U_4d[s, t] = np.kron(a_coeffs[s], a_coeffs[t])
            V_4d[s, t] = np.kron(b_coeffs[s], b_coeffs[t])
            # W index is k*n + i. The standard matrix multiplication is C[i, k].
            # My c_coeffs are indexed by [i, k] (row i, col k).
            # So kron(c_s, c_t) gives a 4x4 matrix M where M[i, k] is the coeff for C[i, k].
            # I need to map this to index k*4 + i.
            # So I should transpose the 4x4 matrix to [k, i] before flattening.
            W_4d[s, t] = np.kron(c_coeffs[s], c_coeffs[t]).T
            
            if time.time() > deadline:
                return {"U": np.zeros((16, 49), dtype=int).tolist(), 
                        "V": np.zeros((16, 49), dtype=int).tolist(), 
                        "W": np.zeros((16, 49), dtype=int).tolist()}

    # Flatten to (16, 49)
    U_flat = U_4d.reshape(7, 7, 16)
    U_perm = np.transpose(U_flat, (2, 0, 1))
    U = U_perm.reshape(16, 49)
    
    V_flat = V_4d.reshape(7, 7, 16)
    V_perm = np.transpose(V_flat, (2, 0, 1))
    V = V_perm.reshape(16, 49)
    
    W_flat = W_4d.reshape(7, 7, 16)
    W_perm = np.transpose(W_flat, (2, 0, 1))
    W = W_perm.reshape(16, 49)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
