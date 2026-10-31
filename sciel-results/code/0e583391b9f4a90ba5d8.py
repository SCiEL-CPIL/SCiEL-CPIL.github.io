import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = DIMENSIONS

    # Strassen 2x2 coefficients
    # a_coeffs[s, i, j] is coefficient of A[i, j] in the s-th linear form for A
    # b_coeffs[s, j, k] is coefficient of B[j, k] in the s-th linear form for B
    # c_coeffs[s, i, k] is coefficient of C[i, k] in the s-th linear form for C
    
    # Standard Strassen coefficients:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

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

    # We want to form U, V, W of shape (16, 49)
    # U[row, r] where row = i*4+j, r = 7*s+t
    # U[i*4+j, 7*s+t] = kron(a_coeffs[s], a_coeffs[t])[i, j]
    
    # Efficient construction using broadcasting and Kronecker product logic
    
    # a_coeffs shape: (7, 2, 2)
    # We need to compute kron(a_s, a_t) for all s, t.
    # kron(A, B) has shape (2*2, 2*2) = (4, 4).
    # kron(A, B)[2i + i', 2j + j'] = A[i, j] * B[i', j']
    
    # Let's create a 4D array U_4d of shape (7, 7, 4, 4)
    # U_4d[s, t, i, j] = a_coeffs[s, i//2, j//2] * a_coeffs[t, i%2, j%2]
    
    # Indices for a_coeffs[s]: S, T
    # Indices for a_coeffs[t]: i2, j2
    # i = 2*S + i2, j = 2*T + j2
    
    S = np.array([0, 0, 1, 1]) # i//2 for i in 0..3
    T = np.array([0, 1, 0, 1]) # j//2 for j in 0..3
    i2 = np.array([0, 1, 0, 1]) # i%2
    j2 = np.array([0, 1, 0, 1]) # j%2
    
    # a_coeffs[s, S, T]
    # Shape (7, 4) -> (7, 1, 1, 4) to broadcast with (1, 7, 4, 1)
    a_s_part = a_coeffs[:, S, T] # Shape (7, 4) where 4 corresponds to (S,T) pairs 00, 01, 10, 11? 
    # Wait, S and T arrays are for the 4 elements of the 4x4 matrix.
    # For a given (s, t), and a specific row i, col j in the 4x4:
    # We need a_coeffs[s, S, T] * a_coeffs[t, i2, j2]
    # where S=i//2, T=j//2, i2=i%2, j2=j%2.
    
    # Let's construct the full 4x4 matrices for each (s, t)
    # U_4d[s, t] is a 4x4 matrix.
    
    U_4d = np.zeros((7, 7, 4, 4), dtype=int)
    V_4d = np.zeros((7, 7, 4, 4), dtype=int)
    W_4d = np.zeros((7, 7, 4, 4), dtype=int)
    
    # Loop over s, t is 49 iterations, which is fast.
    for s in range(7):
        for t in range(7):
            # Kronecker product of 2x2 matrices
            U_4d[s, t] = np.kron(a_coeffs[s], a_coeffs[t])
            V_4d[s, t] = np.kron(b_coeffs[s], b_coeffs[t])
            # Note: W index is k*n + i. 
            # The standard matrix multiplication is C[i, k].
            # The tensor decomposition term is sum_r U_r(A) V_r(B) W_r(C).
            # W_r(C) is a linear form on C.
            # If the evaluator expects W[k*n+i], it means the vector of coefficients is indexed by (k, i).
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
    # U_4d shape (7, 7, 4, 4).
    # We want U_flat[row, r] where row = i*4+j, r = 7*s+t.
    # U_4d[s, t, i, j] -> U_flat[i*4+j, 7*s+t]
    
    # Reshape U_4d to (7, 7, 16) by flattening the last two dims (i, j) in row-major order.
    # i*4+j is the natural flattening order.
    U_flat = U_4d.reshape(7, 7, 16)
    # Permute to (16, 7, 7) -> (16, 49)
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
