import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS

    # Strassen coefficients for 2x2 matrices (0-indexed):
    # s=0..6
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

    # Build 4x4x4 rank-49 decomposition via Kronecker products of 2x2 Strassen
    # Direct 2D construction to avoid 4D intermediate
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for s in range(7):
        for t in range(7):
            idx = s * 7 + t
            
            # U[i*4+j, idx] = kron(a_s, a_t)[i, j]
            u_block = np.kron(a_coeffs[s], a_coeffs[t])
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j, idx] = u_block[i, j]
            
            # V[i*4+j, idx] = kron(b_s, b_t)[i, j]
            v_block = np.kron(b_coeffs[s], b_coeffs[t])
            for i in range(4):
                for j in range(4):
                    V[i * 4 + j, idx] = v_block[i, j]
            
            # W: trace convention W[k*n+i, r] where C[i,k] has coefficient
            # kron(c_s, c_t)[i, k] gives coeff for C[i,k]
            # So W[k*4+i, idx] = kron(c_s, c_t)[i, k]
            c_block = np.kron(c_coeffs[s], c_coeffs[t])
            for i in range(4):
                for k in range(4):
                    W[k * 4 + i, idx] = c_block[i, k]
            
            if time.time() > deadline - 5.0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
