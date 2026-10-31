import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.array([
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0]
    ], dtype=float)
    
    # Kronecker products for 4x4
    # Shape (16, 49)
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Verify correctness
    # Random test
    np.random.seed(42)
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)
    
    # Flatten A and B in row-major order
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # Compute intermediate products
    # Z = U4 @ A_flat  (shape 49)
    Z = U4 @ A_flat
    Z = Z * (V4 @ B_flat)
    
    # Compute C_flat
    C_flat = W4 @ Z
    
    # Reshape C
    C_computed = C_flat.reshape(4, 4)
    
    # Standard matrix multiplication
    C_expected = A @ B
    
    # Check error
    error = np.max(np.abs(C_computed - C_expected))
    
    if error > 1e-9:
        # Fallback to schoolbook if something went wrong (should not happen)
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
    
    # Convert to lists of floats (integers if possible)
    # The task allows integer/half-integer. Strassen uses integers.
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    # Optional: Convert to int if close to integer
    def to_int_if_close(lst):
        new_lst = []
        for row in lst:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    new_row.append(int(round(val)))
                else:
                    new_row.append(val)
            new_lst.append(new_row)
        return new_lst
        
    U_final = to_int_if_close(U_list)
    V_final = to_int_if_close(V_list)
    W_final = to_int_if_close(W_list)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
