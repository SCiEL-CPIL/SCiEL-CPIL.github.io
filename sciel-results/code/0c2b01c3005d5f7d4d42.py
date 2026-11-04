import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2 matrices
    # U maps A (4x1 vector) to 7 intermediates
    # V maps B (4x1 vector) to 7 intermediates
    # W maps 7 intermediates to C (4x1 vector)
    # Convention: U[i, r], V[j, r], W[k, r] such that C_k = sum_r U[i, r] V[j, r] W[k, r] ?
    # Wait, the task says:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Tensor T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}
    # Here i is row of A (0..3), j is col of A / row of B (0..3), k is col of B (0..3).
    # So U is 16x49, V is 16x49, W is 16x49.
    
    # Standard Strassen for 2x2:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # U factors (for A elements a11, a12, a21, a22):
    # m1: 1, 0, 0, 1
    # m2: 0, 0, 1, 1
    # m3: 1, 0, 0, 0
    # m4: 0, 0, 0, 1
    # m5: 1, 1, 0, 0
    # m6: -1, 0, 1, 0
    # m7: 0, 1, -1, 0
    
    U_blk = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    # V factors (for B elements b11, b12, b21, b22):
    # m1: 1, 1, 0, 0
    # m2: 1, 0, 0, 0
    # m3: 0, 1, 0, -1
    # m4: -1, 0, 1, 0
    # m5: 0, 0, 0, 1
    # m6: 1, 1, 0, 0
    # m7: 0, 0, 1, 1
    
    V_blk = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    # W factors (for C elements c11, c12, c21, c22):
    # c11: 1, 0, 0, 1, -1, 0, 1
    # c12: 0, 0, 1, 0, 1, 0, 0
    # c21: 0, 1, 0, 1, 0, 0, 0
    # c22: 1, -1, 1, 0, 0, 1, 0
    
    W_blk = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Construct 4x4 factors using Kronecker products
    # U_4 = U_2 \otimes U_2
    # V_4 = V_2 \otimes V_2
    # W_4 = W_2 \otimes W_2
    
    U = np.kron(U_blk, U_blk)
    V = np.kron(V_blk, V_blk)
    W = np.kron(W_blk, W_blk)
    
    # Verify correctness if time permits (optional, but good for debugging)
    # For 4x4, rank is 49.
    # T_{i,j,k} = sum_r U[i,r] V[j,r] W[k,r]
    # We can check a few random coordinates.
    
    # The evaluator expects a dictionary with lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
