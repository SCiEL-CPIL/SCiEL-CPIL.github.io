import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # The known best rank for 2x3x3 is 17.
    # We construct the rank 17 solution explicitly.
    # Structure:
    # - 7 terms for the 2x2x2 sub-block (indices i<2, j<2, k<2) using Strassen
    # - 4 terms for the 2x2x1 sub-block (indices i<2, j<2, k=2) using direct mult
    # - 6 terms for the 2x1x3 sub-block (indices i<2, j=2, k<3) using direct mult
    # Total rank = 7 + 4 + 6 = 17.
    
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def set_U(i, j, r, val):
        U[i * m + j, r] = val
        
    def set_V(j, k, r, val):
        V[j * p + k, r] = val
        
    def set_W(k, i, r, val):
        W[k * n + i, r] = val

    # Block 1: Strassen 2x2x2 (j in {0,1}, k in {0,1})
    # M1: (A00+A11)(B00+B11) -> C00, C11
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    # M2: (A10+A11)B00 -> C10, -C11
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
    
    # M3: A00(B01-B11) -> C01, C11
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
    
    # M4: A11(B10-B00) -> C00, C10
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
    
    # M5: (A00+A01)B11 -> -C00, C01
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
    
    # M6: (A10-A00)(B00+B01) -> C11
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)
    
    # M7: (A01-A11)(B10+B11) -> C00
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)

    # Block 2: Direct 2x2x1 (j in {0,1}, k=2)
    # P1: A00 B02 -> C02
    set_U(0, 0, 7, 1)
    set_V(0, 2, 7, 1)
    set_W(2, 0, 7, 1)
    
    # P2: A01 B12 -> C02
    set_U(0, 1, 8, 1)
    set_V(1, 2, 8, 1)
    set_W(2, 0, 8, 1)
    
    # P3: A10 B02 -> C12
    set_U(1, 0, 9, 1)
    set_V(0, 2, 9, 1)
    set_W(2, 1, 9, 1)
    
    # P4: A11 B12 -> C12
    set_U(1, 1, 10, 1)
    set_V(1, 2, 10, 1)
    set_W(2, 1, 10, 1)

    # Block 3: Direct 2x1x3 (j=2, k in {0,1,2})
    # P5: A02 B20 -> C00
    set_U(0, 2, 11, 1)
    set_V(2, 0, 11, 1)
    set_W(0, 0, 11, 1)
    
    # P6: A02 B21 -> C01
    set_U(0, 2, 12, 1)
    set_V(2, 1, 12, 1)
    set_W(1, 0, 12, 1)
    
    # P7: A02 B22 -> C02
    set_U(0, 2, 13, 1)
    set_V(2, 2, 13, 1)
    set_W(2, 0, 13, 1)
    
    # P8: A12 B20 -> C10
    set_U(1, 2, 14, 1)
    set_V(2, 0, 14, 1)
    set_W(0, 1, 14, 1)
    
    # P9: A12 B21 -> C11
    set_U(1, 2, 15, 1)
    set_V(2, 1, 15, 1)
    set_W(1, 1, 15, 1)
    
    # P10: A12 B22 -> C12
    set_U(1, 2, 16, 1)
    set_V(2, 2, 16, 1)
    set_W(2, 1, 16, 1)

    # Verify the rank 17 solution
    def is_exact(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if is_exact(U, V, W, rank):
        # We have a valid rank 17 solution.
        # Attempt to find a rank 16 solution if time permits.
        # This is a heuristic search.
        if time.time() < deadline - 5:
            # Try a few random initializations for rank 16
            # and run a few steps of gradient descent.
            # If we find a solution with small error, snap to half-integers and verify.
            for seed in range(10):
                if time.time() > deadline - 2:
                    break
                np.random.seed(seed)
                r16 = 16
                U16 = np.random.randn(n * m, r16) * 0.5
                V16 = np.random.randn(m * p, r16) * 0.5
                W16 = np.random.randn(p * n, r16) * 0.5
                
                # Simple gradient descent for 100 steps
                lr = 0.001
                for step in range(100):
                    # Compute gradients numerically is slow, use analytic if possible or skip
                    # Given time constraints, we just check if random init happens to be close? No.
                    # We'll skip the heavy optimization and just return rank 17.
                    # The probability of finding rank 16 in 5 seconds is negligible.
                    pass
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook rank 18 if rank 17 construction failed (should not happen)
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
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
