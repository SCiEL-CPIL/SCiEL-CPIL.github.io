"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 8 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before 150s kill
    
    def schoolbook():
        """Return the schoolbook decomposition."""
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

    def verify_full(U, V, W, rank, n, m, p):
        """Verify full tensor decomposition."""
        for i in range(n):
            for k in range(p):
                for i_p in range(n):
                    for j_p in range(m):
                        for j_pp in range(m):
                            for k_pp in range(p):
                                total = 0
                                for c in range(rank):
                                    total += U[i_p * m + j_p][c] * V[j_pp * p + k_pp][c] * W[k_pp * n + i][c]
                                expected = 1 if (i_p == i and k_pp == k and j_p == j_pp) else 0
                                if total != expected:
                                    return False
        return True

    def verify_4(U, V, W, rank):
        """Verify decomposition for M_{2,3,4}."""
        return verify_full(U, V, W, rank, 2, 3, 4)

    def lift_4_to_8(U4, V4, W4, R4):
        """Lift M_{2,3,4} decomposition to M_{2,3,8}."""
        R8 = 2 * R4
        U8 = [[0] * R8 for _ in range(6)]
        V8 = [[0] * R8 for _ in range(24)]
        W8 = [[0] * R8 for _ in range(16)]
        
        for c in range(R4):
            # First half (k < 4)
            for i in range(2):
                for j in range(3):
                    U8[i * 3 + j][c] = U4[i * 3 + j][c]
            for j in range(3):
                for k in range(4):
                    V8[j * 8 + k][c] = V4[j * 4 + k][c]
            for k in range(4):
                for i in range(2):
                    W8[k * 2 + i][c] = W4[k * 2 + i][c]
                    
            # Second half (k >= 4)
            for i in range(2):
                for j in range(3):
                    U8[i * 3 + j][c + R4] = U4[i * 3 + j][c]
            for j in range(3):
                for k in range(4, 8):
                    V8[j * 8 + k][c + R4] = V4[j * 4 + (k - 4)][c]
            for k in range(4, 8):
                for i in range(2):
                    W8[k * 2 + i][c + R4] = W4[(k - 4) * 2 + i][c]
                    
        return {"U": U8, "V": V8, "W": W8}

    # Search for M_{2,3,4} decomposition
    random.seed(42)
    best_result = None
    
    # Try ranks for M_{2,3,4}
    # Schoolbook is 24. Try 20, 21, 22, 23.
    for R4 in [20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
            
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            U4 = [[0] * R4 for _ in range(6)]
            V4 = [[0] * R4 for _ in range(12)]
            W4 = [[0] * R4 for _ in range(8)]
            
            # Random sparse initialization
            for c in range(R4):
                # Pick random non-zero entries
                for _ in range(random.randint(2, 4)):
                    idx = random.randint(0, 5)
                    U4[idx][c] = random.choice([-1, 1])
                for _ in range(random.randint(2, 4)):
                    idx = random.randint(0, 11)
                    V4[idx][c] = random.choice([-1, 1])
                for _ in range(random.randint(2, 4)):
                    idx = random.randint(0, 7)
                    W4[idx][c] = random.choice([-1, 1])
            
            if verify_4(U4, V4, W4, R4):
                result = lift_4_to_8(U4, V4, W4, R4)
                # Final verification of the lifted result
                if verify_full(result["U"], result["V"], result["W"], 2 * R4, 2, 3, 8):
                    return result

    # If no better solution found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
