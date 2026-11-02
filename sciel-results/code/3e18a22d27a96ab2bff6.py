import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Constructs the best known rank decomposition for 2x2 by 2x10 matrix multiplication.
    Base: rank-35 via 5 independent Strassen 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    def build_strassen_35():
        rank = 35
        U = [[0] * rank for _ in range(n * m)]      # (4, 35)
        V = [[0] * rank for _ in range(m * p)]      # (20, 35)
        W = [[0] * rank for _ in range(p * n)]      # (20, 35)

        # Strassen 2x2x2 decomposition
        # A = [[a00, a01], [a10, a11]] (U indices: 0=a00, 1=a01, 2=a10, 3=a11)
        # B = [[b00, b01], [b10, b11]] (V indices: 0=b00, 1=b01, 2=b10, 3=b11)
        # C = [[c00, c01], [c10, c11]] (W indices: 0=c00, 1=c01, 2=c10, 3=c11)
        
        # The 7 Strassen products and their contributions:
        # P1 = (a00+a11)*(b00+b11) -> c00+=1, c11+=1
        # P2 = (a10+a11)*b00       -> c01+=1, c11-=-1
        # P3 = a00*(b01-b11)       -> c10+=1, c11+=1
        # P4 = a11*(b10-b00)       -> c00+=1, c01+=1
        # P5 = (a00+a01)*b11       -> c00-=-1, c10+=1
        # P6 = (a10-a00)*(b00+b01) -> c11+=1
        # P7 = (a01-a11)*(b10+b11) -> c00+=1

        strassen_A = [
            [1, 0, 0, 1],   # P1: a00+a11
            [0, 0, 1, 1],   # P2: a10+a11
            [1, 0, 0, 0],   # P3: a00
            [0, 0, 0, 1],   # P4: a11
            [1, 1, 0, 0],   # P5: a00+a01
            [-1, 0, 1, 0],  # P6: a10-a00
            [0, 1, 0, -1],  # P7: a01-a11
        ]

        strassen_B = [
            [1, 0, 0, 1],   # P1: b00+b11
            [1, 0, 0, 0],   # P2: b00
            [0, 1, 0, -1],  # P3: b01-b11
            [-1, 0, 1, 0],  # P4: b10-b00
            [0, 0, 0, 1],   # P5: b11
            [1, 1, 0, 0],   # P6: b00+b01
            [0, 0, 1, 1],   # P7: b10+b11
        ]

        # W contributions: [c00, c01, c10, c11]
        w_contrib = [
            [1, 0, 0, 1],   # P1
            [0, 1, 0, -1],  # P2
            [0, 0, 1, 1],   # P3
            [1, 1, 0, 0],   # P4
            [-1, 0, 1, 0],  # P5
            [0, 0, 0, 1],   # P6
            [1, 0, 0, 0],   # P7
        ]

        for b in range(5):
            for r in range(7):
                col = b * 7 + r
                
                # U: A matrix indices (a00, a01, a10, a11) -> rows 0,1,2,3
                U[0][col] = strassen_A[r][0]
                U[1][col] = strassen_A[r][1]
                U[2][col] = strassen_A[r][2]
                U[3][col] = strassen_A[r][3]

                # V: B matrix for block b, columns 2b and 2b+1
                # B row 0: b_{0,2b} at index 2b, b_{0,2b+1} at index 2b+1
                # B row 1: b_{1,2b} at index 10+2b, b_{1,2b+1} at index 10+2b+1
                k0 = 2 * b
                k1 = 2 * b + 1
                V[0 * p + k0][col] = strassen_B[r][0]
                V[0 * p + k1][col] = strassen_B[r][1]
                V[1 * p + k0][col] = strassen_B[r][2]
                V[1 * p + k1][col] = strassen_B[r][3]

                # W: C matrix, W[k*n+i] where k is col in B, i is row in A
                # For block b: k=2b -> rows 4b, 4b+1; k=2b+1 -> rows 4b+2, 4b+3
                # W[4b][col] = c_{0,2b}, W[4b+1][col] = c_{1,2b}
                # W[4b+2][col] = c_{0,2b+1}, W[4b+3][col] = c_{1,2b+1}
                W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
                W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
                W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
                W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]

        return U, V, W, rank

    U, V, W, rank = build_strassen_35()

    return {"U": U, "V": V, "W": W}
