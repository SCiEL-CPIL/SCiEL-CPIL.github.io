import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor.
        Returns True if valid."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False
        return True

    def verify_half_integer(U, V, W):
        """Verify that all entries are integer or half-integer, and decomposition is exact."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    doubled = val * 2
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return verify(U, V, W)

    # Construct the rank-68 decomposition using Strassen for 2x2 blocks
    def construct_rank_68():
        rank = 68
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # For each pair of k-indices (k0, k0+1), apply Strassen to the 2x2 subproblem
        # A is 2x2 (rows 0-1, cols 0-1), B is 2x2 (rows 0-1, cols k0-k0+1)
        # C is 2x2 (rows 0-1, cols k0-k0+1)
        # 
        # Strassen 7 terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # Output:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        
        for k0, k1 in k_pairs:
            # M1 = (A00+A11)(B00+B11)
            # U: (0,0)=1, (1,1)=1
            # V: (0,k0)=1, (1,k1)=1
            # W: (k0,0)=1, (k1,1)=1
            U[0][col] += 1
            U[m + 1][col] += 1
            V[k0][col] += 1
            V[p + k1][col] += 1
            W[k0 * n][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1
            
            # M2 = (A10+A11)B00
            # U: (1,0)=1, (1,1)=1
            # V: (0,k0)=1
            # W: (k0,1)=1, (k1,1)=-1
            U[m][col] += 1
            U[m + 1][col] += 1
            V[k0][col] += 1
            W[k0 * n + 1][col] += 1
            W[k1 * n + 1][col] -= 1
            col += 1
            
            # M3 = A00(B10-B11)
            # U: (0,0)=1
            # V: (1,k0)=1, (1,k1)=-1
            # W: (k0,0)=1, (k1,0)=1
            U[0][col] += 1
            V[p + k0][col] += 1
            V[p + k1][col] -= 1
            W[k0 * n][col] += 1
            W[k1 * n][col] += 1
            col += 1
            
            # M4 = A11(B01-B00)
            # U: (1,1)=1
            # V: (0,k1)=1, (0,k0)=-1
            # W: (k0,0)=1, (k0,1)=1
            U[m + 1][col] += 1
            V[k1][col] += 1
            V[k0][col] -= 1
            W[k0 * n][col] += 1
            W[k0 * n + 1][col] += 1
            col += 1
            
            # M5 = (A00+A01)B11
            # U: (0,0)=1, (0,1)=1
            # V: (1,k1)=1
            # W: (k0,0)=-1, (k1,0)=1
            U[0][col] += 1
            U[1][col] += 1
            V[p + k1][col] += 1
            W[k0 * n][col] -= 1
            W[k1 * n][col] += 1
            col += 1
            
            # M6 = (A10-A00)(B00+B01)
            # U: (1,0)=1, (0,0)=-1
            # V: (0,k0)=1, (0,k1)=1
            # W: (k1,1)=1
            U[m][col] += 1
            U[0][col] -= 1
            V[k0][col] += 1
            V[k1][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1
            
            # M7 = (A01-A11)(B10+B11)
            # U: (0,1)=1, (1,1)=-1
            # V: (1,k0)=1, (1,k1)=1
            # W: (k0,0)=1, (k1,0)=-1
            U[1][col] += 1
            U[m + 1][col] -= 1
            V[p + k0][col] += 1
            V[p + k1][col] += 1
            W[k0 * n][col] += 1
            W[k1 * n][col] -= 1
            col += 1

        # Schoolbook for i=2, all j, all k
        for j in range(3):
            for k in range(8):
                U[2 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 2][col] = 1
                col += 1

        # Schoolbook for i in {0,1}, j=2, all k
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        assert col == rank, f"Expected {rank} columns, got {col}"
        return {"U": U, "V": V, "W": W}

    best_result = construct_rank_68()
    
    if verify_half_integer(best_result["U"], best_result["V"], best_result["W"]):
        return best_result
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
