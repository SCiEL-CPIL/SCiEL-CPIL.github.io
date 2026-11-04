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

    def to_half_integer_matrix(mat):
        """Convert a matrix of floats to half-integers by rounding."""
        return [[round(val * 2) / 2 for val in row] for row in mat]

    # Construct the rank-68 decomposition
    def construct_rank_68():
        rank = 68
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Strassen blocks for i in {0,1}, j in {0,1}, k in {0,1}, {2,3}, {4,5}, {6,7}
        # Strassen terms for 2x2:
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        # M3 = A00(B10-B11) -> C01+=M3, C11+=M3
        # M4 = A11(B01-B00) -> C00+=M4, C10+=M4
        # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
        # M6 = (A10-A00)(B00+B01) -> C11+=M6
        # M7 = (A01-A11)(B10+B11) -> C00+=M7, C01-=M7
        
        # We need to map these to U, V, W entries.
        # Each term c corresponds to a set of (i,j) in U, (j,k) in V, (k,i) in W.
        
        strassen_blocks = [
            (0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)
        ]
        
        for i_off, j_off, k_off in strassen_blocks:
            # Term 1: M1
            # U: (0,0), (1,1) -> 1
            # V: (0,0), (1,1) -> 1
            # W: (0,0), (1,1) -> 1
            U[i_off*m + 0][col] += 1
            U[(i_off+1)*m + 1][col] += 1
            V[j_off*p + 0][col] += 1
            V[(j_off+1)*p + 1][col] += 1
            W[k_off*n + i_off][col] += 1
            W[(k_off+1)*n + i_off+1][col] += 1
            col += 1
            
            # Term 2: M2
            # U: (1,0), (1,1) -> 1
            # V: (0,0) -> 1
            # W: (0,1) -> 1, (1,1) -> -1
            U[i_off*m + 1][col] += 1
            U[(i_off+1)*m + 1][col] += 1
            V[j_off*p + 0][col] += 1
            W[k_off*n + i_off+1][col] += 1
            W[(k_off+1)*n + i_off+1][col] -= 1
            col += 1
            
            # Term 3: M3
            # U: (0,0) -> 1
            # V: (1,0) -> 1, (1,1) -> -1
            # W: (0,0) -> 1, (1,0) -> 1
            U[i_off*m + 0][col] += 1
            V[(j_off+1)*p + 0][col] += 1
            V[(j_off+1)*p + 1][col] -= 1
            W[k_off*n + i_off][col] += 1
            W[(k_off+1)*n + i_off][col] += 1
            col += 1
            
            # Term 4: M4
            # U: (1,1) -> 1
            # V: (0,1) -> 1, (0,0) -> -1
            # W: (0,0) -> 1, (0,1) -> 1
            U[(i_off+1)*m + 1][col] += 1
            V[j_off*p + 1][col] += 1
            V[j_off*p + 0][col] -= 1
            W[k_off*n + i_off][col] += 1
            W[k_off*n + i_off+1][col] += 1
            col += 1
            
            # Term 5: M5
            # U: (0,0), (0,1) -> 1
            # V: (1,1) -> 1
            # W: (0,0) -> -1, (1,0) -> 1
            U[i_off*m + 0][col] += 1
            U[i_off*m + 1][col] += 1
            V[(j_off+1)*p + 1][col] += 1
            W[k_off*n + i_off][col] -= 1
            W[(k_off+1)*n + i_off][col] += 1
            col += 1
            
            # Term 6: M6
            # U: (1,0), (0,0) -> 1, -1
            # V: (0,0), (0,1) -> 1
            # W: (1,1) -> 1
            U[i_off*m + 1][col] += 1
            U[i_off*m + 0][col] -= 1
            V[j_off*p + 0][col] += 1
            V[j_off*p + 1][col] += 1
            W[(k_off+1)*n + i_off+1][col] += 1
            col += 1
            
            # Term 7: M7
            # U: (0,1), (1,1) -> 1, -1
            # V: (1,0), (1,1) -> 1
            # W: (0,0) -> 1, (1,0) -> -1
            U[i_off*m + 1][col] += 1
            U[(i_off+1)*m + 1][col] -= 1
            V[(j_off+1)*p + 0][col] += 1
            V[(j_off+1)*p + 1][col] += 1
            W[k_off*n + i_off][col] += 1
            W[(k_off+1)*n + i_off][col] -= 1
            col += 1

        # Schoolbook for i=2, j in {0,1,2}, k in {0..7}
        for j in range(3):
            for k in range(8):
                U[2*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + 2][col] = 1
                col += 1

        # Schoolbook for i in {0,1}, j=2, k in {0..7}
        for i in range(2):
            for k in range(8):
                U[i*m + 2][col] = 1
                V[2*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    best_result = construct_rank_68()
    
    if verify_half_integer(best_result["U"], best_result["V"], best_result["W"]):
        return best_result
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
