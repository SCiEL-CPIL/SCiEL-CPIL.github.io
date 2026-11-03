import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def construct_strassen():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen(col, j_base, k_base):
            U[0*m + (j_base+0)][col+0] += 1
            U[1*m + (j_base+1)][col+0] += 1
            U[1*m + (j_base+0)][col+1] += 1
            U[1*m + (j_base+1)][col+1] += 1
            U[0*m + (j_base+0)][col+2] += 1
            U[1*m + (j_base+1)][col+3] += 1
            U[0*m + (j_base+0)][col+4] += 1
            U[0*m + (j_base+1)][col+4] += 1
            U[1*m + (j_base+0)][col+5] += 1
            U[0*m + (j_base+0)][col+5] -= 1
            U[0*m + (j_base+1)][col+6] += 1
            U[1*m + (j_base+1)][col+6] -= 1

            V[(j_base+0)*p + (k_base+0)][col+0] += 1
            V[(j_base+1)*p + (k_base+1)][col+0] += 1
            V[(j_base+0)*p + (k_base+0)][col+1] += 1
            V[(j_base+0)*p + (k_base+1)][col+2] += 1
            V[(j_base+1)*p + (k_base+1)][col+2] -= 1
            V[(j_base+1)*p + (k_base+0)][col+3] += 1
            V[(j_base+0)*p + (k_base+0)][col+3] -= 1
            V[(j_base+1)*p + (k_base+1)][col+4] += 1
            V[(j_base+0)*p + (k_base+0)][col+5] += 1
            V[(j_base+0)*p + (k_base+1)][col+5] += 1
            V[(j_base+1)*p + (k_base+0)][col+6] += 1
            V[(j_base+1)*p + (k_base+1)][col+6] += 1

            W[(k_base+0)*n + 0][col+0] += 1
            W[(k_base+0)*n + 0][col+3] += 1
            W[(k_base+0)*n + 0][col+4] -= 1
            W[(k_base+0)*n + 0][col+6] += 1
            
            W[(k_base+1)*n + 0][col+2] += 1
            W[(k_base+1)*n + 0][col+4] += 1
            
            W[(k_base+0)*n + 1][col+1] += 1
            W[(k_base+0)*n + 1][col+3] += 1
            
            W[(k_base+1)*n + 1][col+0] += 1
            W[(k_base+1)*n + 1][col+1] -= 1
            W[(k_base+1)*n + 1][col+2] += 1
            W[(k_base+1)*n + 1][col+5] += 1

        def add_schoolbook(col, j_base, k_base):
            U[0*m + (j_base+0)][col+0] += 1
            U[0*m + (j_base+1)][col+1] += 1
            U[1*m + (j_base+0)][col+2] += 1
            U[1*m + (j_base+1)][col+3] += 1
            
            V[(j_base+0)*p + (k_base+0)][col+0] += 1
            V[(j_base+1)*p + (k_base+0)][col+1] += 1
            V[(j_base+0)*p + (k_base+0)][col+2] += 1
            V[(j_base+1)*p + (k_base+0)][col+3] += 1
            
            W[(k_base+0)*n + 0][col+0] += 1
            W[(k_base+0)*n + 0][col+1] += 1
            W[(k_base+0)*n + 1][col+2] += 1
            W[(k_base+0)*n + 1][col+3] += 1

        add_strassen(0, 0, 0)
        add_strassen(7, 0, 2)
        add_strassen(14, 0, 4)
        add_schoolbook(21, 0, 6)
        
        add_strassen(25, 2, 0)
        add_strassen(32, 2, 2)
        add_strassen(39, 2, 4)
        add_schoolbook(46, 2, 6)

        return {"U": U, "V": V, "W": W}

    def construct_poly():
        n_A = 11
        n_B = 10
        n_C = 20
        R = 20
        points = [i for i in range(1, R+1)]
        
        M = [[Fraction(points[r]) ** g for g in range(n_C)] for r in range(R)]
        
        M_aug = [row[:] + [Fraction(1 if i==j else 0) for j in range(R)] for i, row in enumerate(M)]
        
        for col in range(R):
            pivot = -1
            for row in range(col, R):
                if M_aug[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                return None
            
            M_aug[col], M_aug[pivot] = M_aug[pivot], M_aug[col]
            pivot_val = M_aug[col][col]
            for j in range(2*R):
                M_aug[col][j] /= pivot_val
            
            for row in range(R):
                if row != col:
                    factor = M_aug[row][col]
                    if factor != 0:
                        for j in range(2*R):
                            M_aug[row][j] -= factor * M_aug[col][j]
        
        M_inv = [row[R:] for row in M_aug]
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W_mat = [[0] * R for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                idx = i*m + j
                power = 7*i + j
                for r in range(R):
                    U[idx][r] = points[r] ** power
        
        for j in range(m):
            for k in range(p):
                idx = j*p + k
                power = 3-j + k
                for r in range(R):
                    V[idx][r] = points[r] ** power
        
        for k in range(p):
            for i in range(n):
                idx = k*n + i
                g = 7*i + k + 3
                for r in range(R):
                    W_mat[idx][r] = M_inv[g][r]
        
        # Convert to float
        U_float = [[float(val) for val in row] for row in U]
        V_float = [[float(val) for val in row] for row in V]
        W_float = [[float(val) for val in row] for row in W_mat]
        
        return {"U": U_float, "V": V_float, "W": W_float}
    
    try:
        res = construct_poly()
        if res:
            return res
    except Exception:
        pass
    
    return construct_strassen()
# EVOLVE-BLOCK-END
