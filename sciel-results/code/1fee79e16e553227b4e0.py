import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # shape (12, 44)
    V = [[0]*rank for _ in range(m*p)]  # shape (16, 44)
    W = [[0]*rank for _ in range(p*n)]  # shape (12, 44)
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        """Add 7-term Strassen decomposition for 2x2x2 block at given positions.
        a_r, a_c: top-left row/col of A block
        b_r, b_c: top-left row/col of B block  
        c_r, c_c: top-left row/col of C block
        """
        c = start_col
        # m1 = (a00 + a11)(b00 + b11)
        U[a_r*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        
        # m2 = (a10 + a11)b00
        U[(a_r+1)*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        W[c_c*n + (c_r+1)][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] -= 1
        c += 1
        
        # m3 = a00(b01 - b11)
        U[a_r*m + a_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] -= 1
        W[(c_c+1)*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        
        # m4 = a11(b10 - b00)
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + b_c][c] += 1
        V[b_r*p + b_c][c] -= 1
        W[c_c*n + c_r][c] += 1
        W[c_c*n + (c_r+1)][c] += 1
        c += 1
        
        # m5 = (a00 + a01)b11
        U[a_r*m + a_c][c] += 1
        U[a_r*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] -= 1
        W[(c_c+1)*n + c_r][c] += 1
        c += 1
        
        # m6 = (a10 - a00)(b00 + b01)
        U[(a_r+1)*m + a_c][c] += 1
        U[a_r*m + a_c][c] -= 1
        V[b_r*p + b_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        
        # m7 = (a01 - a11)(b10 + b11)
        U[a_r*m + (a_c+1)][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] -= 1
        V[(b_r+1)*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        c += 1
        
        return c

    def add_trivial_1x4x2(a_r, b_c, start_col):
        """Add 8-term trivial decomposition for 1x4x2 block.
        a_r: row of A (single row)
        b_c: starting column of B (2 columns)
        """
        c = start_col
        for j in range(4):
            U[a_r*m + j][c] += 1
            V[j*p + b_c][c] += 1
            W[b_c*n + a_r][c] += 1
            c += 1
            U[a_r*m + j][c] += 1
            V[j*p + (b_c+1)][c] += 1
            W[(b_c+1)*n + a_r][c] += 1
            c += 1
        return c

    # Left Side (cols 0,1 of C) - using A rows 0,1 and B cols 0,1
    col = add_strassen(0, 0, 0, 0, 0, 0, col)
    col = add_strassen(0, 2, 2, 0, 0, 0, col)
    col = add_trivial_1x4x2(2, 0, col)
    
    # Right Side (cols 2,3 of C) - using A rows 0,1 and B cols 2,3
    col = add_strassen(0, 0, 0, 2, 0, 2, col)
    col = add_strassen(0, 2, 2, 2, 0, 2, col)
    col = add_trivial_1x4x2(2, 2, col)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
