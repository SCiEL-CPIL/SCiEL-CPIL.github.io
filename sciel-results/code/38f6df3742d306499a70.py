import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    n, m, p = 2, 4, 8
    # We will construct a rank-56 decomposition
    # Based on Strassen for 2x2x2, extended to 2x2x8 and 2x4x8
    
    # We need to build U, V, W
    # U: (n*m, rank) = (8, R)
    # V: (m*p, rank) = (32, R)
    # W: (p*n, rank) = (16, R)
    
    # We'll use a list of tuples to build the matrices
    # Each term is (u_vec, v_vec, w_vec)
    # u_vec is length 8, v_vec is length 32, w_vec is length 16
    
    terms = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # This is inefficient, let's just build the matrices directly
        pass

    # Better: Initialize U, V, W as lists of lists of 0.0
    # We don't know R yet, but we know it's 56.
    R = 56
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: A[0,0], A[0,1], A[1,0], A[1,1]
    # In our global indexing:
    # A is part of U. U[i*m + j, c].
    # For a 2x2 block of A, say rows i0, i0+1 and cols j0, j0+1.
    # U indices: i0*m + j0, i0*m + j0+1, (i0+1)*m + j0, (i0+1)*m + j0+1.
    
    # B is part of V. V[j*p + k, c].
    # For a 2x2 block of B, rows j0, j0+1 and cols k0, k0+1.
    # V indices: j0*p + k0, j0*p + k0+1, (j0+1)*p + k0, (j0+1)*p + k0+1.
    
    # C is part of W. W[k*n + i, c].
    # For a 2x2 block of C, rows k0, k0+1 and cols i0, i0+1.
    # W indices: k0*n + i0, k0*n + i0+1, (k0+1)*n + i0, (k0+1)*n + i0+1.
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # A block: rows i0, i0+1; cols j0, j0+1
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + j0 + 1
        
        # B block: rows j0, j0+1; cols k0, k0+1
        b00 = j0 * p + k0
        b01 = j0 * p + k0 + 1
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + k0 + 1
        
        # C block: rows k0, k0+1; cols i0, i0+1
        c00 = k0 * n + i0
        c01 = k0 * n + i0 + 1
        c10 = (k0 + 1) * n + i0
        c11 = (k0 + 1) * n + i0 + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1 # C00 += M1, C11 += M1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1 # C10 += M2
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1 # C01 += M3
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1 # C00 += M4, C10 += M4
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1 # C00 -= M5, C01 += M5
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1 # C11 += M6
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1 # C00 += M7, C11 += M7
        col += 1
        
        return 7

    # We need to compute C = A B
    # A is 2x4, B is 4x8
    # Split A into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3)
    # Split B into B_L (2x8, rows 0,1) and B_R (2x8, rows 2,3)
    # C = A_L B_L + A_R B_R
    
    # A_L is at i=0,1; j=0,1.
    # B_L is at j=0,1; k=0..7.
    # This is a 2x2x8 multiplication.
    # We can do this by 4 Strassen multiplications for k=0,1; 2,3; 4,5; 6,7.
    
    for k0 in range(0, 8, 2):
        add_strassen(0, 0, k0) # A_L * B_L block k0, k0+1
        
    # A_R is at i=0,1; j=2,3.
    # B_R is at j=2,3; k=0..7.
    # This is a 2x2x8 multiplication.
    
    for k0 in range(0, 8, 2):
        add_strassen(0, 2, k0) # A_R * B_R block k0, k0+1
        
    # Check if col == R
    if col != R:
        # Fallback to schoolbook if something went wrong
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}
