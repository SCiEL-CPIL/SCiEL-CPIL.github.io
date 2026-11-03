<<<<<<< SEARCH
def add_laderman_block(U, V, W, i0, j0, k0, col_offset):
    # Laderman coefficients
    # C = A B
    # A is 3x3, B is 3x3
    # Indices: U[i,j], V[j,k], W[k,i]
    # Relative indices 0,1,2
    
    # ... (coefficients)
=======
def add_laderman_block(U, V, W, i0, j0, k0, col_offset):
    # Laderman coefficients
    # C = A B
    # A is 3x3, B is 3x3
    # Indices: U[i,j], V[j,k], W[k,i]
    # Relative indices 0,1,2
    
    # Laderman 15 products for 3x3 matrix multiplication
    # M1 = (a11 + a12 + a13 + a21 + a22 + a23 + a31 + a32 + a33) * (b11 - b12 - b13 - b21 - b22 - b23 - b31 - b32 - b33)
    # M2 = (a11 + a12 + a13 + a21 + a22 + a23) * (b11 - b12 - b21 - b22)
    # M3 = (a11 + a12 + a13) * (b11 - b12)
    # M4 = (a21 + a22 + a23) * (b21 - b22)
    # M5 = (a31 + a32 + a33) * (b31 - b32)
    # M6 = (a11 + a12) * (b11 - b12)
    # M7 = (a21 + a22) * (b21 - b22)
    # M8 = (a31 + a32) * (b31 - b32)
    # M9 = a11 * b11
    # M10 = a12 * b12
    # M11 = a21 * b21
    # M12 = a22 * b22
    # M13 = a31 * b31
    # M14 = a32 * b32
    # M15 = (a13 + a23 + a33) * (b13 + b23 + b33)
    
    # Extract 3x3 submatrices
    A = U[i0:i0+3, j0:j0+3]
    B = V[j0:j0+3, k0:k0+3]
    
    # Compute 15 Laderman products
    M1 = (A[0,0] + A[0,1] + A[0,2] + A[1,0] + A[1,1] + A[1,2] + A[2,0] + A[2,1] + A[2,2]) * \
         (B[0,0] - B[0,1] - B[0,2] - B[1,0] - B[1,1] - B[1,2] - B[2,0] - B[2,1] - B[2,2])
    M2 = (A[0,0] + A[0,1] + A[0,2] + A[1,0] + A[1,1] + A[1,2]) * \
         (B[0,0] - B[0,1] - B[1,0] - B[1,1])
    M3 = (A[0,0] + A[0,1] + A[0,2]) * (B[0,0] - B[0,1])
    M4 = (A[1,0] + A[1,1] + A[1,2]) * (B[1,0] - B[1,1])
    M5 = (A[2,0] + A[2,1] + A[2,2]) * (B[2,0] - B[2,1])
    M6 = (A[0,0] + A[0,1]) * (B[0,0] - B[0,1])
    M7 = (A[1,0] + A[1,1]) * (B[1,0] - B[1,1])
    M8 = (A[2,0] + A[2,1]) * (B[2,0] - B[2,1])
    M9 = A[0,0] * B[0,0]
    M10 = A[0,1] * B[0,1]
    M11 = A[1,0] * B[1,0]
    M12 = A[1,1] * B[1,1]
    M13 = A[2,0] * B[2,0]
    M14 = A[2,1] * B[2,1]
    M15 = (A[0,2] + A[1,2] + A[2,2]) * (B[0,2] + B[1,2] + B[2,2])
    
    # Combine to get 3x3 result C
    C00 = M1 - M2 + M3 - M6 + M9 - M10 + M15
    C01 = M1 - M2 + M3 + M6 - M9 + M10 - M15
    C02 = M1 + M2 - M3 - M4 - M5 + M6 + M7 + M8 - M9 - M11 - M13 + M15
    C10 = M1 - M2 - M3 + M4 + M5 - M6 + M7 - M8 + M9 + M11 + M13 - M15
    C11 = M1 - M2 - M3 - M4 - M5 - M6 - M7 - M8 - M9 - M11 - M13 + M15
    C12 = M1 + M2 + M3 + M4 + M5 - M6 - M7 - M8 + M9 + M11 + M13 - M15
    C20 = M1 + M2 - M3 - M4 - M5 - M6 - M7 - M8 - M9 - M11 - M13 + M15
    C21 = M1 + M2 - M3 - M4 - M5 + M6 + M7 + M8 + M9 + M11 + M13 - M15
    C22 = M1 - M2 + M3 + M4 + M5 + M6 + M7 + M8 - M9 - M11 - M13 + M15
    
    # Store results in W
    W[k0, i0 + col_offset] = C00
    W[k0, i0 + 1 + col_offset] = C01
    W[k0, i0 + 2 + col_offset] = C02
    W[k0 + 1, i0 + col_offset] = C10
    W[k0 + 1, i0 + 1 + col_offset] = C11
    W[k0 + 1, i0 + 2 + col_offset] = C12
    W[k0 + 2, i0 + col_offset] = C20
    W[k0 + 2, i0 + 1 + col_offset] = C21
    W[k0 + 2, i0 + 2 + col_offset] = C22
>>>>>>> REPLACE
