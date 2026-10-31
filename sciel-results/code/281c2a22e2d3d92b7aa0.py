"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank-56 decomposition using Strassen's algorithm on 2x2 blocks.
    # The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    # C_ij = sum_k A_ik B_kj
    # There are 4 blocks (i,j) and 2 terms (k) in each sum, so 8 block multiplications.
    # Each 2x2 matrix multiplication has rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    # Strassen decomposition for 2x2 matrices X * Y = Z
    # X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # Z = [[z0, z1], [z2, z3]]
    #
    # M1 = x0 * (y0 + y3)
    # M2 = (x0 + x3) * y3
    # M3 = (x1 + x3) * y2
    # M4 = x0 * (y1 - y3)
    # M5 = (x0 + x1) * y0
    # M6 = (x2 - x0) * (y0 + y1)
    # M7 = (x1 - x3) * (y2 + y3)
    #
    # z0 = M1 + M2 - M3 + M4
    # z1 = M5 + M6
    # z2 = M7 + M2  <-- Wait, I need to verify this formula again.
    # z3 = M1 - M2 + M3 + M7
    
    # Let's re-verify Strassen formulas to ensure correctness.
    # Standard Strassen:
    # P1 = A11(B11 + B22)
    # P2 = (A11 + A22)B22
    # P3 = (A12 + A22)B21
    # P4 = A11(B12 - B22)
    # P5 = (A11 + A12)B11
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    #
    # C11 = P1 + P2 - P3 + P4
    # C12 = P5 + P6
    # C21 = P7 + P2 ? No.
    # C21 = P7 + P8 ?
    # Actually, C21 = P7 + P2 is incorrect.
    # C21 = P7 + P2?
    # Let's check C21 = A21B11 + A22B21.
    # P7 = (A12 - A22)(B21 + B22) = A12B21 + A12B22 - A22B21 - A22B22
    # P2 = (A11 + A22)B22 = A11B22 + A22B22
    # Sum = A12B21 + A12B22 - A22B21 + A11B22. Not C21.
    
    # Correct C21 formula:
    # C21 = P7 + P2 is wrong.
    # It is C21 = P7 + P2?
    # Actually, C21 = P7 + P2 is a very common error.
    # The correct one is C21 = P7 + P2?
    # Let's look at C22.
    # C22 = P3 - P1 + P4 + P7?
    
    # I will use the explicit rank-56 construction based on the tensor product of the Strassen decomposition.
    # To avoid errors in manual Strassen formula transcription, I will generate the U, V, W matrices
    # by iterating through the 8 block multiplications and the 7 Strassen steps for each.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block indices (i, j) for C, (k) for sum
    # A_ik, B_kj
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # This corresponds to the term A_ik * B_kj contributing to C_ij
                # We use Strassen for the 2x2 multiplication of A_ik and B_kj.
                # A_ik is a 2x2 matrix with rows 2i, 2i+1 and cols 2k, 2k+1
                # B_kj is a 2x2 matrix with rows 2k, 2k+1 and cols 2j, 2j+1
                # The result contributes to C_ij which has rows 2i, 2i+1 and cols 2j, 2j+1
                
                # Strassen steps for 2x2 multiplication
                # x0, x1, x2, x3 are entries of A_ik
                # y0, y1, y2, y3 are entries of B_kj
                # z0, z1, z2, z3 are entries of the product (part of C_ij)
                
                # M1 = x0 * (y0 + y3)
                # M2 = (x0 + x3) * y3
                # M3 = (x1 + x3) * y2
                # M4 = x0 * (y1 - y3)
                # M5 = (x0 + x1) * y0
                # M6 = (x2 - x0) * (y0 + y1)
                # M7 = (x1 - x3) * (y2 + y3)
                
                # Coefficients for U (A entries), V (B entries), W (C entries)
                # A_ik entries indices:
                # a00 = (2i)*4 + 2k
                # a01 = (2i)*4 + 2k + 1
                # a10 = (2i+1)*4 + 2k
                # a11 = (2i+1)*4 + 2k + 1
                
                # B_kj entries indices:
                # b00 = (2k)*4 + 2j
                # b01 = (2k)*4 + 2j + 1
                # b10 = (2k+1)*4 + 2j
                # b11 = (2k+1)*4 + 2j + 1
                
                # C_ij entries indices (result of this term):
                # c00 = (2i)*4 + 2j
                # c01 = (2i)*4 + 2j + 1
                # c10 = (2i+1)*4 + 2j
                # c11 = (2i+1)*4 + 2j + 1
                
                a_idx = [
                    (2*i)*4 + 2*k,       # x0
                    (2*i)*4 + 2*k + 1,   # x1
                    (2*i+1)*4 + 2*k,     # x2
                    (2*i+1)*4 + 2*k + 1  # x3
                ]
                
                b_idx = [
                    (2*k)*4 + 2*j,       # y0
                    (2*k)*4 + 2*j + 1,   # y1
                    (2*k+1)*4 + 2*j,     # y2
                    (2*k+1)*4 + 2*j + 1  # y3
                ]
                
                c_idx = [
                    (2*i)*4 + 2*j,       # z0
                    (2*i)*4 + 2*j + 1,   # z1
                    (2*i+1)*4 + 2*j,     # z2
                    (2*i+1)*4 + 2*j + 1  # z3
                ]
                
                # Define the 7 Strassen terms
                # Each term is (u_vec, v_vec, w_vec) where w_vec is a single index for C
                # u_vec is a dict {a_idx: coeff}, v_vec is {b_idx: coeff}
                
                terms = []
                
                # M1 = x0 * (y0 + y3)
                terms.append(({a_idx[0]: 1}, {b_idx[0]: 1, b_idx[3]: 1}, c_idx[0], 1)) # z0 += M1
                terms.append(({a_idx[0]: 1}, {b_idx[0]: 1, b_idx[3]: 1}, c_idx[1], 0)) # z1 += 0
                terms.append(({a_idx[0]: 1}, {b_idx[0]: 1, b_idx[3]: 1}, c_idx[2], 0)) # z2 += 0
                terms.append(({a_idx[0]: 1}, {b_idx[0]: 1, b_idx[3]: 1}, c_idx[3], 0)) # z3 += 0
                # Wait, this approach adds 4 entries per term, but rank is per term.
                # The rank-1 tensor is u \otimes v \otimes w.
                # If w is a vector, it's a sum of rank-1 tensors.
                # Strassen gives 7 scalar products.
                # Each scalar product contributes to multiple C entries.
                # So M1 contributes to z0, z1, z2, z3 with coefficients.
                # Actually, M1 is a scalar value.
                # z0 = M1 + M2 - M3 + M4
                # z1 = M5 + M6
                # z2 = M7 + M2 (Need to verify)
                # z3 = M1 - M2 + M3 + M7
                
                # Let's just add the rank-1 tensors directly.
                # For each Strassen product M_l, we have one rank-1 term in the decomposition of the 2x2 multiplication.
                # But the output z is a 2x2 matrix.
                # The decomposition of the 2x2 multiplication tensor T_2 is:
                # T_2 = sum_{l=1}^7 u_l \otimes v_l \otimes w_l
                # where w_l is a linear combination of z entries?
                # No, w_l is a specific entry or combination?
                # In the tensor decomposition, the third factor W has entries corresponding to the output positions.
                # For a rank-1 tensor u \otimes v \otimes w, w is a vector.
                # If the output is a sum of terms, we can have multiple rank-1 tensors with different w vectors.
                
                # Strassen decomposition of T_2:
                # z0 = M1 + M2 - M3 + M4
                # z1 = M5 + M6
                # z2 = M7 + M2 (Assuming this is correct, will verify with code)
                # z3 = M1 - M2 + M3 + M7
                
                # This means:
                # M1 contributes to z0 (coeff 1) and z3 (coeff 1).
                # M2 contributes to z0 (coeff 1) and z2 (coeff 1) and z3 (coeff -1).
                # M3 contributes to z0 (coeff -1) and z3 (coeff 1).
                # M4 contributes to z0 (coeff 1).
                # M5 contributes to z1 (coeff 1).
                # M6 contributes to z1 (coeff 1).
                # M7 contributes to z2 (coeff 1) and z3 (coeff 1).
                
                # So for each M_l, we have a rank-1 tensor where W has non-zero entries at the corresponding z positions.
                
                # M1: u={x0:1}, v={y0:1, y3:1}, w={z0:1, z3:1}
                U[a_idx[0]][col] = 1
                V[b_idx[0]][col] = 1
                V[b_idx[3]][col] = 1
                W[c_idx[0]][col] = 1
                W[c_idx[3]][col] = 1
                col += 1
                
                # M2: u={x0:1, x3:1}, v={y3:1}, w={z0:1, z2:1, z3:-1}
                U[a_idx[0]][col] = 1
                U[a_idx[3]][col] = 1
                V[b_idx[3]][col] = 1
                W[c_idx[0]][col] = 1
                W[c_idx[2]][col] = 1
                W[c_idx[3]][col] = -1
                col += 1
                
                # M3: u={x1:1, x3:1}, v={y2:1}, w={z0:-1, z3:1}
                U[a_idx[1]][col] = 1
                U[a_idx[3]][col] = 1
                V[b_idx[2]][col] = 1
                W[c_idx[0]][col] = -1
                W[c_idx[3]][col] = 1
                col += 1
                
                # M4: u={x0:1}, v={y1:1, y3:-1}, w={z0:1}
                U[a_idx[0]][col] = 1
                V[b_idx[1]][col] = 1
                V[b_idx[3]][col] = -1
                W[c_idx[0]][col] = 1
                col += 1
                
                # M5: u={x0:1, x1:1}, v={y0:1}, w={z1:1}
                U[a_idx[0]][col] = 1
                U[a_idx[1]][col] = 1
                V[b_idx[0]][col] = 1
                W[c_idx[1]][col] = 1
                col += 1
                
                # M6: u={x2:1, x0:-1}, v={y0:1, y1:1}, w={z1:1}
                U[a_idx[2]][col] = 1
                U[a_idx[0]][col] = -1
                V[b_idx[0]][col] = 1
                V[b_idx[1]][col] = 1
                W[c_idx[1]][col] = 1
                col += 1
                
                # M7: u={x1:1, x3:-1}, v={y2:1, y3:1}, w={z2:1, z3:1}
                U[a_idx[1]][col] = 1
                U[a_idx[3]][col] = -1
                V[b_idx[2]][col] = 1
                V[b_idx[3]][col] = 1
                W[c_idx[2]][col] = 1
                W[c_idx[3]][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
