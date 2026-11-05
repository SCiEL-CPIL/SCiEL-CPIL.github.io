"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit

    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
        rank = U.shape[1]
        # Check all valid (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    if time.time() < deadline - 15:
        # Construct the known rank-85 decomposition using Strassen on the 4x4x4 core
        # and schoolbook for the remaining 4x1 and 1x5 strips.
        
        # 4x4x4 Strassen decomposition
        # We split the 4x4 matrices into 2x2 blocks.
        # A = [[A11, A12], [A21, A22]] where Aij are 2x2
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # Strassen formulas for 4x4 using 2x2 blocks:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M is a 2x2 matrix product. We use the rank-7 Strassen decomposition for 2x2x2.
        
        # 2x2x2 Strassen factors (rank 7)
        # Indices for 2x2: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        # U, V, W are 4x7 matrices
        sU = np.zeros((4, 7))
        sV = np.zeros((4, 7))
        sW = np.zeros((4, 7))
        
        # Term 0: (a00+a11)(b00+b11)
        sU[0,0] = 1; sU[3,0] = 1
        sV[0,0] = 1; sV[3,0] = 1
        sW[0,0] = 1; sW[3,0] = 1
        
        # Term 1: (a10+a11)(b00-b10)
        sU[2,1] = 1; sU[3,1] = 1
        sV[0,1] = 1; sV[2,1] = -1
        sW[2,1] = 1; sW[3,1] = -1
        
        # Term 2: a00(b01-b11)
        sU[0,2] = 1
        sV[1,2] = 1; sV[3,2] = -1
        sW[1,2] = 1; sW[3,2] = -1
        
        # Term 3: a11(b10-b00)
        sU[3,3] = 1
        sV[2,3] = 1; sV[0,3] = -1
        sW[0,3] = 1; sW[2,3] = -1
        
        # Term 4: (a00+a01)(b10-b00)
        sU[0,4] = 1; sU[1,4] = 1
        sV[2,4] = 1; sV[0,4] = -1
        sW[0,4] = 1; sW[2,4] = 1
        
        # Term 5: (a10-a00)(b00+b01)
        sU[2,5] = 1; sU[0,5] = -1
        sV[0,5] = 1; sV[1,5] = 1
        sW[3,5] = 1
        
        # Term 6: (a01-a11)(b10+b11)
        sU[1,6] = 1; sU[3,6] = -1
        sV[2,6] = 1; sV[3,6] = 1
        sW[0,6] = 1

        # Now we build the 4x4x4 decomposition.
        # We have 7 Strassen terms, each contributing 7 scalar products. Total 49 terms.
        
        # We need to map the 2x2 block indices to global 4x4 indices.
        # Block (br, bc) of size 2x2 corresponds to global rows 2*br to 2*br+1, cols 2*bc to 2*bc+1.
        
        # Let's define the Strassen terms for 4x4 in terms of 2x2 block products.
        # M1 = (A11+A22)(B11+B22)
        # A part: A11 (coeff 1), A22 (coeff 1)
        # B part: B11 (coeff 1), B22 (coeff 1)
        # C part: C11 (coeff 1), C22 (coeff 1)
        
        # This is getting complex to implement generically. Let's just hardcode the 49 terms
        # by iterating through the 7 outer terms and 7 inner terms.
        
        # Outer terms:
        # 0: (A11+A22)(B11+B22) -> C11+1, C22+1
        # 1: (A21+A22)B11 -> C21+1, C22-1
        # 2: A11(B12-B22) -> C12+1, C22+1
        # 3: A22(B21-B11) -> C11+1, C21+1
        # 4: (A11+A12)B22 -> C11-1, C12+1
        # 5: (A21-A11)(B11+B12) -> C22+1
        # 6: (A12-A22)(B21+B22) -> C11+1
        
        # For each outer term, we have a linear combination of A blocks and B blocks.
        # Then we use the 2x2 Strassen factors to decompose the product of these combinations.
        
        # Let's define the coefficients for each outer term.
        # a_coeffs: dict (br, bc) -> coeff
        # b_coeffs: dict (br, bc) -> coeff
        # c_coeffs: dict (br, bc) -> coeff
        
        outer_terms = [
            # (a_coeffs, b_coeffs, c_coeffs)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7
        ]
        
        U_44 = np.zeros((16, 49))
        V_44 = np.zeros((16, 49))
        W_44 = np.zeros((16, 49))
        
        col = 0
        for a_c, b_c, c_c in outer_terms:
            # For each of the 7 2x2 Strassen terms
            for t in range(7):
                # u_t is a vector of 4 coeffs for the 2x2 product
                # We need to expand this into the global 4x4 indices.
                # The 2x2 product is between X = sum(a_c[blk] * A_blk) and Y = sum(b_c[blk] * B_blk)
                # The entries of X are x_ij = sum_{(br,bc) in a_c} a_c[(br,bc)] * A_blk[i,j]
                # Similarly for Y.
                
                # For a given 2x2 Strassen term t, we have:
                # u_t[i,j] for i,j in {0,1}
                # v_t[i,j] for i,j in {0,1}
                # w_t[i,j] for i,j in {0,1}
                
                # The global U coefficient for A[(2*br+i), (2*bc+j)] is:
                # sum_{(br,bc) in a_c} a_c[(br,bc)] * u_t[i,j]
                
                # Let's compute this.
                for i in range(2):
                    for j in range(2):
                        # U part
                        for (br, bc), coeff in a_c.items():
                            row = 2*br + i
                            col_idx = 2*bc + j
                            U_44[row*4 + col_idx, col] += coeff * sU[i*2+j, t]
                        
                        # V part
                        for (br, bc), coeff in b_c.items():
                            row = 2*br + i
                            col_idx = 2*bc + j
                            V_44[row*4 + col_idx, col] += coeff * sV[i*2+j, t]
                        
                        # W part
                        for (br, bc), coeff in c_c.items():
                            row = 2*br + i
                            col_idx = 2*bc + j
                            W_44[row*4 + col_idx, col] += coeff * sW[i*2+j, t]
                
                col += 1
        
        # Now we have the 4x4x4 decomposition.
        # We need to embed this into the 4x5x5 problem.
        
        # The 4x5x5 problem:
        # A is 4x5, B is 5x5, C is 4x5.
        # We split A into A_44 (4x4) and A_41 (4x1).
        # We split B into B_44 (4x4), B_41 (4x1), B_14 (1x4), B_11 (1x1).
        # We split C into C_44 (4x4) and C_41 (4x1).
        
        # C_44 = A_44 * B_44 + A_41 * B_14
        # C_41 = A_44 * B_41 + A_41 * B_11
        
        # We already have the decomposition for A_44 * B_44 (rank 49).
        # We need decompositions for:
        # A_41 * B_14 (4x1 * 1x4 = 4x4, rank 16)
        # A_44 * B_41 (4x4 * 4x1 = 4x1, rank 16)
        # A_41 * B_11 (4x1 * 1x1 = 4x1, rank 4)
        
        # Total rank: 49 + 16 + 16 + 4 = 85.
        
        U_total = np.zeros((20, 85))
        V_total = np.zeros((25, 85))
        W_total = np.zeros((20, 85))
        
        # Copy the 4x4x4 part
        # A_44 indices: rows 0-3, cols 0-3 of A (global 0-3, 0-3)
        # B_44 indices: rows 0-3, cols 0-3 of B (global 0-3, 0-3)
        # C_44 indices: rows 0-3, cols 0-3 of C (global 0-3, 0-3)
        
        # U_44 has shape (16, 49), indices (i*4+j) for i,j in 0-3
        # U_total has shape (20, 85), indices (i*5+j) for i in 0-3, j in 0-4
        for i in range(4):
            for j in range(4):
                U_total[i*5+j, 0:49] = U_44[i*4+j, 0:49]
        
        # V_44 has shape (16, 49), indices (i*4+j) for i,j in 0-3
        # V_total has shape (25, 85), indices (i*5+j) for i in 0-4, j in 0-4
        for i in range(4):
            for j in range(4):
                V_total[i*5+j, 0:49] = V_44[i*4+j, 0:49]
        
        # W_44 has shape (16, 49), indices (i*4+j) for i,j in 0-3 (k*n+i, n=4)
        # W_total has shape (20, 85), indices (k*4+i) for k in 0-4, i in 0-3
        for k in range(4):
            for i in range(4):
                W_total[k*4+i, 0:49] = W_44[k*4+i, 0:49]
        
        # Now add the other terms.
        col = 49
        
        # A_41 * B_14 (4x4, rank 16)
        # A_41 is 4x1, B_14 is 1x4.
        # A_41[i, 0] * B_14[0, j]
        for i in range(4):
            for j in range(4):
                U_total[i*5+4, col] = 1
                V_total[4*5+j, col] = 1
                W_total[j*4+i, col] = 1
                col += 1
        
        # A_44 * B_41 (4x1, rank 16)
        # A_44[i, k] * B_41[k, 0]
        for i in range(4):
            for k in range(4):
                U_total[i*5+k, col] = 1
                V_total[k*5+4, col] = 1
                W_total[4*4+i, col] = 1
                col += 1
        
        # A_41 * B_11 (4x1, rank 4)
        # A_41[i, 0] * B_11[0, 0]
        for i in range(4):
            U_total[i*5+4, col] = 1
            V_total[4*5+4, col] = 1
            W_total[4*4+i, col] = 1
            col += 1
        
        # Verify the decomposition
        if verify_decomposition(U_total, V_total, W_total, n, m, p):
            best_rank = 85
            best_U, best_V, best_W = U_total, V_total, W_total
        else:
            # If verification fails, fall back to schoolbook
            pass

    U = best_U.tolist()
    V = best_V.tolist()
    W = best_W.tolist()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
