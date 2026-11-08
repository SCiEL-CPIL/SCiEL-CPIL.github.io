"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{4,4,7} into M_{4,4,4} + M_{4,4,3}
    # Rank(M_{4,4,4}) <= 49 (Strassen)
    # Rank(M_{4,4,3}) <= 48 (Schoolbook)
    # Total Rank <= 97
    
    # 1. Strassen decomposition for 4x4x4
    # We need U1, V1, W1 for 4x4x4
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4x4, we apply this to 2x2 blocks of A and B.
    # A = [[A11, A12], [A21, A22]] where Aij are 2x2
    # B = [[B11, B12], [B21, B22]] where Bij are 2x2
    # C = [[C11, C12], [C21, C22]]
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The "rank" is the number of scalar multiplications.
    # Each Mx is a product of two 2x2 matrices, which requires 7 scalar multiplications (Strassen).
    # So 7 * 7 = 49 scalar multiplications.
    
    # We need to generate the U, V, W matrices for this decomposition.
    # U[i*m+j, c] = coefficient of A[i,j] in the c-th product
    # V[j*p+k, c] = coefficient of B[j,k] in the c-th product
    # W[k*n+i, c] = coefficient of C[i,k] in the c-th product
    
    # Let's build the 4x4x4 decomposition explicitly.
    U_444 = []
    V_444 = []
    W_444 = []
    
    # Helper to add a product term
    # term_A: list of (i, j, coeff)
    # term_B: list of (j, k, coeff)
    # term_C: list of (i, k, coeff)
    def add_term(term_A, term_B, term_C):
        idx = len(U_444)
        u = [0]*16
        v = [0]*16
        w = [0]*16
        for (i, j, c) in term_A:
            u[i*4+j] = c
        for (j, k, c) in term_B:
            v[j*4+k] = c
        for (i, k, c) in term_C:
            w[k*4+i] = c
        U_444.append(u)
        V_444.append(v)
        W_444.append(w)

    # Define the 7 Strassen terms for 2x2x2
    # A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    
    # M1: (A11+A22)(B11+B22)
    add_term([(0,0,1), (1,1,1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,1)], 
             [(0,0,1), (1,1,1), (0,1,-1), (1,0,-1)]) # C11=..., C22=...
    # Wait, the C coefficients are for the output of the 2x2x2 block.
    # The output C is 2x2.
    # C11 gets M1, C12 gets M3+M5, C21 gets M2+M4, C22 gets M1-M2+M3+M6.
    # So for the 4x4x4 tensor, the W matrix maps the scalar products to the 4x4 output.
    
    # This manual construction is error-prone.
    # I will use a simpler approach: Schoolbook for 4x4x4 is rank 16.
    # Wait, schoolbook for 4x4x4 is 4*4*4 = 64.
    # Strassen is 49.
    # Laderman is 21.
    
    # Let's just use Schoolbook for the whole 4x4x7 to be safe, but that's 112.
    # The current score is 1.0 (112).
    # I need to beat 112.
    
    # Let's use the 21+48=69 approach if I can find Laderman.
    # If not, I will use a recursive Strassen implementation.
    
    # Recursive Strassen for 4x4x4:
    # We can implement the 7 terms for 2x2x2, and each term is a 2x2x2 product.
    # So we need to "expand" the 2x2x2 Strassen into 4x4x4.
    
    # Let's do it.
    
    def strassen_2x2_terms():
        # Returns list of (A_coeffs, B_coeffs, C_coeffs) for 2x2x2
        # A_coeffs: dict {(i,j): coeff}
        # B_coeffs: dict {(j,k): coeff}
        # C_coeffs: dict {(i,k): coeff}
        terms = []
        # M1
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}))
        # M2
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):1}))
        # M3
        terms.append(({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}))
        # M4
        terms.append(({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,0):1, (0,0):-1}))
        # M5
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1}))
        # M6
        terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1, (0,0):-1}))
        # M7
        terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,0):-1, (0,0):1}))
        return terms

    terms_2x2 = strassen_2x2_terms()
    
    # Now expand to 4x4x4
    # A is 4x4, indexed (I,J) where I=0,1, J=0,1 are block indices.
    # Within block, i=0,1, j=0,1.
    # Global index (i, j) = (I*2+i, J*2+j).
    # Similarly for B and C.
    
    U_444 = []
    V_444 = []
    W_444 = []
    
    for A_c, B_c, C_c in terms_2x2:
        u = [0]*16
        v = [0]*16
        w = [0]*16
        for (I, J), coeff_A in A_c.items():
            for i in range(2):
                for j in range(2):
                    # The coefficient is the same for all 4 elements in the block?
                    # No, Strassen sums the blocks.
                    # M1 = (A11+A22)(B11+B22).
                    # A11 is a 2x2 matrix. A22 is a 2x2 matrix.
                    # So (A11+A22) is a 2x2 matrix with elements A11[i,j] + A22[i,j].
                    # So the coefficient for A[I*2+i, J*2+j] is coeff_A * 1.
                    u[(I*2+i)*4 + (J*2+j)] = coeff_A
        for (J, K), coeff_B in B_c.items():
            for j in range(2):
                for k in range(2):
                    v[(J*2+j)*4 + (K*2+k)] = coeff_B
        for (I, K), coeff_C in C_c.items():
            for i in range(2):
                for k in range(2):
                    w[(K*2+k)*4 + (I*2+i)] = coeff_C
        U_444.append(u)
        V_444.append(v)
        W_444.append(w)
        
    # Now we have 49 terms for 4x4x4.
    # We need 48 terms for 4x4x3 (schoolbook).
    U_443 = []
    V_443 = []
    W_443 = []
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u = [0]*16
                v = [0]*12
                w = [0]*12
                u[i*4+j] = 1
                v[j*4+k] = 1
                w[k*4+i] = 1
                U_443.append(u)
                V_443.append(v)
                W_443.append(w)
                
    # Combine
    # U: 16 x (49+48)
    # V: 16 x (49+48) -- Wait, V for 4x4x3 is 12 x 48.
    # We need to pad V for 4x4x4 to 12 columns? No, V for 4x4x4 is 16 x 49.
    # V for 4x4x3 is 12 x 48.
    # The total V matrix is 12 x 97? No.
    # The task is 4x4x7.
    # U is (4*4, rank) = (16, rank).
    # V is (4*7, rank) = (28, rank).
    # W is (7*4, rank) = (28, rank).
    
    # My 4x4x4 decomposition uses V of size (4*4, 49) = (16, 49).
    # My 4x4x3 decomposition uses V of size (4*3, 48) = (12, 48).
    # We need to combine them into V of size (28, 97).
    # The first 16 rows of V correspond to B's first 4 columns.
    # The last 12 rows of V correspond to B's last 3 columns.
    # The 4x4x4 part only uses the first 4 columns of B. So its V rows 16-27 are 0.
    # The 4x4x3 part only uses the last 3 columns of B. So its V rows 0-15 are 0.
    
    rank = 49 + 48
    U = [ [0]*rank for _ in range(16) ]
    V = [ [0]*rank for _ in range(28) ]
    W = [ [0]*rank for _ in range(28) ]
    
    # Fill U, V, W for 4x4x4 part (cols 0-48)
    for c in range(49):
        for i in range(16):
            U[i][c] = U_444[c][i]
        for j in range(16):
            V[j][c] = V_444[c][j]
        # W for 4x4x4 is 16 x 49.
        # But W for 4x4x7 is 28 x 97.
        # The 4x4x4 part computes C1 (4x4), which is the first 4 columns of the final C (4x7).
        # So W rows 0-15 correspond to C1.
        # W rows 16-27 correspond to C2 (4x3).
        for k in range(16):
            W[k][c] = W_444[c][k]
            
    # Fill U, V, W for 4x4x3 part (cols 49-96)
    for c in range(48):
        col_idx = 49 + c
        for i in range(16):
            U[i][col_idx] = U_443[c][i]
        # V for 4x4x3 is 12 x 48.
        # These correspond to B's columns 4,5,6 (indices 4,5,6).
        # So V rows 16-27.
        for j in range(12):
            V[16+j][col_idx] = V_443[c][j]
        # W for 4x4x3 is 12 x 48.
        # These correspond to C's columns 4,5,6 (indices 4,5,6).
        # So W rows 16-27.
        for k in range(12):
            W[16+k][col_idx] = W_443[c][k]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
