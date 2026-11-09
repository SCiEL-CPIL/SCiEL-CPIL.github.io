"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 4, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 exact tensor decomposition for 2x4 * 4x6 matrix multiplication
    using Strassen's algorithm on six 2x2x2 sub-blocks.
    
    The matrix multiplication tensor T satisfies:
      T[i, j, k] = 1 iff (i, j) is in A, (j, k) is in B, (i, k) is in C
    for the bilinear form C = A * B where A is 2x4, B is 4x6, C is 2x6.
    
    We decompose into 6 Strassen 2x2x2 blocks:
      For each row-block r in {0, 1} (splitting the 4 columns of A into 2 pairs):
        For each col-block c in {0, 1, 2} (splitting the 6 columns of B into 3 pairs):
          A_sub: rows [0,1], cols [2*r, 2*r+1]  (2x2)
          B_sub: rows [2*r, 2*r+1], cols [2*c, 2*c+1]  (2x2)
          C_sub: rows [0,1], cols [2*c, 2*c+1]  (2x2)
    
    Each Strassen block uses 7 multiplications instead of 4 (schoolbook),
    giving total rank = 6 * 7 = 42 < 48 (schoolbook).
    
    U has shape (n*m, rank) = (8, 42)
    V has shape (m*p, rank) = (24, 42)
    W has shape (p*n, rank) = (12, 42)
    
    Index conventions:
      U[i*m + j, col] where i in [0,n), j in [0,m)
      V[j*p + k, col] where j in [0,m), k in [0,p)
      W[k*n + i, col] where k in [0,p), i in [0,n)
    """
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2: compute [[a,b],[c,d]] * [[e,f],[g,h]] = [[s1, s2], [s3, s4]]
    # where:
    #   m1 = (a + d)(e + h)
    #   m2 = (c + d)e
    #   m3 = a(f - h)
    #   m4 = d(g - e)
    #   m5 = (a + b)h
    #   m6 = (c - a)(e + f)
    #   m7 = (b - d)(g + h)
    # 
    # Result:
    #   s1 = m1 + m4 - m5 + m7
    #   s2 = m3 + m5
    #   s3 = m2 + m4
    #   s4 = m1 - m2 + m3 + m6
    #
    # Each multiplication mj corresponds to a rank-1 term with coefficients:
    #   U entries: product of A coefficient
    #   V entries: product of B coefficient
    #   W entries: product of C coefficient
    #
    # For each Strassen block, we need to map local indices to global indices.
    
    col = 0
    for row_block in range(2):  # r: which pair of columns of A (and rows of B)
        for col_block in range(3):  # c: which pair of columns of B (and C)
            # Global indices:
            # A columns: j_a = 2*row_block + {0, 1}
            # B rows: j_b = 2*row_block + {0, 1}
            # B/C columns: k_c = 2*col_block + {0, 1}
            # A/C rows: i in {0, 1}
            
            # Strassen's 7 intermediate products:
            # Each product mj has:
            #   A-coefficients: for each A position (i, j_local), coefficient in A
            #   B-coefficients: for each B position (j_local, k_local), coefficient in B
            #   C-coefficients: for each C position (i, k_local), coefficient in C
            #
            # We'll define the 7 terms explicitly.
            
            # Local indices: i in {0,1}, j_l in {0,1}, k_l in {0,1}
            # A local position (i, j_l) -> global A row i, col 2*row_block + j_l
            # B local position (j_l, k_l) -> global B row 2*row_block + j_l, col 2*col_block + k_l
            # C local position (i, k_l) -> global C row i, col 2*col_block + k_l
            
            # Define the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs)
            # U_coeffs: dict from (i, j_l) to coefficient
            # V_coeffs: dict from (j_l, k_l) to coefficient
            # W_coeffs: dict from (i, k_l) to coefficient
            
            terms = [
                # m1 = (a + d)(e + h)
                {
                    'U': {(0,0): 1, (1,1): 1},
                    'V': {(0,0): 1, (1,1): 1},
                    'W': {(0,0): 1, (0,1): -1, (1,0): 1, (1,1): -1}  # m1 - m5 + m7 contribution... 
                    # Actually, let's handle this more carefully.
                },
            ]
            
            # Let me redefine this more carefully. Each of the 7 intermediate values 
            # mj is a scalar. The output C entries are linear combinations of mj.
            # For a rank-1 decomposition, each term corresponds to one scalar product.
            # The U matrix for that term has the A-coefficients, V has B-coefficients,
            # and W has the C-coefficients such that the product gives the contribution.
            #
            # For Strassen:
            #   m1 = (a+d)(e+h): U has a:1, d:1; V has e:1, h:1; W needs to contribute to C
            #   m2 = (c+d)e: U has c:1, d:1; V has e:1
            #   m3 = a(f-h): U has a:1; V has f:1, h:-1
            #   m4 = d(g-e): U has d:1; V has g:1, e:-1
            #   m5 = (a+b)h: U has a:1, b:1; V has h:1
            #   m6 = (c-a)(e+f): U has c:1, a:-1; V has e:1, f:1
            #   m7 = (b-d)(g+h): U has b:1, d:-1; V has g:1, h:1
            #
            # And C entries:
            #   s1 = m1 + m4 - m5 + m7
            #   s2 = m3 + m5
            #   s3 = m2 + m4
            #   s4 = m1 - m2 + m3 + m6
            #
            # So for each mj, the W coefficients are determined by how mj contributes to each s.
            #
            # Let's define each term as (U_dict, V_dict, W_dict) where each dict maps 
            # local indices to coefficients.

            strassen_terms = [
                # (U_coeffs, V_coeffs, W_coeffs)
                # m1
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                # m2
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,1): 1}),
                # m3
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
                # m4
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}),
                # m5
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
                # m6
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
                # m7
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ]
            
            for U_local, V_local, W_local in strassen_terms:
                # Map local to global and add to matrices
                for (i, j_l), u_val in U_local.items():
                    global_j = 2 * row_block + j_l
                    U[i * m + global_j][col] += u_val
                
                for (j_l, k_l), v_val in V_local.items():
                    global_j = 2 * row_block + j_l
                    global_k = 2 * col_block + k_l
                    V[global_j * p + global_k][col] += v_val
                
                for (i, k_l), w_val in W_local.items():
                    global_k = 2 * col_block + k_l
                    W[global_k * n + i][col] += w_val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
