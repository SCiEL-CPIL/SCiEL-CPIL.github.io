"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    # We decompose 4x4 * 4x6 as:
    # 4x4 * 4x4 (Strassen, rank 49) + 4x4 * 4x2 (schoolbook, rank 32)
    # Total rank = 81
    
    # Part 1: 4x4 * 4x4 using Strassen (rank 49)
    # Split into 2x2 blocks of 2x2 matrices
    # A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    # Strassen products M1..M7, each is 2x2*2x2 (rank 4 schoolbook or rank 7 Strassen)
    # Using Strassen recursively: each 2x2*2x2 has rank 7
    # Total: 7*7 = 49
    
    # For 2x2 * 2x2 Strassen rank 7:
    # u, v, w vectors for the 7 terms
    # A = [[a1,a2],[a3,a4]], B = [[b1,b2],[b3,b4]]
    # C11 = a1*b1 + a2*b3
    # C12 = a1*b2 + a2*b4
    # C21 = a3*b1 + a4*b3
    # C22 = a3*b2 + a4*b4
    #
    # Strassen:
    # M1 = (a1)(b2-b3)
    # M2 = (a1+a2)(b4)
    # M3 = (a3+a4)(b1)
    # M4 = (a4)(b3-b1)
    # M5 = (a1+a4)(b1+b4)
    # M6 = (a2-a4)(b3+b4)
    # M7 = (a1-a3)(b1+b2)
    # C11 = M5+M4-M2+M6
    # C12 = M1+M5
    # C21 = M3+M5
    # C22 = M1+M2-M4+M7
    
    # For the 4x4*4x4 part, we use 7 Strassen terms at top level,
    # each involving a 2x2*2x2 multiplication (rank 7 with inner Strassen)
    # So 7*7 = 49 rank-1 terms for 4x4*4x4
    
    # For the 4x4*4x2 part, schoolbook: 4*4*2 = 32 rank-1 terms
    
    total_rank = 49 + 32
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 81
    V = [[0]*total_rank for _ in range(m*p)]  # 24 x 81
    W = [[0]*total_rank for _ in range(p*n)]  # 24 x 81
    
    # --- Part 1: 4x4 * 4x4 using double Strassen ---
    # A indices: row i (0-3), col j (0-3) -> U[i*4+j]
    # B indices: row j (0-3), col k (0-3) -> V[j*4+k]
    # C indices: row i (0-3), col k (0-3) -> W[k*4+i] (trace convention: W[k*n+i])
    
    # 2x2 Strassen rank-1 terms for inner 2x2*2x2:
    # For multiplying 2x2 X * 2x2 Y:
    # X = [[x0,x1],[x2,x3]], Y = [[y0,y1],[y2,y3]]
    # 7 terms:
    s2_strassen = [
        # (u_coeffs, v_coeffs, w_coeffs) for 2x2*2x2
        # u: 4 components (x0,x1,x2,x3), v: 4 components (y0,y1,y2,y3), w: 4 components (z0,z1,z2,z3)
        # M1 = x0 * (y1-y2)
        ([1,0,0,0], [0,1,-1,0], [0,1,0,0]),   # contributes to C12 (z1)
        # M2 = (x0+x1) * y3
        ([1,1,0,0], [0,0,0,1], [1,-1,0,0]),   # C11+=1, C22+=1 via M1+M2-M4+M7... 
        # Wait, let me be more careful.
    ]
    
    # Let me reconsider. For 2x2*2x2:
    # C = X*Y where X,Y are 2x2
    # C[0,0] = x0*y0 + x1*y2
    # C[0,1] = x0*y1 + x1*y3
    # C[1,0] = x2*y0 + x3*y2
    # C[1,1] = x2*y1 + x3*y3
    #
    # Strassen 7 products:
    # M1 = x0 * (y1 - y2)     -> contributes to C[0,1]
    # M2 = (x0+x1) * y3       -> contributes to C[0,0] and C[1,1]
    # M3 = (x2+x3) * y0       -> contributes to C[1,0]
    # M4 = x3 * (y2 - y0)     -> contributes to C[1,0] and C[1,1]  
    # M5 = (x0+x3) * (y0+y3)  -> contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # M6 = (x1-x3) * (y2+y3)  -> contributes to C[0,0], C[1,1]
    # M7 = (x0-x2) * (y0+y1)  -> contributes to C[1,1]
    #
    # C[0,0] = M5 + M4 - M2 + M6
    # C[0,1] = M1 + M5
    # C[1,0] = M3 + M5
    # C[1,1] = M1 + M2 - M4 + M7
    #
    # So the 7 rank-1 terms for 2x2*2x2:
    # Each term s has u (4-dim for X), v (4-dim for Y), w (4-dim for C)
    
    def strassen_2x2_terms():
        """Return list of (u, v, w) tuples for 2x2*2x2 Strassen."""
        terms = []
        # M1 = x0 * (y1 - y2)
        # C[0,1] += 1*M1
        terms.append(([1,0,0,0], [0,1,-1,0], [0,1,0,0]))
        # M2 = (x0+x1) * y3
        # C[0,0] += -1*M2, C[1,1] += 1*M2
        terms.append(([1,1,0,0], [0,0,0,1], [-1,0,0,1]))
        # M3 = (x2+x3) * y0
        # C[1,0] += 1*M3
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        # M4 = x3 * (y2 - y0)
        # C[0,0] += 1*M4, C[1,0] += 0, C[1,1] += -1*M4
        terms.append(([0,0,0,1], [1,0,-1,0], [1,0,0,-1]))
        # M5 = (x0+x3) * (y0+y3)
        # C[0,0] += 1, C[0,1] += 1, C[1,0] += 1, C[1,1] += 0
        terms.append(([1,0,0,1], [1,0,0,1], [1,1,1,0]))
        # M6 = (x1-x3) * (y2+y3)
        # C[0,0] += 1, C[1,1] += 1
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1]))
        # M7 = (x0-x2) * (y0+y1)
        # C[1,1] += 1
        terms.append(([1,0,-1,0], [1,1,0,0], [0,0,0,1]))
        return terms
    
    s2 = strassen_2x2_terms()
    
    # For 4x4 * 4x4:
    # A = [[A11, A12],[A21, A22]] where each Ai is 2x2
    # B = [[B11, B12],[B21, B22]] where each Bj is 2x2
    # Top-level Strassen:
    # M1 = A11 * (B12 - B22)
    # M2 = (A11+A12) * B22
    # M3 = (A21+A22) * B11
    # M4 = A22 * (B21 - B11)
    # M5 = (A11+A22) * (B11+B22)
    # M6 = (A12-A22) * (B21+B22)
    # M7 = (A11-A21) * (B11+B12)
    #
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M5
    # C21 = M3 + M5
    # C22 = M1 + M2 - M4 + M7
    #
    # Each Mi is a 2x2 * 2x2 product, decomposed into 7 rank-1 terms.
    # So total 7*7 = 49 rank-1 terms.
    
    # Top-level Strassen coefficients:
    # Each top term has:
    #   u_block: 4x2 matrix (2x2 blocks of 2x2) - but actually for the 4x4*4x4,
    #   the "X" in 2x2*2x2 is a 2x2 matrix from A, "Y" is a 2x2 matrix from B.
    #   
    # Let me index: A is 4x4, A[i,j] for i,j in 0..3
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    #
    # For top-level term M_t with coefficient pattern:
    # M_t = X_t * Y_t where X_t is linear combo of A-blocks, Y_t is linear combo of B-blocks
    # Then decompose X_t * Y_t using inner Strassen.
    
    # Top-level Strassen terms (X_coeff for A blocks, Y_coeff for B blocks):
    # A blocks: A11, A12, A21, A22 (each 2x2)
    # B blocks: B11, B12, B21, B22 (each 2x2)
    top_terms = [
        # (a11_coeff, a12_coeff, a21_coeff, a22_coeff, b11_coeff, b12_coeff, b21_coeff, b22_coeff)
        ([1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [1,0,0,0], [0,0,0,0], [-1,0,0,0]),  # M1: A11*(B12-B22)
        ([1,0,0,0], [1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [1,0,0,0]),  # M2: (A11+A12)*B22
        ([0,0,0,0], [0,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]),  # M3: (A21+A22)*B11
        ([0,0,0,0], [0,0,0,0], [0,0,0,0], [1,0,0,0], [0,0,0,0], [0,0,0,0], [1,0,0,0], [-1,0,0,0]),  # M4: A22*(B21-B11)
        ([1,0,0,0], [0,0,0,0], [0,0,0,0], [1,0,0,0], [1,0,0,0], [0,0,0,0], [0,0,0,0], [1,0,0,0]),  # M5: (A11+A22)*(B11+B22)
        ([0,0,0,0], [1,0,0,0], [0,0,0,0], [-1,0,0,0], [0,0,0,0], [0,0,0,0], [1,0,0,0], [1,0,0,0]),  # M6: (A12-A22)*(B21+B22)
        ([1,0,0,0], [0,0,0,0], [-1,0,0,0], [0,0,0,0], [1,0,0,0], [1,0,0,0], [0,0,0,0], [0,0,0,0]),  # M7: (A11-A21)*(B11+B12)
    ]
    
    # C block contributions:
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M5
    # C21 = M3 + M5
    # C22 = M1 + M2 - M4 + M7
    # So for output block Ci, the coefficients from top terms:
    c_block_coeffs = {
        (0,0): [0, -1, 0, 1, 1, 1, 0],  # C11: M1*M1coeff=0, M2=-1, M3=0, M4=1, M5=1, M6=1, M7=0
        (0,1): [1, 0, 0, 0, 1, 0, 0],   # C12: M1=1, M5=1
        (1,0): [0, 0, 1, 0, 1, 0, 0],   # C21: M3=1, M5=1
        (1,1): [1, 1, 0, -1, 0, 0, 1],  # C22: M1=1, M2=1, M4=-1, M7=1
    }
    
    # A block row/col to full index:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    a_block_pos = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}  # block_idx -> (row_offset, col_offset)
    
    # B block:
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    b_block_pos = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
    
    # C block:
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    c_block_pos = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
    
    col = 0
    for t in range(7):  # top-level Strassen term
        a_coeffs = top_terms[t][0]  # 4 values for A11-A22 (but each is scalar coeff for the 2x2 block)
        b_coeffs = top_terms[t][1]  # Wait, I structured top_terms wrong.
        
        # Let me restructure: top_terms[t] = (a_block_coeffs, b_block_coeffs)
        # where a_block_coeffs = [c11, c12, c21, c22] for A blocks
        # b_block_coeffs = [c11, c12, c21, c22] for B blocks
    
    # Let me redo top_terms properly:
    top_terms = [
        # (a11, a12, a21, a22, b11, b12, b21, b22)
        (1, 0, 0, 0, 0, 1, 0, -1),   # M1: A11*(B12-B22)
        (1, 1, 0, 0, 0, 0, 0, 1),    # M2: (A11+A12)*B22
        (0, 0, 1, 1, 1, 0, 0, 0),    # M3: (A21+A22)*B11
        (0, 0, 0, 1, 0, 0, 1, -1),   # M4: A22*(B21-B11)
        (1, 0, 0, 1, 1, 0, 0, 1),    # M5: (A11+A22)*(B11+B22)
        (0, 1, 0, -1, 0, 0, 1, 1),   # M6: (A12-A22)*(B21+B22)
        (1, 0, -1, 0, 1, 1, 0, 0),   # M7: (A11-A21)*(B11+B12)
    ]
    
    col = 0
    for t in range(7):
        a_c = top_terms[t][:4]   # coeffs for A11, A12, A21, A22
        b_c = top_terms[t][4:]   # coeffs for B11, B12, B21, B22
        
        for inner_idx in range(7):  # inner Strassen term
            u_inner, v_inner, w_inner = s2[inner_idx]
            
            # Build full u (16-dim for A), v (16-dim for B first 4 cols), w (16-dim for C first 4 cols)
            u_full = [0]*16
            v_full = [0]*16
            w_full = [0]*16
            
            # u: for each A block (2x2), multiply by a_c[block_idx] * u_inner
            for bi in range(4):
                if a_c[bi] != 0:
                    r0, c0 = a_block_pos[bi]
                    for ii in range(2):
                        for jj in range(2):
                            u_full[(r0+ii)*4 + (c0+jj)] += a_c[bi] * u_inner[ii*2+jj]
            
            # v: for each B block (2x2), multiply by b_c[block_idx] * v_inner
            # B is 4x4 (first 4 cols), V index = j*4+k
            for bi in range(4):
                if b_c[bi] != 0:
                    r0, c0 = b_block_pos[bi]
                    for ii in range(2):
                        for jj in range(2):
                            v_full[(r0+ii)*4 + (c0+jj)] += b_c[bi] * v_inner[ii*2+jj]
            
            # w: for each C block, sum contributions from c_block_coeffs
            # w_inner is 4-dim for a 2x2 C block
            for ci in range(4):
                coeff = c_block_coeffs[(ci//2, ci%2)][t]
                if coeff != 0:
                    r0, c0 = c_block_pos[ci]
                    for ii in range(2):
                        for jj in range(2):
                            # W index: k*n+i where k=col, i=row
                            # C block ci is at rows r0..r0+1, cols c0..c0+1
                            # W[k*n+i] = W[(c0+jj)*4 + (r0+ii)]
                            w_full[(c0+jj)*4 + (r0+ii)] += coeff * w_inner[ii*2+jj]
            
            # Add to decomposition
            for i in range(16):
                U[i][col] = u_full[i]
            for j in range(16):
                V[j][col] = v_full[j]
            for k in range(16):
                W[k][col] = w_full[k]
            col += 1
    
    # col should be 49
    # Part 2: 4x4 * 4x2 schoolbook (rank 32)
    # B columns 4,5 (indices 4,5 in the 4x6 matrix)
    # A[i,j] * B[j,k] for k=4,5
    # V index for B[j,k] = j*6+k
    # W index for C[i,k] = k*4+i
    
    for i in range(4):
        for j in range(4):
            for k in [4, 5]:
                U[i*4+j][col] = 1
                V[j*6+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    # col should be 81
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

