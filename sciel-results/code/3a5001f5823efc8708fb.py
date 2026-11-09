"""Matrix multiplication tensor decomposition for 4x4x5."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0
    
    # We'll build the decomposition column by column.
    # Each column r has:
    #   U[i*m + j, r] = coefficient of A[i,j]
    #   V[j*p + k, r] = coefficient of B[j,k]  
    #   W[k*n + i, r] = coefficient for C[i,k]
    
    # Strategy: 
    # 1. Decompose the 4x4x4 part using recursive Strassen (rank 49)
    # 2. Add rank 16 for the 4x4x1 part (the 5th column of B)
    # Total rank = 49 + 16 = 65
    
    # Build Strassen decomposition for 4x4x4
    # We represent each "multiplication" as a tuple (U_coeffs, V_coeffs, W_coeffs)
    # where U_coeffs is a dict mapping (i,j) -> coeff, etc.
    
    def strassen_2x2_2x2():
        """Return list of 7 terms, each (A_coeffs, B_coeffs, C_coeffs)"""
        # A indices: (0,0)->a0, (0,1)->a1, (1,0)->a2, (1,1)->a3
        # B indices: (0,0)->b0, (0,1)->b1, (1,0)->b2, (1,1)->b3
        # C indices: (0,0)->c0, (0,1)->c1, (1,0)->c2, (1,1)->c3
        
        terms = []
        # M1 = (a0+a3)(b0+b3)
        terms.append((
            {(0,0): 1, (1,1): 1},  # A coeffs
            {(0,0): 1, (1,1): 1},  # B coeffs
            {(0,0): 1, (1,1): 1}   # C coeffs: c00 += M1, c11 += M1
        ))
        # M2 = (a2+a3)(b0)
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}  # c10 += M2, c11 -= M2
        ))
        # M3 = a0(b1-b3)
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}   # c01 += M3, c11 += M3
        ))
        # M4 = a3(b2-b0)
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}   # c00 += M4, c10 += M4
        ))
        # M5 = (a0+a1)b3
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}  # c00 -= M5, c01 += M5
        ))
        # M6 = (a2+a0)(b1+b3)
        terms.append((
            {(1,0): 1, (0,0): 1},
            {(0,1): 1, (1,1): 1},
            {(1,1): 1}             # c11 += M6
        ))
        # M7 = (a1-a3)(b2+b0)
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (0,0): 1},
            {(0,0): 1}             # c00 += M7
        ))
        return terms
    
    def strassen_4x4_4x4():
        """Recursively apply Strassen to get 49 terms for 4x4x4."""
        # Split 4x4 into four 2x2 blocks
        # A = [[A00, A01], [A10, A11]] where each is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        base_terms = strassen_2x2_2x2()
        
        # For each of the 7 Strassen multiplications, 
        # the "A factor" is a 2x2 matrix (linear combo of A blocks)
        # and the "B factor" is a 2x2 matrix (linear combo of B blocks)
        # Their product is a 2x2 matrix which contributes to C blocks.
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10+A00)(B01+B11)
        # M7 = (A01-A11)(B10+B00)
        
        # For each Mi, the A-factor is a 2x2 matrix built from A blocks,
        # the B-factor is a 2x2 matrix built from B blocks.
        # The product Pi = Ai * Bi is a 2x2 matrix.
        # Pi contributes to C blocks with specific coefficients.
        
        # Let's define the block structure:
        # A block (bi, bj) means A[2*bi:2*bi+2, 2*bj:2*bj+2]
        # Similarly for B and C.
        
        # For each Strassen multiplication, we need to express:
        # Ai = sum over A-blocks with coefficients
        # Bi = sum over B-blocks with coefficients
        # Then Pi = Ai * Bi contributes to C-blocks.
        
        # Let me define the 7 Strassen multiplications at the block level:
        # Each has: A_block_coeffs: dict from block_index -> coeff
        #            B_block_coeffs: dict from block_index -> coeff
        #            C_block_coeffs: dict from (row_block, col_block) -> coeff
        
        block_A_coeffs = [
            {(0,0): 1, (1,1): 1},   # M1: A00 + A11
            {(1,0): 1, (1,1): 1},   # M2: A10 + A11
            {(0,0): 1},              # M3: A00
            {(1,1): 1},              # M4: A11
            {(0,0): 1, (0,1): 1},   # M5: A00 + A01
            {(1,0): 1, (0,0): 1},   # M6: A10 + A00
            {(0,1): 1, (1,1): -1},  # M7: A01 - A11
        ]
        block_B_coeffs = [
            {(0,0): 1, (1,1): 1},   # M1: B00 + B11
            {(0,0): 1},              # M2: B00
            {(0,1): 1, (1,1): -1},  # M3: B01 - B11
            {(1,0): 1, (0,0): -1},  # M4: B10 - B00
            {(1,1): 1},              # M5: B11
            {(0,1): 1, (1,1): 1},   # M6: B01 + B11
            {(1,0): 1, (0,0): 1},   # M7: B10 + B00
        ]
        block_C_coeffs = [
            {(0,0): 1, (1,1): 1},   # M1 -> C00, C11
            {(1,0): 1, (1,1): -1},  # M2 -> C10, C11
            {(0,1): 1, (1,1): 1},   # M3 -> C01, C11
            {(0,0): 1, (1,0): 1},   # M4 -> C00, C10
            {(0,0): -1, (0,1): 1},  # M5 -> C00, C01
            {(1,1): 1},             # M6 -> C11
            {(0,0): 1},             # M7 -> C00
        ]
        
        # Now for each of the 7 block-level multiplications,
        # we need to expand into 4x4 = 16 scalar-level terms.
        # Actually, each block multiplication is a 2x2 * 2x2 = 2x2 multiplication,
        # which itself can be done with 7 scalar multiplications (Strassen again).
        
        # But wait - at this level, the "A factor" is a 2x2 matrix where each entry
        # is a linear combination of the 2x2 A blocks. Similarly for B.
        # The product is a 2x2 matrix, each entry being a bilinear form.
        
        # Let me think differently. For each of the 7 Strassen multiplications at the
        # 4x4 level, the A-factor is a 2x2 matrix whose entries are 2x2 matrices (block sums).
        # Actually no - the A-factor in M1 = (A00+A11)(B00+B11) is the 2x2 matrix A00+A11,
        # which is itself a 2x2 matrix. So we're computing a 2x2 matrix times a 2x2 matrix.
        
        # To decompose this into scalar multiplications, we apply Strassen to each 2x2*2x2.
        # So for each of the 7 block multiplications, we get 7 scalar multiplications.
        # Total: 7 * 7 = 49.
        
        # For each block multiplication m (0..6):
        #   A_factor = sum of A-blocks with coefficients from block_A_coeffs[m]
        #   B_factor = sum of B-blocks with coefficients from block_B_coeffs[m]
        #   Then A_factor * B_factor contributes to C-blocks with coefficients from block_C_coeffs[m]
        
        # A_factor is a 2x2 matrix. Its (r,c) entry is:
        #   sum over (bi,bj) in block_A_coeffs[m] of coeff * A_block(bi,bj)[r,c]
        # where A_block(bi,bj)[r,c] = A[2*bi+r, 2*bj+c]
        
        # So the (r,c) entry of A_factor = sum over (bi,bj) of coeff * A[2*bi+r, 2*bj+c]
        
        # Similarly for B_factor: (r,c) entry = sum over (bi,bj) of coeff * B[2*bi+r, 2*bj+c]
        
        # Now, A_factor * B_factor is a 2x2 matrix P where:
        # P[u,v] = sum over t of A_factor[u,t] * B_factor[t,v]
        
        # Each P[u,v] contributes to C-block (u,v) with coefficient from block_C_coeffs[m].
        
        # So for scalar A[i,j] where i=2*bi+r, j=2*bj+c:
        #   It appears in A_factor[u,t] where u=r, t=c with coefficient coeff_{bi,bj}
        #   if (bi,bj) is in block_A_coeffs[m]
        
        # This is getting complex. Let me just build it computationally.
        
        all_terms = []
        
        for m in range(7):
            a_blocks = block_A_coeffs[m]  # dict from (bi,bj) -> coeff
            b_blocks = block_B_coeffs[m]  # dict from (bi,bj) -> coeff
            c_blocks = block_C_coeffs[m]  # dict from (ci,cj) -> coeff
            
            # Build the 2x2 A_factor matrix entries as dicts of (i,j) -> coeff
            # A_factor[u,t] where u,t in {0,1}
            A_factor = {}  # (u,t) -> dict of (i,j) -> coeff
            for u in range(2):
                for t in range(2):
                    coeffs = {}
                    for (bi, bj), coeff in a_blocks.items():
                        i = 2*bi + u
                        j = 2*bj + t
                        coeffs[(i,j)] = coeffs.get((i,j), 0) + coeff
                    A_factor[(u,t)] = coeffs
            
            # Build the 2x2 B_factor matrix entries
            B_factor = {}  # (t,v) -> dict of (j,k) -> coeff
            for t in range(2):
                for v in range(2):
                    coeffs = {}
                    for (bi, bj), coeff in b_blocks.items():
                        j = 2*bi + t
                        k = 2*bj + v
                        coeffs[(j,k)] = coeffs.get((j,k), 0) + coeff
                    B_factor[(t,v)] = coeffs
            
            # Now P[u,v] = sum_t A_factor[u,t] * B_factor[t,v]
            # Each P[u,v] contributes to C-block (u,v)
            
            for u in range(2):
                for v in range(2):
                    # C-block (u,v) gets coefficient c_blocks.get((u,v), 0)
                    cv = c_blocks.get((u,v), 0)
                    if cv == 0:
                        continue
                    
                    # P[u,v] = sum_t A_factor[u,t] * B_factor[t,v]
                    # This is a sum of products of bilinear forms.
                    # We need to decompose this into rank-1 terms.
                    
                    # A_factor[u,t] is a vector in R^{16} (coefficients of A elements)
                    # B_factor[t,v] is a vector in R^{16} (coefficients of B elements)
                    # Their product contributes to C[u*2:u*2+2, v*2:v*2+2] with coefficient cv
                    
                    # But A_factor[u,t] * B_factor[t,v] is already a rank-1 term!
                    # It's a single scalar multiplication: (sum_i A_factor[u,t][i,j] * A[i,j]) * (sum_k B_factor[t,v][j,k] * B[j,k])
                    
                    # Wait, no. A_factor[u,t] is a linear combination of A elements, giving a scalar.
                    # B_factor[t,v] is a linear combination of B elements, giving a scalar.
                    # Their product is one scalar multiplication.
                    # This scalar contributes to C-block (u,v) with coefficient cv.
                    
                    # So for each t, we get one term:
                    for t in range(2):
                        a_vec = A_factor[(u,t)]  # dict (i,j) -> coeff
                        b_vec = B_factor[(t,v)]  # dict (j,k) -> coeff
                        
                        # C contribution: to C-block (u,v), each element C[2*u+ru, 2*v+cv_col] gets coefficient cv
                        c_vec = {}
                        for ru in range(2):
                            for cv_col in range(2):
                                i_c = 2*u + ru
                                k_c = 2*v + cv_col
                                c_vec[(k_c, i_c)] = cv  # W index is (k, i)
                        
                        all_terms.append((a_vec, b_vec, c_vec))
        
        return all_terms
    
    # Get the 49 terms for 4x4x4
    terms_4x4x4 = strassen_4x4_4x4()
    
    # Now add 16 terms for the 4x4x1 part (column 4 of B)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    terms_extra = []
    for i in range(4):
        for j in range(4):
            a_vec = {(i,j): 1}
            b_vec = {(j,4): 1}
            c_vec = {(4, i): 1}  # W[k*n+i] with k=4, i=i
            terms_extra.append((a_vec, b_vec, c_vec))
    
    all_terms = terms_4x4x4 + terms_extra
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (a_vec, b_vec, c_vec) in enumerate(all_terms):
        for (i, j), coeff in a_vec.items():
            if coeff != 0:
                U[i * m + j][r] = coeff
        for (j, k), coeff in b_vec.items():
            if coeff != 0:
                V[j * p + k][r] = coeff
        for (k, i), coeff in c_vec.items():
            if coeff != 0:
                W[k * n + i][r] = coeff
    
    # Verify the decomposition
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # This should hold for the 4x4x4 part (i,j,k with k<4) and the extra part handles k=4
    
    # Actually, the tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    # Let's verify numerically
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
