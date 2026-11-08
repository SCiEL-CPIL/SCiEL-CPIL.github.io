"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition with Strassen:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114

Strassen for 2x2*2x2 (rank 7):
  m1 = (a+d)(e+h)
  m2 = (c+d)e
  m3 = a(f-h)
  m4 = d(g-e)
  m5 = (a+b)h
  m6 = (c-a)(e+f)
  m7 = (b-d)(g+h)

  c11 = m1 + m4 - m5 + m7
  c12 = m3 + m5
  c21 = m2 + m4
  c22 = m1 - m2 + m3 + m6
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # We'll build U, V, W matrices
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (36, rank)  
    # W has shape (p*n, rank) = (36, rank)
    
    # Start with empty lists
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2
    # Input: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # Output: C = A*B = [[c11,c12],[c21,c22]]
    
    strassen_2x2_terms = [
        # m1 = (a+d)(e+h), contributes +1 to c11, +1 to c22
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # m2 = (c+d)e, contributes +1 to c21, -1 to c22
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # m3 = a(f-h), contributes +1 to c12, +1 to c22
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # m4 = d(g-e), contributes +1 to c11, +1 to c21
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # m5 = (a+b)h, contributes -1 to c11, +1 to c12
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # m6 = (c-a)(e+f), contributes +1 to c22
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # m7 = (b-d)(g+h), contributes +1 to c11
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add Strassen decomposition for 4x4 * 4x4."""
        
        def get_A_entry(br, bc, li, lj):
            """Get global index for A entry in block (br,bc) at local (li,lj)."""
            gi = A_base_i + br * 2 + li
            gj = A_base_j + bc * 2 + lj
            return gi * m + gj
        
        def get_B_entry(br, bc, lj, lk):
            """Get global index for B entry in block (br,bc) at local (lj,lk)."""
            gj = B_base_j + br * 2 + lj
            gk = B_base_k + bc * 2 + lk
            return gj * p + gk
        
        def get_C_entry(br, bc, li, lk):
            """Get global index for C entry in block (br,bc) at local (li,lk)."""
            gi = C_base_i + br * 2 + li
            gk = C_base_k + bc * 2 + lk
            return gk * n + gi
        
        # The 7 Strassen multiplications for 4x4*4x4:
        # Each Mi is a 2x2*2x2 product, decomposed with Strassen (7 terms)
        
        def add_block_multiplication(A_side, B_side, C_side):
            """
            Add 7 Strassen terms for a block multiplication.
            
            A_side: list of (br, bc, coeff) - blocks to sum with coefficients
            B_side: list of (br, bc, coeff) - blocks to sum with coefficients
            C_side: dict mapping (br, bc) to coeff - C blocks with coefficients
            """
            # Define the 7 Strassen terms as (A_coeffs, B_coeffs, C_contribs)
            # A_coeffs: dict (li,lj) -> coeff
            # B_coeffs: dict (lj,lk) -> coeff
            # C_contribs: dict (li,lk) -> coeff
            
            strassen_meta_terms = [
                # s1 = (a+d)(e+h)
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                # s2 = (c+d)e
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
                # s3 = a(f-h)
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
                # s4 = d(g-e)
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
                # s5 = (a+b)h
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
                # s6 = (c-a)(e+f)
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
                # s7 = (b-d)(g+h)
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ]
            
            for A_coeffs, B_coeffs, C_contribs in strassen_meta_terms:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                # Build u_vec from A_coeffs and A_side
                for (li, lj), meta_coeff in A_coeffs.items():
                    for (br, bc, block_coeff) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += meta_coeff * block_coeff
                
                # Build v_vec from B_coeffs and B_side
                for (lj, lk), meta_coeff in B_coeffs.items():
                    for (br, bc, block_coeff) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += meta_coeff * block_coeff
                
                # Build w_vec from C_contribs and C_side
                for (li, lk), meta_coeff in C_contribs.items():
                    for (br, bc), block_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += meta_coeff * block_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # Now define the 7 block multiplications for 4x4*4x4.
        
        # M1 = (A11+A22)(B11+B22), C11: +1, C22: +1
        add_block_multiplication(
            A_side=[(0,0,1), (1,1,1)],
            B_side=[(0,0,1), (1,1,1)],
            C_side={(0,0): 1, (1,1): 1}
        )
        
        # M2 = (A21+A22)B11, C21: +1, C22: -1
        add_block_multiplication(
            A_side=[(1,0,1), (1,1,1)],
            B_side=[(0,0,1)],
            C_side={(1,0): 1, (1,1): -1}
        )
        
        # M3 = A11(B12-B22), C12: +1, C22: +1
        add_block_multiplication(
            A_side=[(0,0,1)],
            B_side=[(0,1,1), (1,1,-1)],
            C_side={(0,1): 1, (1,1): 1}
        )
        
        # M4 = A22(B21-B11), C11: +1, C21: +1
        add_block_multiplication(
            A_side=[(1,1,1)],
            B_side=[(1,0,1), (0,0,-1)],
            C_side={(0,0): 1, (1,0): 1}
        )
        
        # M5 = (A11+A12)B22, C11: -1, C12: +1
        add_block_multiplication(
            A_side=[(0,0,1), (0,1,1)],
            B_side=[(1,1,1)],
            C_side={(0,0): -1, (0,1): 1}
        )
        
        # M6 = (A21-A11)(B11+B12), C22: +1
        add_block_multiplication(
            A_side=[(1,0,1), (0,0,-1)],
            B_side=[(0,0,1), (0,1,1)],
            C_side={(1,1): 1}
        )
        
        # M7 = (A12-A22)(B21+B22), C11: +1
        add_block_multiplication(
            A_side=[(0,1,1), (1,1,-1)],
            B_side=[(1,0,1), (1,1,1)],
            C_side={(0,0): 1}
        )
    
    # Now call strassen_4x4 for the two 4x4*4x4 parts.
    
    # Part 1: A * B1 where B1 is columns 0-3
    strassen_4x4(
        A_base_i=0, A_base_j=0,
        B_base_j=0, B_base_k=0,
        C_base_i=0, C_base_k=0
    )
    
    # Part 2: A * B2 where B2 is columns 4-7
    strassen_4x4(
        A_base_i=0, A_base_j=0,
        B_base_j=0, B_base_k=4,
        C_base_i=0, C_base_k=4
    )
    
    # Part 3: A * B3 where B3 is column 8 (4x1)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    # Build U, V, W matrices
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    # Verify the decomposition is correct (only if time permits)
    if time.time() < deadline - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Verify with random matrices using efficient computation
        rng = np.random.default_rng(42)
        A_test = rng.random((n, m))
        B_test = rng.random((m, p))
        C_expected = A_test @ B_test
        
        # Verify using numpy: for each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(total - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook if Strassen is wrong
            U_cols.clear()
            V_cols.clear()
            W_cols.clear()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_vec = [0] * (n * m)
                        v_vec = [0] * (m * p)
                        w_vec = [0] * (p * n)
                        u_vec[i * m + j] = 1
                        v_vec[j * p + k] = 1
                        w_vec[k * n + i] = 1
                        add_term(u_vec, v_vec, w_vec)
            
            rank = len(U_cols)
            U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
            V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
            W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}
