"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
_START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    
    We split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    We split B (4x6) into 2x2 blocks: B11, B12, B13, B21, B22, B23
    Output C (4x6) has 2x2 blocks: C11, C12, C13, C21, C22, C23
    
    For each output block C[i,j] (i in {1,2}, j in {1,2,3}):
    C[i,j] = A[i,1]*B[1,j] + A[i,2]*B[2,j]
    
    Each term is a 2x2 * 2x2 product. We use Strassen to reduce multiplications.
    """
    n, m, p = 4, 4, 6
    
    # We'll build the decomposition column by column (each column = one scalar multiplication)
    # Using Strassen for 2x2 blocks: 7 mults per 2x2*2x2 instead of 4
    
    # Block indices: A blocks indexed by (row_block, col_block) where row_block in {0,1}, col_block in {0,1}
    # B blocks indexed by (row_block, col_block) where row_block in {0,1}, col_block in {0,1,2}
    # C blocks indexed by (row_block, col_block) where row_block in {0,1}, col_block in {0,1,2}
    
    # For C[rb, cb] = A[rb, 0]*B[0, cb] + A[rb, 1]*B[1, cb]
    # Let X = A[rb, 0], Y = A[rb, 1], U = B[0, cb], V = B[1, cb]
    # C = X*U + Y*V
    # This is a sum of two 2x2 products.
    
    # Actually, let's think of it as: we have a 2x4 matrix [X | Y] times a 4x2 matrix [U; V]
    # = X*U + Y*V
    # Naively: 2*4*2 = 16 mults per output block, 6 blocks = 96 total (schoolbook)
    
    # Using Strassen-like: for 2x2 * 2x2, Strassen uses 7 mults.
    # But here we have X*U + Y*V where X,Y,U,V are all 2x2.
    # We can write this as [X, Y] * [U; V] where [X,Y] is 2x4 and [U;V] is 4x2.
    # The rank of 2x4x2 matrix mult is at most 16, but can we do better?
    
    # Let me just use the straightforward Strassen on each 2x2*2x2 and add them.
    # X*U: 7 mults (Strassen)
    # Y*V: 7 mults (Strassen)
    # Total per output block: 14 mults
    # 6 output blocks: 84 mults total
    
    # This matches the observed improvement!
    
    # Build the decomposition directly.
    # Each "multiplication" in the tensor decomposition corresponds to one column in U, V, W.
    # U has shape (4*4, rank) = (16, rank)
    # V has shape (4*6, rank) = (24, rank)
    # W has shape (6*4, rank) = (24, rank)
    
    # The tensor T[i,j,k] = delta[i,k] (for matrix mult A[i,j]*B[j,k])
    # Wait, the convention is: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if... no. The tensor for matrix mult is T[i,j,k] = delta[i,j+k] or something.
    
    # Let me re-read: "W[k*n+i,r]" - so W is indexed by [k*n + i, r] where k is the column of B / row of C, i is row of A / col of C.
    # U is [i*m + j, r] where i is row of A, j is col of A / row of B
    # V is [j*p + k, r] where j is row of B, k is col of B / col of C
    
    # So the decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 if this is a valid matrix mult term, i.e., always 1 for the standard mult.
    # Wait no, T[i,j,k] = delta for the identity... 
    
    # Actually for matrix multiplication A(4x4) * B(4x6) = C(4x6):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid i,j,k (since every term contributes).
    # No wait, that's not right either. The tensor decomposition represents the bilinear map.
    # T[i,j,k] = 1 means that A[i,j] * B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,6).
    # And the decomposition: sum_r U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r] = T[i,j,k] = 1
    
    # So we need: for each (i,j,k), sum_r U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r] = 1
    
    # Schoolbook: one term per (i,j,k), rank = 4*4*6 = 96.
    
    # For Strassen: we group terms.
    
    # Let me build it block by block.
    # A is 4x4, split into 2x2 blocks: A[rb, cb] for rb in {0,1}, cb in {0,1}
    # B is 4x6, split into 2x2 blocks: B[rb, cb] for rb in {0,1}, cb in {0,1,2}
    # C is 4x6, split into 2x2 blocks: C[rb, cb] for rb in {0,1}, cb in {0,1,2}
    
    # C[rb, cb] = sum_{rb2 in {0,1}} A[rb, rb2] * B[rb2, cb]
    
    # For each (rb, cb), we compute C[rb, cb] = A[rb,0]*B[0,cb] + A[rb,1]*B[1,cb]
    
    # Let me define the Strassen decomposition for 2x2*2x2.
    # For matrices X (2x2) and Y (2x2), XY has rank 7 via Strassen.
    # The 7 multiplications are:
    # m1 = (X11+X22)(Y11+Y22)
    # m2 = (X21+X22)Y11
    # m3 = X11(Y12-Y22)
    # m4 = X22(Y21-Y11)
    # m5 = (X11+X12)Y22
    # m6 = (X21-X11)(Y11+Y12)
    # m7 = (X12-X22)(Y21+Y22)
    # C11 = m1+m4-m5+m7
    # C12 = m3+m5
    # C21 = m2+m4
    # C22 = m1-m2+m3+m6
    
    # For C[rb,cb] = A[rb,0]*B[0,cb] + A[rb,1]*B[1,cb]:
    # Let X = A[rb,0], Y = B[0,cb], X' = A[rb,1], Y' = B[1,cb]
    # C = X*Y + X'*Y'
    # I can use Strassen for X*Y (7 mults) and Strassen for X'*Y' (7 mults), total 14.
    # 6 output blocks * 14 = 84 total.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_term(offset_i, offset_j, offset_k):
        """
        Add Strassen decomposition of X_block * Y_block where:
        X_block is a 2x2 block of A at (offset_i, offset_j) 
        Y_block is a 2x2 block of B at (offset_j, offset_k)
        offset_i: starting row index in A (0 or 2)
        offset_j: starting col index in A / row index in B (0 or 2)
        offset_k: starting col index in B (0, 2, or 4)
        """
        # X_block elements: A[offset_i + di, offset_j + dj] for di,dj in {0,1}
        # Y_block elements: B[offset_j + di, offset_k + dk] for di,dk in {0,1}
        # Output: C[offset_i + di, offset_k + dk]
        
        # Strassen multiplications:
        # Each is of the form: (sum of X elements with coeffs) * (sum of Y elements with coeffs)
        # contributing to specific C elements with coeffs.
        
        # m1 = (X11+X22)(Y11+Y22)
        # X11 = A[offset_i+0, offset_j+0], X22 = A[offset_i+1, offset_j+1]
        # Y11 = B[offset_j+0, offset_k+0], Y22 = B[offset_j+1, offset_k+1]
        
        strassen_mults = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U_coeffs: dict mapping (i,j) to coeff for A[i,j]
            # V_coeffs: dict mapping (j,k) to coeff for B[j,k]
            # W_coeffs: dict mapping (i,k) to coeff for C[i,k]
            
            # m1 = (X11+X22)(Y11+Y22)
            ({(offset_i+0, offset_j+0): 1, (offset_i+1, offset_j+1): 1},
             {(offset_j+0, offset_k+0): 1, (offset_j+1, offset_k+1): 1},
             {(offset_i+0, offset_k+0): 1, (offset_i+1, offset_k+1): 1,
              (offset_i+0, offset_k+1): -1, (offset_i+1, offset_k+0): 1}),
            # Wait, this isn't right. Let me be more careful.
        ]
        
        # Let me redo this properly.
        # For X*Y where X is 2x2, Y is 2x2:
        # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
        # XY = [[x00*y00+x01*y10, x00*y01+x01*y11],
        #       [x10*y00+x11*y10, x10*y01+x11*y11]]
        
        # Strassen:
        # m1 = (x00+x11)(y00+y11)
        # m2 = (x10+x11)y00
        # m3 = x00(y01-y11)
        # m4 = x11(y10-y00)
        # m5 = (x00+x01)y11
        # m6 = (x10-x00)(y00+y01)
        # m7 = (x01-x11)(y10+y11)
        # C00 = m1+m4-m5+m7
        # C01 = m3+m5
        # C10 = m2+m4
        # C11 = m1-m2+m3+m6
        
        # In terms of the full matrices:
        # x00 = A[offset_i, offset_j], x01 = A[offset_i, offset_j+1]
        # x10 = A[offset_i+1, offset_j], x11 = A[offset_i+1, offset_j+1]
        # y00 = B[offset_j, offset_k], y01 = B[offset_j, offset_k+1]
        # y10 = B[offset_j+1, offset_k], y11 = B[offset_j+1, offset_k+1]
        # C00 = C[offset_i, offset_k], C01 = C[offset_i, offset_k+1]
        # C10 = C[offset_i+1, offset_k], C11 = C[offset_i+1, offset_k+1]
        
        # Each Strassen multiplication m_r = (linear combo of X) * (linear combo of Y)
        # contributes to C with specific coefficients.
        
        # m1: U = x00+x11, V = y00+y11
        #   U coeffs: {(offset_i, offset_j): 1, (offset_i+1, offset_j+1): 1}
        #   V coeffs: {(offset_j, offset_k): 1, (offset_j+1, offset_k+1): 1}
        #   W coeffs: {(offset_i, offset_k): 1, (offset_i+1, offset_k+1): 1}
        #   (C00 += m1, C11 += m1)
        
        # m2: U = x10+x11, V = y00
        #   U coeffs: {(offset_i+1, offset_j): 1, (offset_i+1, offset_j+1): 1}
        #   V coeffs: {(offset_j, offset_k): 1}
        #   W coeffs: {(offset_i+1, offset_k): 1, (offset_i+1, offset_k+1): -1}
        #   (C10 += m2, C11 -= m2)
        
        # m3: U = x00, V = y01-y11
        #   U coeffs: {(offset_i, offset_j): 1}
        #   V coeffs: {(offset_j, offset_k+1): 1, (offset_j+1, offset_k+1): -1}
        #   W coeffs: {(offset_i, offset_k+1): 1, (offset_i+1, offset_k+1): 1}
        #   (C01 += m3, C11 += m3)
        
        # m4: U = x11, V = y10-y00
        #   U coeffs: {(offset_i+1, offset_j+1): 1}
        #   V coeffs: {(offset_j+1, offset_k): 1, (offset_j, offset_k): -1}
        #   W coeffs: {(offset_i, offset_k): 1, (offset_i+1, offset_k): 1}
        #   (C00 += m4, C10 += m4)
        
        # m5: U = x00+x01, V = y11
        #   U coeffs: {(offset_i, offset_j): 1, (offset_i, offset_j+1): 1}
        #   V coeffs: {(offset_j+1, offset_k+1): 1}
        #   W coeffs: {(offset_i, offset_k): -1, (offset_i, offset_k+1): 1}
        #   (C00 -= m5, C01 += m5)
        
        # m6: U = x10-x00, V = y00+y01
        #   U coeffs: {(offset_i+1, offset_j): 1, (offset_i, offset_j): -1}
        #   V coeffs: {(offset_j, offset_k): 1, (offset_j, offset_k+1): 1}
        #   W coeffs: {(offset_i+1, offset_k+1): 1}
        #   (C11 += m6)
        
        # m7: U = x01-x11, V = y10+y11
        #   U coeffs: {(offset_i, offset_j+1): 1, (offset_i+1, offset_j+1): -1}
        #   V coeffs: {(offset_j+1, offset_k): 1, (offset_j+1, offset_k+1): 1}
        #   W coeffs: {(offset_i, offset_k): 1}
        #   (C00 += m7)
        
        mults = [
            # m1
            ({(offset_i, offset_j): 1, (offset_i+1, offset_j+1): 1},
             {(offset_j, offset_k): 1, (offset_j+1, offset_k+1): 1},
             {(offset_i, offset_k): 1, (offset_i+1, offset_k+1): 1}),
            # m2
            ({(offset_i+1, offset_j): 1, (offset_i+1, offset_j+1): 1},
             {(offset_j, offset_k): 1},
             {(offset_i+1, offset_k): 1, (offset_i+1, offset_k+1): -1}),
            # m3
            ({(offset_i, offset_j): 1},
             {(offset_j, offset_k+1): 1, (offset_j+1, offset_k+1): -1},
             {(offset_i, offset_k+1): 1, (offset_i+1, offset_k+1): 1}),
            # m4
            ({(offset_i+1, offset_j+1): 1},
             {(offset_j+1, offset_k): 1, (offset_j, offset_k): -1},
             {(offset_i, offset_k): 1, (offset_i+1, offset_k): 1}),
            # m5
            ({(offset_i, offset_j): 1, (offset_i, offset_j+1): 1},
             {(offset_j+1, offset_k+1): 1},
             {(offset_i, offset_k): -1, (offset_i, offset_k+1): 1}),
            # m6
            ({(offset_i+1, offset_j): 1, (offset_i, offset_j): -1},
             {(offset_j, offset_k): 1, (offset_j, offset_k+1): 1},
             {(offset_i+1, offset_k+1): 1}),
            # m7
            ({(offset_i, offset_j+1): 1, (offset_i+1, offset_j+1): -1},
             {(offset_j+1, offset_k): 1, (offset_j+1, offset_k+1): 1},
             {(offset_i, offset_k): 1}),
        ]
        
        for U_coeffs, V_coeffs, W_coeffs in mults:
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            for (i, j), coeff in U_coeffs.items():
                U_col[i * m + j] = coeff
            
            for (j, k), coeff in V_coeffs.items():
                V_col[j * p + k] = coeff
            
            for (i, k), coeff in W_coeffs.items():
                W_col[k * n + i] = coeff
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    # For each output block (rb, cb):
    # C[rb, cb] = A[rb, 0]*B[0, cb] + A[rb, 1]*B[1, cb]
    # Use Strassen for A[rb,0]*B[0,cb] and A[rb,1]*B[1,cb]
    
    for rb in range(2):  # row block of A/C
        for cb in range(3):  # col block of B/C
            if _time_remaining() < 5.0:
                break
            # Term 1: A[rb, 0] * B[0, cb]
            # A[rb, 0] is at offset_i = rb*2, offset_j = 0
            # B[0, cb] is at offset_j = 0, offset_k = cb*2
            add_strassen_term(rb*2, 0, cb*2)
            
            # Term 2: A[rb, 1] * B[1, cb]
            # A[rb, 1] is at offset_i = rb*2, offset_j = 2
            # B[1, cb] is at offset_j = 2, offset_k = cb*2
            add_strassen_term(rb*2, 2, cb*2)
    
    rank = len(U_cols)
    
    # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
