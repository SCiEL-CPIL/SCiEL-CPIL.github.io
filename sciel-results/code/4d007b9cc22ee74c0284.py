import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Verify by computing actual multiplication for random matrices
    np.random.seed(12345)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U_arr[:, c]
            vc = V_arr[:, c]
            wc = W_arr[:, c]
            A_c = uc.reshape(n, m)
            B_c = vc.reshape(m, p)
            A_scaled = A * A_c
            B_scaled = B * B_c
            C_computed += wc.reshape(p, n).T * (A_scaled @ B_scaled)
        
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_strassen_444():
    """Build the Strassen decomposition for 4x4x4 multiplication.
    Returns U, V, W matrices of shapes (16, 49), (16, 49), (16, 49).
    """
    # We will use the Strassen algorithm on 2x2 blocks.
    # There are 7 block multiplications.
    # Each block multiplication is a 2x2x2 multiplication with rank 7.
    # Total rank = 7 * 7 = 49.
    
    # The 7 Strassen intermediate matrices M1..M7 are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Each Mi is a 2x2 matrix product.
    # We need the rank-1 decomposition of 2x2x2.
    # The standard Strassen decomposition for 2x2x2 has rank 7.
    # Let's define the 7 rank-1 tensors for 2x2x2.
    # U_2x2, V_2x2, W_2x2 of shapes (4, 7), (4, 7), (4, 7).
    
    # Strassen 2x2x2 tensors:
    # c1 = (a11+a22)(b11+b22) -> C11 += c1, C22 += c1
    # c2 = (a21+a22)b11 -> C21 += c2
    # c3 = a11(b12-b22) -> C12 += c3
    # c4 = a22(b21-b11) -> C11 += c4
    # c5 = (a11+a12)b22 -> C12 += c5, C22 -= c5
    # c6 = (a21-a11)(b11+b12) -> C22 += c6
    # c7 = (a12-a22)(b21+b22) -> C11 += c7
    
    # We can define the U, V, W factors for each of the 7 terms.
    # For term c1:
    # U_c1 = [1, 0, 0, 1] (a11, a12, a21, a22)
    # V_c1 = [1, 1, 0, 0] (b11, b12, b21, b22)
    # W_c1 = [1, 0, 0, 1] (c11, c12, c21, c22)
    
    # Let's list the 7 terms for 2x2x2:
    # Term 1: U=[1,0,0,1], V=[1,1,0,0], W=[1,0,0,1]
    # Term 2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    # Term 3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    # Term 4: U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,0,0]
    # Term 5: U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,-1]
    # Term 6: U=[0,0,1,-1], V=[1,1,0,0], W=[0,0,0,1]
    # Term 7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
    
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ], dtype=float)
    
    V_2x2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, -1, -1, 1, 0, 1]
    ], dtype=float)
    
    W_2x2 = np.array([
        [1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, -1, 1, 0]
    ], dtype=float)
    
    # Now we combine these for the 4x4x4 case.
    # The 4x4 matrices are split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # The indices for A are 0..15.
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Similarly for B and C.
    
    # We will create U, V, W of shape (16, 49).
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # The 7 Strassen terms for the block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each block-level term s (0..6), we have 7 sub-terms t (0..6).
    # The total index is s*7 + t.
    
    # Define the block-level coefficients for A, B, C.
    # For M1: A blocks: A11 (1), A22 (1). B blocks: B11 (1), B22 (1).
    # For M2: A blocks: A21 (1), A22 (1). B blocks: B11 (1).
    # For M3: A blocks: A11 (1). B blocks: B12 (1), B22 (-1).
    # For M4: A blocks: A22 (1). B blocks: B21 (1), B11 (-1).
    # For M5: A blocks: A11 (1), A12 (1). B blocks: B22 (1).
    # For M6: A blocks: A21 (1), A11 (-1). B blocks: B11 (1), B12 (1).
    # For M7: A blocks: A12 (1), A22 (-1). B blocks: B21 (1), B22 (1).
    
    # We need to map the 2x2 block indices to the 4x4 indices.
    # A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
    
    block_indices = {
        0: [0, 1, 4, 5],   # 11
        1: [2, 3, 6, 7],   # 12
        2: [8, 9, 12, 13], # 21
        3: [10, 11, 14, 15] # 22
    }
    
    # For each block-level term s, we determine which A and B blocks are involved and with what coefficient.
    # Then we combine with the 2x2x2 decomposition.
    
    for s in range(7):
        # Determine A and B block coefficients
        A_blocks = {}
        B_blocks = {}
        
        if s == 0: # M1
            A_blocks = {0: 1, 3: 1}
            B_blocks = {0: 1, 3: 1}
        elif s == 1: # M2
            A_blocks = {2: 1, 3: 1}
            B_blocks = {0: 1}
        elif s == 2: # M3
            A_blocks = {0: 1}
            B_blocks = {1: 1, 3: -1}
        elif s == 3: # M4
            A_blocks = {3: 1}
            B_blocks = {2: 1, 0: -1}
        elif s == 4: # M5
            A_blocks = {0: 1, 1: 1}
            B_blocks = {3: 1}
        elif s == 5: # M6
            A_blocks = {2: 1, 0: -1}
            B_blocks = {0: 1, 1: 1}
        elif s == 6: # M7
            A_blocks = {1: 1, 3: -1}
            B_blocks = {2: 1, 3: 1}
            
        for t in range(7):
            col = s * 7 + t
            
            # Fill U
            for block_id, coeff in A_blocks.items():
                indices = block_indices[block_id]
                # The 2x2x2 U factor for sub-term t is U_2x2[:, t]
                # We need to scale it by coeff and place it in the correct rows
                for k, idx in enumerate(indices):
                    U[idx, col] += coeff * U_2x2[k, t]
            
            # Fill V
            for block_id, coeff in B_blocks.items():
                indices = block_indices[block_id]
                for k, idx in enumerate(indices):
                    V[idx, col] += coeff * V_2x2[k, t]
            
            # Fill W
            # The C blocks are the same as A and B blocks (since it's square)
            # But we need to know which C blocks are involved for each M_s.
            # M1 -> C11, C22
            # M2 -> C21
            # M3 -> C12
            # M4 -> C11
            # M5 -> C12, C22
            # M6 -> C22
            # M7 -> C11
            
            C_blocks = {}
            if s == 0: # M1
                C_blocks = {0: 1, 3: 1}
            elif s == 1: # M2
                C_blocks = {2: 1}
            elif s == 2: # M3
                C_blocks = {1: 1}
            elif s == 3: # M4
                C_blocks = {0: 1}
            elif s == 4: # M5
                C_blocks = {1: 1, 3: -1}
            elif s == 5: # M6
                C_blocks = {3: 1}
            elif s == 6: # M7
                C_blocks = {0: 1}
                
            for block_id, coeff in C_blocks.items():
                indices = block_indices[block_id]
                for k, idx in enumerate(indices):
                    W[idx, col] += coeff * W_2x2[k, t]
                    
    return U, V, W

def _build_schoolbook_443():
    """Build the schoolbook decomposition for 4x4x3 multiplication.
    Returns U, V, W matrices of shapes (16, 48), (12, 48), (12, 48).
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    
    # 1. Strassen for 4x4x4 (columns 0-3)
    U_str, V_str, W_str = _build_strassen_444()
    
    # 2. Schoolbook for 4x4x3 (columns 4-6)
    U_sch, V_sch, W_sch = _build_schoolbook_443()
    
    # Combine them
    # U: (16, 49) and (16, 48) -> (16, 97)
    U = np.hstack([U_str, U_sch])
    
    # V: (16, 49) and (12, 48)
    # V_str is for 4x4 input, so shape (16, 49)
    # V_sch is for 4x3 input, so shape (12, 48)
    # We need to pad V_sch to shape (16, 48) by adding zeros for the rows corresponding to columns 4-6 of B?
    # Wait, V is of shape (m*p, rank). Here m=4, p=7. So V is (28, rank).
    # V_str is for the first 4 columns of B. So it occupies rows 0-15.
    # V_sch is for the last 3 columns of B. So it occupies rows 16-27.
    
    V = np.zeros((28, 97))
    V[0:16, 0:49] = V_str
    V[16:28, 49:97] = V_sch
    
    # W: (16, 49) and (12, 48)
    # W_str is for the first 4 columns of C. So it occupies rows 0-15.
    # W_sch is for the last 3 columns of C. So it occupies rows 16-27.
    # W is of shape (p*n, rank) = (28, rank).
    
    W = np.zeros((28, 97))
    W[0:16, 0:49] = W_str
    W[16:28, 49:97] = W_sch
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if not _verify_decomposition(U_list, V_list, W_list, n, m, p):
        # Fallback to schoolbook if verification fails
        U_fb = np.zeros((16, 112))
        V_fb = np.zeros((28, 112))
        W_fb = np.zeros((28, 112))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*7+k, col] = 1
                    W_fb[k*4+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
