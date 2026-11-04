"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_strassen_2x2_decomposition():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2 = (c+d)e
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1
    
    # M3 = a(f-h)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # M4 = d(g-f)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    
    # M5 = (a+b)h
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _get_strassen_4x4_decomposition():
    """
    Returns U, V, W for 4x4x4 Strassen decomposition.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U_2, V_2, W_2 = _get_strassen_2x2_decomposition()
    
    # Block indices for 4x4 matrix
    # 0: 0:2, 0:2
    # 1: 0:2, 2:4
    # 2: 2:4, 0:2
    # 3: 2:4, 2:4
    
    # Strassen steps for 4x4
    # Each step defines:
    # A_blocks: list of (block_idx, sign)
    # B_blocks: list of (block_idx, sign)
    # C_blocks: list of (block_idx, sign)
    
    steps = [
        # M1
        { 'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)] },
        # M2
        { 'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)] },
        # M3
        { 'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)] },
        # M4
        { 'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)] },
        # M5
        { 'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)] },
        # M6
        { 'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)] },
        # M7
        { 'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)] },
    ]
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for step in steps:
        for inner_col in range(7):
            # Get inner U, V, W values
            u_inner = U_2[:, inner_col]
            v_inner = V_2[:, inner_col]
            w_inner = W_2[:, inner_col]
            
            # Accumulate U
            for block_idx, sign in step['A']:
                # Map block_idx and inner_idx to 4x4 index
                # block_idx 0: rows 0-1, cols 0-1
                # block_idx 1: rows 0-1, cols 2-3
                # block_idx 2: rows 2-3, cols 0-1
                # block_idx 3: rows 2-3, cols 2-3
                
                # Inner index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                
                # We need to add u_inner * sign to the appropriate positions in U
                # u_inner is length 4, corresponding to inner block elements
                # We need to map inner block elements to global 4x4 elements
                
                # This mapping is complex. Let's do it explicitly.
                # For a given block_idx and inner_idx, we find the global row and col.
                
                # Helper to map (block_idx, inner_idx) to (row, col)
                def map_idx(b_idx, i_idx):
                    # b_idx: 0,1,2,3
                    # i_idx: 0,1,2,3
                    # b_idx 0: r 0-1, c 0-1
                    # b_idx 1: r 0-1, c 2-3
                    # b_idx 2: r 2-3, c 0-1
                    # b_idx 3: r 2-3, c 2-3
                    
                    # i_idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    
                    r_base = 0 if b_idx < 2 else 2
                    c_base = 0 if b_idx % 2 == 0 else 2
                    
                    r_off = 0 if i_idx < 2 else 1
                    c_off = 0 if i_idx % 2 == 0 else 1
                    
                    return r_base + r_off, c_base + c_off

                for i_idx in range(4):
                    val = u_inner[i_idx]
                    if val != 0:
                        r, c = map_idx(block_idx, i_idx)
                        U[r*4+c, col] += val * sign
                        
            # Accumulate V
            for block_idx, sign in step['B']:
                for i_idx in range(4):
                    val = v_inner[i_idx]
                    if val != 0:
                        r, c = map_idx(block_idx, i_idx)
                        V[r*4+c, col] += val * sign
                        
            # Accumulate W
            for block_idx, sign in step['C']:
                for i_idx in range(4):
                    val = w_inner[i_idx]
                    if val != 0:
                        r, c = map_idx(block_idx, i_idx)
                        W[r*4+c, col] += val * sign
                        
            col += 1
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    """
    rank = U.shape[1]
    
    # Construct the full tensor T from U, V, W
    # T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
    # Note: Indices in problem statement:
    # U: (n*m, rank) -> U[i*m + j, c]
    # V: (m*p, rank) -> V[j*p + k, c]
    # W: (p*n, rank) -> W[k*n + i, c]
    
    # We need to check if this equals the standard multiplication tensor.
    # Standard tensor: T_std[i, j, k] = 1 if j is the "middle" index?
    # No, the tensor for matrix multiplication C = A B is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices i, j, k.
    # T[i, j, k] = 1 if we are computing C[i, k] using A[i, j] and B[j, k].
    # So T[i, j, k] is 1 for all i, j, k?
    # No, the decomposition is:
    # C[i, k] = sum_c (sum_j U[i*m+j, c] A[i, j]) * (sum_j V[j*p+k, c] B[j, k])?
    # No, the factorization is:
    # C[i, k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i, j]) * (sum_j V[j*p+k, c] B[j, k])?
    # This is not quite right.
    
    # The standard tensor decomposition is:
    # T = sum_c u_c \otimes v_c \otimes w_c
    # where T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
    # And the matrix multiplication tensor is T[i, j, k] = 1 for all i, j, k?
    # No.
    # The multiplication tensor M_{n,m,p} is defined such that:
    # M_{n,m,p}(A, B) = A B
    # In tensor form, the coefficient of A[i, j] B[j, k] in C[i, k] is 1.
    # So the tensor T has T[i, j, k] = 1 for all i, j, k?
    # Yes, because C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor is all ones.
    
    # Wait, is that right?
    # If T[i, j, k] = 1, then sum_c U[i, c] V[j, c] W[k, c] should be 1 for all i, j, k.
    # Let's check the schoolbook decomposition.
    # U[i*m+j, c] = 1 if c corresponds to (i, j, k).
    # V[j*p+k, c] = 1 if c corresponds to (i, j, k).
    # W[k*n+i, c] = 1 if c corresponds to (i, j, k).
    # Then for a fixed (i, j, k), only one term c is non-zero.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # Product is 1.
    # So sum_c ... = 1.
    # So yes, the target tensor is all ones.
    
    # So we need to check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
    
    # We can do this efficiently using numpy.
    # Reshape U, V, W to 3D?
    # U is (n*m, rank). Reshape to (n, m, rank).
    # V is (m*p, rank). Reshape to (m, p, rank).
    # W is (p*n, rank). Reshape to (p, n, rank).
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # We want T[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
    # Note: W index is k*n+i. So W_3d[k, i, c].
    
    # This is a tensor contraction.
    # T[i, j, k] = einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    # Check if T is all ones.
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try Strassen + Schoolbook (Rank 53)
    try:
        U_s, V_s, W_s = _get_strassen_4x4_decomposition()
        
        # Schoolbook for 4x1x4
        # A (4x4) * B_col (4x1) -> C_col (4x1)
        # Rank 4
        U_sb = np.zeros((16, 4))
        V_sb = np.zeros((20, 4)) # m*p = 4*5 = 20
        W_sb = np.zeros((20, 4)) # p*n = 5*4 = 20
        
        for j in range(4):
            U_sb[j, j] = 1 # A[i, j] -> U[i*4+j, c]. Here i is 0..3.
            # Wait, U_sb should be (16, 4).
            # For term j, we use A[:, j].
            # So U_sb[i*4+j, j] = 1 for all i?
            # No, U_sb[i*4+j, c] should be 1 if c corresponds to j.
            # And we sum over i? No, U is a factor of A.
            # C[i, 4] = sum_j A[i, j] B[j, 4].
            # Term j: A[i, j] * B[j, 4] -> C[i, 4].
            # U[i*4+j, j] = 1.
            # V[j*5+4, j] = 1.
            # W[4*4+i, j] = 1.
            
            for i in range(4):
                U_sb[i*4+j, j] = 1
            V_sb[j*5+4, j] = 1
            for i in range(4):
                W_sb[4*4+i, j] = 1
        
        # Combine
        U = np.hstack([U_s, U_sb])
        V = np.hstack([V_s, V_sb])
        W = np.hstack([W_s, W_sb])
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        else:
            print("Strassen+Schoolbook verification failed.")
    except Exception as e:
        print(f"Error in Strassen+Schoolbook: {e}")
    
    # Fallback to Schoolbook (Rank 80)
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
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
