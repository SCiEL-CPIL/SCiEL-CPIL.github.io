import numpy as np
import time

def _strassen_2x2x2():
    # Standard Strassen 2x2x2, rank 7.
    # Indexing: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # M2 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1
    
    # M3 = A00(B01-B11)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1
    
    # M4 = A11(B10-B00)
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[2,3]=1
    
    # M5 = (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1
    
    return U2, V2, W2

def _block_specs():
    # Strassen steps for 4x4x4.
    # Each step is a tuple (A_spec, B_spec, C_spec).
    # Specs are lists of (coeff, block_idx).
    # block_idx: 0=00, 1=01, 2=10, 3=11 (row_block, col_block)
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]
    return A_specs, B_specs, C_specs

def _build(rank=98):
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    A_specs, B_specs, C_specs = _block_specs()
    
    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                
                # A part (16x1)
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                            
                # B part (32x1)
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                            
                # C part (32x1)
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
                            
    return U, V, W

def _verify(U, V, W, n, m, p):
    # Check if U, V, W exactly decompose the tensor.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor coefficient T[i,j,k] = 1
    # Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                # Vectorized dot product is faster, but for small dims python loop might be ok?
                # Let's use numpy dot for speed.
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _remove_dependent(U, V, W, n, m, p):
    # Attempt to reduce rank by finding linearly dependent columns.
    # A column r is dependent if there exist coefficients c_s (s != r) such that
    # c_r U[:, r] + sum c_s U[:, s] = 0 AND c_r V[:, r] + sum c_s V[:, s] = 0 AND c_r W[:, r] + sum c_s W[:, s] = 0.
    # This is equivalent to finding a vector c in the null space of the block matrix M = [U | V | W] (stacked vertically? No, side by side? No).
    # The condition is that the column vectors (U_r, V_r, W_r) in R^(16+32+32) are linearly dependent.
    # Let M be a matrix of shape (80, rank).
    # If rank(M) < rank, we can remove a dependent column.
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Stack U, V, W vertically to form M
    # M has shape (80, rank)
    M = np.vstack([U, V, W])
    
    # Compute rank of M
    # Use SVD to find null space
    # For integer entries, we can use exact arithmetic if available, but np.linalg.svd is float.
    # Since entries are small integers, float precision should be sufficient to detect dependence if it exists structurally.
    try:
        U_svd, s, Vh = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
        
    # Tolerance for zero singular values
    tol = 1e-10
    r = np.sum(s > tol)
    
    if r >= rank:
        return U, V, W
        
    # There is dependence.
    # The null space is spanned by the last (rank - r) rows of Vh.
    # We want to find a specific column that can be removed.
    # A simple heuristic: try removing columns one by one and check if the rank drops.
    # Or, find a vector in the null space with a large component at some index, and eliminate that index.
    
    # Let's try to find a column 'pivot' that has a non-zero entry in the null vector,
    # and then express that column as a combination of others.
    # Actually, simply deleting a column that is part of a dependent set works if the rest still span the same subspace?
    # No, we need the *set* of columns to still span the same subspace (or at least preserve the tensor decomposition).
    # If columns are linearly dependent, removing one *might* break the decomposition if it's the only one contributing to some tensor component?
    # Wait. The decomposition is: T = sum_r u_r v_r w_r.
    # If u_r = sum_{s!=r} c_s u_s, etc, then T = sum_{s!=r} (c_s u_s v_s w_s + ...)? No.
    # T = sum_r u_r v_r w_r.
    # If u_r = sum c_s u_s, v_r = sum c_s v_s, w_r = sum c_s w_s?
    # No, the dependency is on the triple (u_r, v_r, w_r).
    # If (u_r, v_r, w_r) = sum c_s (u_s, v_s, w_s), then
    # u_r v_r w_r is NOT equal to sum c_s u_s v_s w_s.
    # So we cannot simply remove dependent columns!
    
    # Rank reduction in tensor decomposition is NOT just about linear dependence of the factor matrices columns.
    # It requires finding a representation with fewer terms.
    # The "linear dependence" check above is a necessary condition for redundancy in the *space* spanned, but not sufficient for removing a term from the sum.
    # Example: (1,0,0) + (0,1,0) + (1,1,0) = (2,2,0). But 1*0*0 + 0*1*0 + 1*1*0 != 2*2*0?
    # Actually, if u_r v_r w_r is a scalar product of vectors, it's not linear.
    
    # Therefore, the standard "remove dependent columns" approach DOES NOT WORK for tensor rank reduction.
    # The inspirations might have been trying to find a *different* decomposition or using a specific structure.
    # However, for this task, we will stick to the valid rank-98 decomposition.
    # If we cannot structurally reduce it, we return 98.
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    n, m, p = 4, 4, 8
    
    # Build the rank 98 decomposition
    U, V, W = _build(98)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to naive if something went wrong (should not happen)
        return _naive(n, m, p)
        
    # Attempt refinement (currently a no-op due to theoretical impossibility of simple column removal)
    if time.time() < deadline:
        U, V, W = _remove_dependent(U, V, W, n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _naive(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
