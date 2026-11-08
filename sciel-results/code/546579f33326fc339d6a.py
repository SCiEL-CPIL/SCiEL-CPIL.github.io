"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if isinstance(U[0], list) else U.shape[1]
    # Check shapes
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), the tensor entry T[i*m+j, j*p+k, k*n+i] should be 1
    # and all other entries should be 0.
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # Build the full tensor and check
    # T has shape (n*m, m*p, p*n)
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    # Check all other entries are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # This is the "diagonal" - already checked
    # More efficient: check that T is zero except at the diagonal positions
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0
    if np.max(np.abs(T_check)) > 1e-9:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find better decompositions
    # Strategy: Use block decomposition with Strassen-like ideas
    
    # Approach: Try various ranks from low to high
    # Known: R(2,2,2) = 7 (Strassen)
    # For (2,3,7), we can try to embed Strassen in blocks
    
    # Let's try a systematic approach:
    # Split the middle dimension 3 into 2+1
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:3] @ B[2:3,:]
    
    # For the (2,2,7) part: multiply 2x2 by 2x7
    # This is equivalent to 7 independent (2,2,1) multiplications
    # R(2,2,1) = 4 (schoolbook, since p=1, it's just 2*2*1=4)
    # Actually for (2,2,1): C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # This is a rank-4 tensor (schoolbook)
    # Can we do better? R(2,2,1) = 4 is optimal since it's just dot products.
    
    # So (2,2,7) has rank 7*4 = 28 by this decomposition
    # And (2,1,7) has rank 7*2 = 14
    # Total: 42, same as schoolbook.
    
    # Let me try a different approach: use the full Strassen structure
    # For (2,2,2), Strassen uses rank 7.
    # Can we use this for part of our computation?
    
    # Idea: Split 7 into 2+2+2+1
    # C = A @ B[0:2] + A @ B[2:4] + A @ B[4:6] + A @ B[6:7]
    # Each A @ B[j:j+2] is (2,3,2)
    # R(2,3,2) - what's this?
    # (2,3,2): C[i,k] = sum_j A[i,j] B[j,k] for i in 0,1; j in 0,1,2; k in 0,1
    # Schoolbook: 2*3*2 = 12
    # Can we do better?
    
    # Let's just try to find any rank below 42 by direct construction
    # Try rank 35, 36, ..., 41
    
    for target_rank in range(35, 42):
        if time.time() > deadline - 10:
            break
        
        # Try to construct a decomposition of this rank
        # Use a heuristic: start from schoolbook and try to merge terms
        # Or use a randomized approach
        
        # For now, let's try a specific construction for rank 39
        if target_rank == 39:
            # Try: use Strassen for the 2x2x2 blocks
            # Split 3 = 2 + 1, 7 = 2 + 2 + 2 + 1
            # This gets complex. Let's just return schoolbook for now.
            pass
    
    # If we haven't found better, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
