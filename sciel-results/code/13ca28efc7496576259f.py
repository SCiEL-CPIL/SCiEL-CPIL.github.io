"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 145.0  # Safety margin for 150s limit

    # 1. Try Rank 35: Strassen for 2x2x2 block + Schoolbook for rest
    # The 2x2x2 block covers i in {0,1}, j in {0,1}, k in {0,1}
    # This corresponds to indices:
    # U indices: i*m + j -> 0, 1, 3, 4 (for i,j in 0,1)
    # V indices: j*p + k -> 0, 1, 4, 5 (for j,k in 0,1) -- wait, p=4, so j=0 -> 0,1; j=1 -> 4,5
    # W indices: k*n + i -> 0, 1, 3, 4 (for k,i in 0,1) -- wait, n=3, so k=0 -> 0,1; k=1 -> 3,4
    
    # Let's define the full U, V, W matrices for rank 35
    rank = 35
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen decomposition for 2x2 matmul
    # We need 7 terms for the sub-tensor T_sub[i,j,k] where i,j,k in {0,1}
    # The standard Strassen formulas for C = A*B:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 + M3 - M2 + M6
    
    # We map these to rank-1 tensors (u, v, w) such that sum u_i_j v_j_k w_i_k = T_i_j_k
    # For the sub-block, we need to represent the identity tensor of size 2x2x2?
    # No, the matmul tensor T[i,j,k] = 1 for all i,j,k is not the identity.
    # The matmul tensor is defined by T[i,j,k] = 1 if we are computing C[i,k] = sum_j A[i,j]B[j,k].
    # In the decomposition context, we want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k in the block?
    # Actually, the tensor T is such that T[i*m+j, j*p+k, k*n+i] = 1.
    # For the sub-block i,j,k in {0,1}, we have 8 entries in the full tensor that are 1.
    # We want to cover these 8 entries with 7 rank-1 terms.
    
    # Let's define the 7 Strassen terms explicitly for the 2x2 case.
    # Term r contributes u_r[i,j] * v_r[j,k] * w_r[i,k] to the sum for (i,j,k).
    # We need sum_{r=1}^7 u_r[i,j] v_r[j,k] w_r[i,k] = 1 for all i,j,k in {0,1}.
    
    # This is equivalent to decomposing the 2x2x2 "all-ones" tensor?
    # No, the matmul tensor has T[i,j,k]=1 for ALL i,j,k.
    # Wait, is the matmul tensor the all-ones tensor?
    # T[i*m+j, j*p+k, k*n+i] = 1.
    # Yes, for every valid triple (i,j,k), the entry is 1.
    # So we are decomposing the all-ones tensor of size 2x2x2 into 7 rank-1 terms.
    # The rank of the all-ones tensor of size dxdxd is 1 (it's just 1*1*1... wait).
    # The all-ones tensor J of size 2x2x2 has J[i,j,k]=1.
    # J = vec(1) vec(1)^T vec(1)^T? No.
    # J[i,j,k] = 1.
    # A rank-1 tensor is u_i v_j w_k.
    # If we take u=1, v=1, w=1, we get 1 for all entries.
    # So the rank of the all-ones tensor is 1?
    # No, the matrix multiplication tensor is NOT the all-ones tensor in the index space (i,j,k).
    # The tensor T is defined on indices (a,b,c) where a=i*m+j, b=j*p+k, c=k*n+i.
    # T[a,b,c] is non-zero only if the indices are consistent.
    # For the sub-block i,j,k in {0,1}, the indices are:
    # a in {0,1,3,4}, b in {0,1,4,5}, c in {0,1,3,4}.
    # The non-zero entries are at (0,0,0), (0,1,1), (0,4,3), (0,5,4) ... wait.
    # Let's list them:
    # i=0, j=0, k=0 -> a=0, b=0, c=0
    # i=0, j=0, k=1 -> a=0, b=1, c=3
    # i=0, j=1, k=0 -> a=1, b=4, c=0
    # i=0, j=1, k=1 -> a=1, b=5, c=3
    # i=1, j=0, k=0 -> a=3, b=0, c=1
    # i=1, j=0, k=1 -> a=3, b=1, c=4
    # i=1, j=1, k=0 -> a=4, b=4, c=1
    # i=1, j=1, k=1 -> a=4, b=5, c=4
    
    # These 8 entries are 1. All other entries in the sub-block's bounding box are 0.
    # This is exactly the structure of the 2x2 matmul tensor.
    # The rank of the 2x2 matmul tensor is 7 (Strassen).
    
    # So we need to find 7 vectors (u_r, v_r, w_r) such that their sum reconstructs this sparse tensor.
    
    # Let's use the known Strassen decomposition for the 2x2 matmul tensor.
    # The 7 terms are:
    # 1. u=[1,0;0,1], v=[1,0;0,1], w=[1,1;1,1] -> (A00+A11)(B00+B11)
    #    Wait, the indices for u are (i,j).
    #    u1 = [[1,0],[0,1]]
    #    v1 = [[1,0],[0,1]]
    #    w1 = [[1,1],[1,1]]
    #    Check (0,0,0): 1*1*1 = 1. (0,0,1): 1*0*1 = 0. (0,1,0): 0*1*1 = 0. (0,1,1): 0*0*1 = 0.
    #    (1,0,0): 0*1*1 = 0. (1,0,1): 0*0*1 = 0. (1,1,0): 1*1*1 = 1. (1,1,1): 1*0*1 = 0.
    #    This term contributes to (0,0,0) and (1,1,0).
    #    But we need (1,1,1) to be 1. And (0,1,1) to be 1.
    #    This specific term doesn't cover all. We sum 7 such terms.
    
    # Let's hardcode the 7 Strassen terms for the 2x2x2 matmul tensor.
    # Reference: "A note on the tensor rank of matrix multiplication"
    # The 7 rank-1 tensors are:
    
    # Term 1:
    # u1 = [[1, 0], [0, 1]]
    # v1 = [[1, 0], [0, 1]]
    # w1 = [[1, 1], [1, 1]]
    
    # Term 2:
    # u2 = [[0, 0], [1, 1]]
    # v2 = [[1, 0], [0, 0]]
    # w2 = [[0, 0], [1, 0]]
    
    # Term 3:
    # u3 = [[1, 0], [0, 0]]
    # v3 = [[0, 0], [1, -1]]
    # w3 = [[0, 0], [1, 0]]
    
    # Term 4:
    # u4 = [[0, 0], [0, 1]]
    # v4 = [[0, 1], [-1, 0]]
    # w4 = [[0, 1], [0, 0]]
    
    # Term 5:
    # u5 = [[1, 1], [0, 0]]
    # v5 = [[0, 0], [0, 1]]
    # w5 = [[0, 1], [0, 0]]
    
    # Term 6:
    # u6 = [[-1, 0], [1, 0]]
    # v6 = [[1, 0], [1, 0]]
    # w6 = [[0, 0], [1, 0]]
    
    # Term 7:
    # u7 = [[0, 1], [-1, 0]]
    # v7 = [[0, 1], [0, 1]]
    # w7 = [[0, 1], [0, 0]]
    
    # Let's verify this set numerically.
    # We create a 2x2x2 tensor S and check if it matches the matmul tensor.
    
    S = np.zeros((2, 2, 2))
    terms = [
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([[1,1],[1,1]])),
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,0],[1,0]])),
        (np.array([[1,0],[0,0]]), np.array([[0,0],[1,-1]]), np.array([[0,0],[1,0]])),
        (np.array([[0,0],[0,1]]), np.array([[0,1],[-1,0]]), np.array([[0,1],[0,0]])),
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[0,1],[0,0]])),
        (np.array([[-1,0],[1,0]]), np.array([[1,0],[1,0]]), np.array([[0,0],[1,0]])),
        (np.array([[0,1],[-1,0]]), np.array([[0,1],[0,1]]), np.array([[0,1],[0,0]]))
    ]
    
    for u, v, w in terms:
        # Add outer product
        S += np.einsum('ij,jk,ik->ijk', u, v, w)
        
    # The target tensor for 2x2 matmul has 1s at all (i,j,k).
    target = np.ones((2,2,2))
    
    if not np.allclose(S, target):
        # If the hardcoded terms are wrong, fall back to schoolbook
        # But let's assume they are correct or fix them.
        # Actually, let's just use schoolbook for safety if verification fails.
        pass
    
    # For now, let's assume the Strassen terms are correct and proceed to build the 35-rank decomposition.
    # We will verify the full 3x3x4 decomposition at the end.
    
    col = 0
    # Embed Strassen terms into U, V, W
    for u, v, w in terms:
        for i in range(2):
            for j in range(2):
                U[i*m + j, col] = u[i,j]
        for j in range(2):
            for k in range(2):
                V[j*p + k, col] = v[j,k]
        for k in range(2):
            for i in range(2):
                W[k*n + i, col] = w[i,k]
        col += 1
        
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Verify the full decomposition
    # Build the full tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
                
    # Compute decomposition
    D = np.einsum('ar,br,cr->abc', U, V, W)
    
    if np.allclose(D, T):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook rank 36
        rank_fb = 36
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, col] = 1
                    V_fb[j*p + k, col] = 1
                    W_fb[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }

# EVOLVE-BLOCK-END
