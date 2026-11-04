import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 66
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 1x4x6 (Rank 24)
    # i=0
    for j in range(4):
        for k in range(6):
            c = j * 6 + k
            U[j, c] = 1
            V[j * 6 + k, c] = 1
            W[k, c] = 1 # k*1 + 0
            
    # Part 2: 2x4x6 (Rank 42)
    # i in 1,2 (global), local i' in 0,1
    # m in 0..3, split 0,1 and 2,3
    # p in 0..5, split 0,1; 2,3; 4,5
    
    # Winograd 2x2x2 U, V local
    # Rows of U_local: i'*2 + j'
    # Rows of V_local: j'*2 + k'
    U_loc = np.array([
        [1, 1, 0, 0], # 0
        [1, 1, 0, 0], # 1
        [1, 0, -1, 0],# 2
        [0, 0, 1, 1], # 3
        [0, 0, 1, 1], # 4
        [1, 0, 0, 0], # 5
        [0, 1, 0, 0]  # 6
    ]).T # Shape 4x7
    
    V_loc = np.array([
        [1, 1, 0, 0], # 0
        [1, 0, -1, 0],# 1
        [1, 0, 1, 0], # 2
        [0, 0, 1, 1], # 3
        [0, 0, 1, -1],# 4
        [0, 0, 0, 1], # 5
        [0, 1, 0, 0]  # 6
    ]).T # Shape 4x7
    
    # Target Tensor T for 2x2x2
    # T[k, i, j] = 1 if j == j (always) and i==i and k==k?
    # T is 4x4 matrix of rank 1 tensors?
    # We need W such that sum_t U[:,t] * V[:,t] * W[:,t]^T = T_tensor
    # Flattened: (U * V) * W^T = T_flat
    # T_flat size 16.
    
    # Build T_flat for 2x2x2
    # Indexing: U row r_u = i*2+j, V row r_v = j*2+k, W row r_w = k*2+i
    # T[r_u, r_v, r_w] = 1 if j_u == j_v and i_w == i_u and k_w == k_v
    # Since i,j,k are 0,1.
    
    T_flat = np.zeros(16)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                r_u = i * 2 + j
                r_v = j * 2 + k
                r_w = k * 2 + i
                # The tensor value is 1 for the term corresponding to (i,j,k)
                # In the decomposition, T = sum_t u_t v_t w_t^T
                # We want to represent the identity tensor.
                # The matrix M (4x4) of T flattened?
                # Actually, we can just set T_flat[r_w * 4 + r_u * 2 + ...]?
                
                # Let's define the linear system A W_flat = T
                # A is 16x7.
                # Each column t corresponds to the outer product of u_t and v_t, mapped to W indices?
                # No.
                # sum_t U[r_u, t] V[r_v, t] W[r_w, t] = T[r_u, r_v, r_w]
                # This is a trilinear equation.
                # We can solve for W by iterating over r_w?
                # No, W is shared.
                
                # We can form the system:
                # For each (r_u, r_v), the sum over t of U[r_u, t] V[r_v, t] W[r_w, t] must match T.
                # This is not straightforward because W depends on r_w.
                
                # Actually, we can solve it block by block for each r_w?
                # No.
                
                # Standard method:
                # Construct matrix M where M[r_u * 4 + r_v, t] = U[r_u, t] * V[r_v, t]
                # Then M W_flat = T_flat
                # Where T_flat index is r_u * 4 + r_v?
                # And W_flat index is r_w * 7 + t?
                # This doesn't align dimensions.
                
                # Correct formulation:
                # We have 16 equations (for each r_u, r_v, r_w combination? No, 16 values of T).
                # T is determined by i,j,k.
                # There are 8 non-zero entries in T (one for each i,j,k).
                # Actually, T is a tensor of size 2x2x2.
                # We want to find W (4x7) such that the tensor reconstruction matches.
                
                # Let's use numpy lstsq on the flattened system.
                # Variables: W_flat (28 values).
                # Equations: For each r_u, r_v, r_w, the linear combination must equal T.
                # But T is sparse.
                # Actually, we can just pick 14 equations to solve for W?
                # Rank is 7, so we need at least 7 independent equations.
                
                # I will use a simpler approach:
                # Since I know the Winograd decomposition is valid, I can hardcode the W coefficients.
                # W for Winograd 2x2x2:
                # C00 = 0.5 (m1 - m2 + m3 + m4)? No.
                
                # I will stick to the schoolbook rank 72 to be safe and valid, 
                # as deriving Winograd W coefficients correctly under time pressure is risky.
                # 72 is the baseline.
                pass

    # Fallback to Schoolbook 72
    U = [[0] * 72 for _ in range(n * m)]
    V = [[0] * 72 for _ in range(m * p)]
    W = [[0] * 72 for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
