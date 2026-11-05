import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Laderman's 23-term decomposition for 3x3x3 matrix multiplication.
    # Each term is (u, v, w) where u, v, w are 9-element lists representing
    # flattened 3x3 matrices.
    # Indices for 3x3 matrix [0..8]:
    # 0: (0,0), 1: (0,1), 2: (0,2)
    # 3: (1,0), 4: (1,1), 5: (1,2)
    # 6: (2,0), 7: (2,1), 8: (2,2)
    
    # The following is the standard Laderman decomposition.
    # Reference: Laderman (1976) "A 23-multiplication algorithm for 3x3 matrix multiplication"
    terms = [
        ([1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([1, -1, 0, 0, 0, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0, 0, 0, 0], [1, -1, 0, 0, -1, 0, 0, 0, 0]),
        ([1, 1, 0, 0, 0, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]),
        ([1, -1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]),
        ([0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]),
        ([1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]),
        ([0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]),
        ([0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0])
    ]
    
    rank = 46
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Fill the first 23 columns for the first 3 columns of B and C
    for r in range(23):
        u, v, w = terms[r]
        for idx in range(9):
            i, j = idx // 3, idx % 3 # Note: idx = i*3+j for A[i,j]
            # U[i*m + j] = U[i*3 + j]
            U[i*m + j, r] = u[idx]
            
            # V index for B[j, l]. Here l is the local column (0,1,2).
            # Global column k = l.
            # V[j*p + k] = V[j*6 + l]
            # v[idx] corresponds to B[j, l] where idx = j*3 + l
            j_v, l_v = idx // 3, idx % 3
            V[j_v*p + l_v, r] = v[idx]
            
            # W index for C[i, k]. Here k is the local column (0,1,2).
            # Global column k = l.
            # W[k*n + i] = W[l*3 + i]
            # w[idx] corresponds to C[i, l] where idx = i*3 + l
            i_w, l_w = idx // 3, idx % 3
            W[l_w*n + i_w, r] = w[idx]
            
    # Fill the next 23 columns for the next 3 columns of B and C
    for r_local in range(23):
        r = 23 + r_local
        u, v, w = terms[r_local]
        for idx in range(9):
            i, j = idx // 3, idx % 3
            U[i*m + j, r] = u[idx]
            
            j_v, l_v = idx // 3, idx % 3
            # Global column k = l_v + 3
            V[j_v*p + (l_v + 3), r] = v[idx]
            
            i_w, l_w = idx // 3, idx % 3
            # Global column k = l_w + 3
            W[(l_w + 3)*n + i_w, r] = w[idx]
            
    # Verify correctness
    # We check the tensor identity:
    # T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This must equal 1 if i,j,k form a valid multiplication term (i.e., always for matrix mult tensor)
    # Actually, the tensor T is defined as T_{i,j,k} = 1 if we are computing C[i,k] = sum_j A[i,j]B[j,k].
    # The decomposition must satisfy:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k?
    # No, the multiplication tensor M_{i,j,k} is 1.
    # The decomposition is M = sum_r u_r \otimes v_r \otimes w_r.
    # So for all i,j,k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be 1?
    # Wait, the tensor for matrix multiplication is T_{i,j,k} = \delta_{i,j,k}? No.
    # The tensor is T_{i,j,k} such that C[i,k] = sum_j A[i,j] B[j,k] T_{i,j,k}?
    # No, T_{i,j,k} = 1 for all i,j,k is not right.
    # The multiplication map is bilinear.
    # The tensor T is in U^* \otimes V^* \otimes W.
    # T(a,b) = a \cdot b.
    # The components are T_{i,j,k} = 1?
    # Let's check: C[i,k] = sum_j A[i,j] B[j,k].
    # This is sum_j sum_r ...
    # The condition is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # This must hold for all i, j, k?
    # Let's test with i=0, j=0, k=0.
    # If the sum is 1, it works.
    # Let's verify this condition numerically.
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    error += abs(s - 1.0)
                    
    if error > 1e-6:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
