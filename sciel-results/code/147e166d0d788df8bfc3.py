import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 72 # Start with schoolbook
    best_result = None
    best_rank = target_rank
    
    # Precompute the target tensor indices
    # T[i, j, k] = 1 for all i in n, j in m, k in p
    # We flatten U, V, W into vectors for optimization
    
    # Shapes:
    # U: (n*m, r)
    # V: (m*p, r)
    # W: (p*n, r)
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Try to find a decomposition for ranks less than 72
    # We iterate from 71 downwards
    for r in range(71, 49, -1):
        if time.time() > deadline:
            break
            
        # Define the loss function
        def loss(params):
            # params: vector of length r * (n*m + m*p + p*n)
            size_u = n * m
            size_v = m * p
            size_w = p * n
            
            U = params[0:size_u*r].reshape(r, size_u).T # (size_u, r)
            V = params[size_u*r:size_u*r + size_v*r].reshape(r, size_v).T # (size_v, r)
            W = params[size_u*r + size_v*r:].reshape(r, size_w).T # (size_w, r)
            
            # Compute the decomposed tensor
            # We need to check if sum_c U[i*m+j, c] * V[j*p+k, c] == 1
            # And W is just the output coefficient?
            # The condition is that the tensor formed by U, V, W equals the target tensor.
            # Target tensor T has entries T[i,j,k] = 1.
            # Decomposed tensor D has entries D[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]?
            # NO. The tensor decomposition is T = sum U_c \otimes V_c \otimes W_c.
            # This means T[i,j,k] = sum_c U_c[i,j] V_c[j,k] W_c[k,i]?
            # Wait, the indices of W are (k, i).
            # So T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i].
            
            # We want D[i,j,k] = 1 for all i,j,k.
            
            # Efficient computation:
            # D is a tensor of size n x m x p.
            # We can compute it by looping over c.
            # D = 0
            # for c in range(r):
            #   Uc = U[:, c] # vector of size nm
            #   Vc = V[:, c] # vector of size mp
            #   Wc = W[:, c] # vector of size pn
            #   # Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
            #   # This is a rank 1 update to D.
            
            # However, we only care about the error.
            # Error = sum_{i,j,k} (D[i,j,k] - 1)^2
            
            # We can compute D as a matrix product?
            # Let's just loop over c for simplicity, r is small.
            D = np.zeros((n, m, p))
            for c in range(r):
                Uc = U[:, c]
                Vc = V[:, c]
                Wc = W[:, c]
                
                # Reshape Uc to (n, m)
                Uc_mat = Uc.reshape(n, m)
                # Reshape Vc to (m, p)
                Vc_mat = Vc.reshape(m, p)
                # Reshape Wc to (p, n)
                Wc_mat = Wc.reshape(p, n)
                
                # The term is Uc_mat[i,j] * Vc_mat[j,k] * Wc_mat[k,i]
                # This is element-wise product summed over j? No.
                # The tensor entry is the sum over c.
                # For a fixed c, the contribution to D[i,j,k] is Uc_mat[i,j] * Vc_mat[j,k] * Wc_mat[k,i].
                
                # We can compute this as:
                # Temp[i,j] = Uc_mat[i,j]
                # Temp2[j,k] = Vc_mat[j,k] * Wc_mat[k,i]? No, i is fixed.
                
                # It's easier to just use einsum or loops.
                # D += Uc_mat[:, :, None] * Vc_mat[None, :, :] * Wc_mat[None, :, :]?
                # Wc_mat is (p, n). We need Wc_mat[k, i].
                # So Wc_mat.T is (n, p). Wc_mat.T[i, k].
                # So term = Uc_mat[i, j] * Vc_mat[j, k] * Wc_mat.T[i, k].
                
                # D[i,j,k] += Uc_mat[i,j] * Vc_mat[j,k] * Wc_mat.T[i,k]
                
                # Vectorized:
                # Uc_mat[:, :, None] is (n, m, 1)
                # Vc_mat[None, :, :] is (1, m, p)
                # Wc_mat.T[:, None, :] is (n, 1, p)
                
                D += Uc_mat[:, :, None] * Vc_mat[None, :, :] * Wc_mat.T[:, None, :]
            
            error = np.sum((D - 1.0)**2)
            return np.sqrt(error) # Return scalar for least_squares? No, least_squares expects vector.
            # We can return the flattened error vector.
        
        # least_squares expects a vector of residuals.
        # We can return the flattened (D - 1).
        def residuals(params):
            size_u = n * m
            size_v = m * p
            size_w = p * n
            
            U = params[0:size_u*r].reshape(r, size_u).T
            V = params[size_u*r:size_u*r + size_v*r].reshape(r, size_v).T
            W = params[size_u*r + size_v*r:].reshape(r, size_w).T
            
            D = np.zeros((n, m, p))
            for c in range(r):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
            
            return (D - 1.0).flatten()
        
        # Initialize params
        # Use schoolbook initialization for the first r terms?
        # Schoolbook has 72 terms.
        # If r < 72, we can't use all.
        # We can use a random initialization.
        np.random.seed(42)
        params0 = np.random.randn(r * (n*m + m*p + p*n)) * 0.1
        
        try:
            result = least_squares(residuals, params0, method='lm', max_nfev=5000)
            if result.success and result.cost < 1e-8:
                # Valid decomposition found
                size_u = n * m
                size_v = m * p
                size_w = p * n
                U = result.x[0:size_u*r].reshape(r, size_u).T
                V = result.x[size_u*r:size_u*r + size_v*r].reshape(r, size_v).T
                W = result.x[size_u*r + size_v*r:].reshape(r, size_w).T
                
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
                
        except Exception as e:
            pass
            
    # If no better rank found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
