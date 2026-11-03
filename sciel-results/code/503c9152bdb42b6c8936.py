import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # Tensor T (n*m, m*p, p*n) -> Actually T[i,j, k*n+l] ?
    # Convention: W[k*n+i, r].
    # The multiplication is C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # T is a 3-tensor T[i, j, k] = 1 if we want to compute A[i,j]*B[j,k] to add to C[i,k].
    # Wait, the tensor for matrix multiplication M_{n,m,p} is defined such that:
    # sum_r U[i,j, r] V[j,k, r] W[i,k, r] = sum_j A[i,j] B[j,k]
    # Here U, V, W are the factor matrices.
    # U has indices (i, j). V has indices (j, k). W has indices (i, k).
    # In the task, U is (n*m, rank). So U[i*m + j, r].
    # V is (m*p, rank). So V[j*p + k, r].
    # W is (p*n, rank). So W[k*n + i, r]. Note: W index is k*n + i, which maps to output C[i,k].
    
    # Let's build the target tensor T_flat or T_3d.
    # T_3d[i, j, k] corresponds to the coefficient of A[i,j] B[j,k] in C[i,k].
    # It is 1 for all i,j,k.
    # The decomposition is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta? No.
    # The tensor T has T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
    
    # Flatten T for optimization:
    # x = vec(U), y = vec(V), z = vec(W)
    # f(x,y,z) = sum_r x[i*m+j, r] y[j*p+k, r] z[k*n+i, r] - 1
    
    target = np.ones((n, m, p))
    
    def compute_residual(params):
        # params: flattened [U, V, W]
        # R is current rank
        # Split params
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(n, m, R)
        V = V_flat.reshape(m, p, R)
        W = W_flat.reshape(p, n, R)
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ik' ? No, we need to check against T[i,j,k]
        # The term is U[i,j,r] V[j,k,r] W[k,i,r].
        # Let's sum over r.
        # Result C[i,k] = sum_r sum_j U[i,j,r] V[j,k,r] W[k,i,r] ?
        # No, the standard tensor decomposition of matrix multiplication T_{i,j,k} is:
        # T = sum_r u_r (vector in A-space) \otimes v_r (vector in B-space) \otimes w_r (vector in C-space)
        # Here A-space is n*m, B-space is m*p, C-space is p*n.
        # T[i*m+j, j*p+k, k*n+i] = 1.
        # The product is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        
        # We need this sum to be 1 for all i,j,k.
        
        # Let's compute the tensor S[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: W is indexed by [k, i] in the logical sense, but stored as W[k*n+i].
        
        # Efficient computation:
        # S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # This is a contraction.
        # We can iterate or use einsum.
        # 'ijr,jkr,ki r -> ijk'
        
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        
        # Residual is S - target
        return (S - target).flatten()

    def verify_exact(U, V, W):
        # Check if U,V,W entries are half-integers
        # And if the reconstruction is exact
        # U, V, W are 3D arrays (n,m,R), etc.
        
        # Check half-integer property
        for mat in [U, V, W]:
            # mat * 2 should be integer
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        
        # Check exact reconstruction
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        if not np.allclose(S, target, atol=1e-9):
            return False
            
        # Convert to 2D lists as required by interface
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        U_2d = U.reshape(n*m, R).tolist()
        V_2d = V.reshape(m*p, R).tolist()
        W_2d = W.reshape(p*n, R).tolist()
        
        # Ensure they are lists of lists of numbers (int or float)
        return True

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n, m, rank))
        V = np.zeros((m, p, rank))
        W = np.zeros((p, n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        return U.reshape(n*m, rank).tolist(), V.reshape(m*p, rank).tolist(), W.reshape(p*n, rank).tolist()

    U_fb, V_fb, W_fb = schoolbook()
    
    # Search for lower rank
    start_time = time.time()
    deadline = start_time + 100 # 20s margin
    
    # Try ranks from 54 down to 20
    # We can skip some if they seem unlikely, but let's try a range.
    # 54 is safe.
    # Known: 3x3x3 is 19. 3x3x6 might be 38?
    # Let's try 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53.
    
    candidate_ranks = list(range(53, 15, -1))
    
    for R in candidate_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        for restart in range(3):
            if time.time() > deadline:
                break
                
            # Initialize
            # Small random integers or half-integers?
            # Random uniform in [-1, 1]
            np.random.seed(42 + restart + R)
            U_init = np.random.randn(n, m, R)
            V_init = np.random.randn(m, p, R)
            W_init = np.random.randn(p, n, R)
            
            # Scale to make initial residual not huge?
            # The target is 1.
            # Sum of products of 3 random vars ~ 1.
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(compute_residual, params, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            # Check residual
            if res.cost < 1e-10: # cost is 0.5 * sum(residuals^2)
                # Extract
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U+size_V]
                W_flat = res.x[size_U+size_V:]
                
                U = U_flat.reshape(n, m, R)
                V = V_flat.reshape(m, p, R)
                W = W_flat.reshape(p, n, R)
                
                # Round to nearest half-integer
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                if verify_exact(U_round, V_round, W_round):
                    U_ret = U_round.reshape(n*m, R).tolist()
                    V_ret = V_round.reshape(m*p, R).tolist()
                    W_ret = W_round.reshape(p*n, R).tolist()
                    return {"U": U_ret, "V": V_ret, "W": W_ret}

    # If search fails, return schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
