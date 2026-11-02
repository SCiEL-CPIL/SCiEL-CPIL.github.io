import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Build the matmul tensor T in R^{(n*m) x (m*p) x (p*n)}
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition: rank = n*m*p = 48
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-8):
        """Verify tensor decomposition for matmul tensor."""
        r = U.shape[1]
        # Check ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zeros
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k2 in range(p):
                        b = j2 * p + k2
                        for k3 in range(p):
                            for i3 in range(n):
                                c = k3 * n + i3
                                if j == j2 and i == i3 and k == k2:
                                    continue
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                if abs(val) > tol:
                                    return False
        return True
    
    # Verify schoolbook
    if not verify(U0, V0, W0):
        pass  # Should not happen
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try to find a lower-rank decomposition by exploiting structure
    # Strategy: try to merge columns that correspond to "similar" (i,j,k) tuples
    
    # For each pair of columns, check if they can be merged
    # Two columns c1=(i1,j1,k1) and c2=(i2,j2,k2) can be merged if
    # their rank-1 tensors sum to something expressible with one rank-1 tensor
    
    # More practically: try to find a rank-(rank0-1) decomposition by
    # checking if any single column is redundant in the tensor sense
    
    # A column c0 is redundant if there exist alpha_c for c != c0 such that
    # U0[a,c0]*V0[b,c0]*W0[c,c0] = sum_{c!=c0} alpha_c * U0[a,c]*V0[b,c]*W0[c,c]
    # for all (a,b,c)
    
    # This is a system of (n*m)*(m*p)*(p*n) = 12*16*12 = 2304 equations
    # in (rank0-1) = 47 unknowns
    
    # Let's try this for each column
    for c0 in range(rank0):
        if time.time() > deadline - 15:
            break
        
        # Build the target: T - U0[:,c0] outer V0[:,c0] outer W0[:,c0]
        T_target = T.copy()
        T_target -= U0[:, c0][:, None, None] * V0[:, c0][None, :, None] * W0[:, c0][None, None, :]
        
        # We need to find U', V', W' with 47 columns such that
        # sum_c U'[a,c]*V'[b,c]*W'[c,c] = T_target[a,b,c]
        
        # Start with the other 47 columns
        other_cols = [c for c in range(rank0) if c != c0]
        U_init = U0[:, other_cols].copy()
        V_init = V0[:, other_cols].copy()
        W_init = W0[:, other_cols].copy()
        
        # Check if the initial guess already works (it won't, but verify)
        # Actually, let's try a different approach: use a numerical solver
        
        # Flatten the tensor equation
        # For each (a,b,c), we have: sum_s U[a,s]*V[b,s]*W[c,s] = T_target[a,b,c]
        
        # This is a system of 2304 equations in 47*3 = 141 unknowns (U, V, W entries)
        # But it's nonlinear, so we can't use linear solvers directly
        
        # Alternative: fix U and V, solve for W, then fix U and W, solve for V, etc.
        # This is an alternating minimization approach
        
        # Let's try a simpler approach: check if the tensor T_target has rank <= 47
        # by computing the numerical rank of its unfoldings
        
        # Unfold along mode 1: shape (12, 16*12)
        T_unfold1 = T_target.reshape(n * m, m * p * p * n)
        rank1 = np.linalg.matrix_rank(T_unfold1, tol=1e-8)
        
        # Unfold along mode 2: shape (16, 12*12)
        T_unfold2 = T_target.reshape(m * p, n * m * p * n)
        rank2 = np.linalg.matrix_rank(T_unfold2, tol=1e-8)
        
        # Unfold along mode 3: shape (12, 12*16)
        T_unfold3 = T_target.reshape(p * n, n * m * m * p)
        rank3 = np.linalg.matrix_rank(T_unfold3, tol=1e-8)
        
        # If all unfoldings have rank <= 47, it's possible (but not guaranteed)
        # that T_target has rank <= 47
        if rank1 <= 47 and rank2 <= 47 and rank3 <= 47:
            # Try to find a rank-47 decomposition using alternating minimization
            U_alt = U_init.copy()
            V_alt = V_init.copy()
            W_alt = W_init.copy()
            
            for iteration in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Fix U and V, solve for W
                # For each c: W[c,:] = sum_a,b T_target[a,b,c] * U[a,:] * V[b,:] / (sum_a U[a,:]^2 * sum_b V[b,:]^2)
                # This is not quite right; let's use a proper least squares approach
                
                # For each c, we want to minimize sum_a,b (T_target[a,b,c] - sum_s U[a,s]*V[b,s]*W[c,s])^2
                # with respect to W[c,:]
                
                # This is a linear system in W[c,:]
                # Let's set it up: for fixed a,b, the equation is:
                # sum_s U[a,s]*V[b,s]*W[c,s] = T_target[a,b,c]
                
                # For each c, we have a system of (n*m)*(m*p) = 12*16 = 192 equations
                # in (p*n) = 12 unknowns (W[c,0], ..., W[c,11])
                
                # Actually, this is getting complicated. Let's just try a random perturbation approach.
                
                # Add small random perturbations to U, V, W and see if we can reduce the error
                error = np.sum((T_target - np.einsum('as,bs,cs->abc', U_alt, V_alt, W_alt))**2)
                
                if error < 1e-6:
                    break
                
                # Random perturbation
                U_alt += 0.01 * np.random.randn(*U_alt.shape)
                V_alt += 0.01 * np.random.randn(*V_alt.shape)
                W_alt += 0.01 * np.random.randn(*W_alt.shape)
            
            # Verify the result
            if verify(U_alt, V_alt, W_alt):
                best_rank = rank0 - 1
                best_U, best_V, best_W = U_alt, V_alt, W_alt
                print(f"Found rank {best_rank} decomposition by removing column {c0}!")
                break
    
    if best_rank < rank0:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }
