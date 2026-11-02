import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Generate the target tensor T of shape (n*m, m*p, p*n) is wrong.
    # T is a 3-tensor. We flatten it to a matrix for linear algebra.
    # T[i, j, k] = 1 for all i,j,k? No.
    # T corresponds to the map A, B -> C.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T has indices (i, j, k) for A, B, C?
    # Standard: T in V_A* x V_B* x V_C.
    # T[i, j, k] is the coefficient of A[i, j] B[j, k] in C[i, k].
    # So T[i, j, k] = 1 if we consider the basis e_i for C, e_j for B, e_i for A?
    # Let's define the tensor as a matrix M of size (n*m) x (m*p) for each k? No.
    # Let's use the flattening:
    # Vectorize A (size nm), B (size mp), C (size np).
    # The tensor is a 3D array T[nm, mp, np] such that C = sum_{a,b} T[a,b,c] A[a] B[b].
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k).
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
                
    # We want to find U (nm x R), V (mp x R), W (np x R)
    # such that T = sum_r U_r (outer) V_r (outer) W_r
    # i.e. T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_school[a_idx, col] = 1
                V_school[b_idx, col] = 1
                W_school[c_idx, col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank
    # Search R from best_rank - 1 down to some limit
    # Use random restarts and ALS
    
    np.random.seed(42)
    
    for R in range(best_rank - 1, max(10, best_rank - 10), -1):
        if time.time() > deadline - 10:
            break
            
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V randomly
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.zeros((p*n, R))
            
            # Alternating Least Squares
            for it in range(100):
                if time.time() > deadline - 2:
                    break
                    
                # Solve for W given U, V
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For each r, W[:, r] is determined by projecting T onto U[:, r] (outer) V[:, r]
                # Let A_r = U[:, r] (outer) V[:, r] (shape nm x mp)
                # We want sum_r A_r (outer) W[:, r] = T
                # This is a linear system for W.
                # Flatten T to vector t (size nm*mp*np)
                # Flatten W to vector w (size np*R)
                # The operator is a Kronecker product.
                
                # More efficient: For each c, T[:, :, c] = sum_r U[:, r] V[:, r]^T W[c, r]
                # T[:, :, c] is a matrix of size nm x mp.
                # Let M_c = T[:, :, c]
                # M_c = sum_r W[c, r] (U[:, r] V[:, r]^T)
                # This is a linear system for W[c, :] for each c.
                # Since the system is the same for all c (coefficients U, V are same),
                # we can solve for all c at once.
                
                # Build the design matrix for the linear system.
                # Variables: W[c, r] for c in 0..np-1, r in 0..R-1.
                # Equation for each a, b:
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                
                # We can solve for W by least squares.
                # Reshape U, V to form the basis.
                # Let B_r = U[:, r] (outer) V[:, r] (size nm*mp)
                # We want sum_r W[:, r] (outer) B_r = T (reshaped to nm*mp x np)
                
                T_mat = T.reshape(n*m*m*p, p*n) # Wrong shape
                # T is (nm, mp, np). Reshape to (nm*mp, np)
                T_flat = T.reshape(n*m*m*p, p*n)
                
                # Design matrix D of size (nm*mp) x (np*R)
                # D[:, c*R + r] = B_r[c] ? No.
                # The term is W[c, r] * B_r[a, b].
                # So for a given (a,b), the equation is sum_{c,r} D[(a,b), c*R+r] W_flat[c*R+r] = T_flat[(a,b), c]?
                # No, T_flat index is (a*b, c).
                # So for each c, we have a system:
                # sum_r B_r * W[c, r] = T[:, :, c]
                
                # We can solve this by stacking the systems for all c.
                # But B_r is the same for all c.
                # So we can form a matrix X of size (nm*mp) x R, where X[:, r] = B_r.flatten()
                # Then for each c, T[:, :, c].flatten() = X * W[c, :]
                # So W = pinv(X) * T_flat.T ?
                
                X = np.zeros((n*m*m*p, R))
                for r in range(R):
                    B_r = np.outer(U[:, r], V[:, r])
                    X[:, r] = B_r.flatten()
                
                # T_flat is (nm*mp, np). We want W (np, R).
                # T_flat.T is (np, nm*mp).
                # W = pinv(X) @ T_flat.T
                try:
                    W = np.linalg.pinv(X) @ T_flat.T
                except:
                    break
                
                # Solve for U given V, W
                # T[a, b, c] = sum_r U[a, r] (V[b, r] W[c, r])
                # For each a, T[a, :, :] = sum_r U[a, r] (V[:, r] (outer) W[:, r])
                # Let C_r = V[:, r] (outer) W[:, r] (size mp*np)
                # T[a, :, :] is (mp, np).
                # T_flat_a = T[a, :, :].flatten()
                # sum_r U[a, r] C_r.flatten() = T_flat_a
                # So for each a, T_flat[a, :] = Y * U[a, :]
                # Y is (mp*np) x R, Y[:, r] = C_r.flatten()
                
                Y = np.zeros((m*p*p*n, R))
                for r in range(R):
                    C_r = np.outer(V[:, r], W[:, r])
                    Y[:, r] = C_r.flatten()
                
                T_flat_a = T.reshape(n*m, m*p*p*n)
                try:
                    U = np.linalg.pinv(Y) @ T_flat_a.T
                except:
                    break
                    
                # Solve for V given U, W
                # T[a, b, c] = sum_r (U[a, r] W[c, r]) V[b, r]
                # For each b, T[:, b, :] = sum_r V[b, r] (U[:, r] (outer) W[:, r])
                # Let D_r = U[:, r] (outer) W[:, r] (size nm*np)
                # T[:, b, :].flatten() = sum_r V[b, r] D_r.flatten()
                # So for each b, T_flat_b = Z * V[b, :]
                # Z is (nm*np) x R, Z[:, r] = D_r.flatten()
                
                Z = np.zeros((n*m*p*n, R))
                for r in range(R):
                    D_r = np.outer(U[:, r], W[:, r])
                    Z[:, r] = D_r.flatten()
                    
                T_flat_b = T.reshape(m*p, n*m*p*n)
                try:
                    V = np.linalg.pinv(Z) @ T_flat_b.T
                except:
                    break
                
            # Check residual
            if time.time() > deadline - 1:
                continue
                
            # Compute T_reconstructed
            T_rec = np.zeros_like(T)
            for r in range(R):
                T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            residual = np.linalg.norm(T_rec - T)
            
            if residual < 1e-6:
                # Found a solution
                # Check if it can be made exact (rational)
                # For now, accept it if rank is lower
                if R < best_rank:
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W
                    # We can stop searching for lower R if we find one?
                    # Or continue to find even lower?
                    # Given time, maybe stop?
                    # But we want minimal.
                    pass

    # Return the best found
    # Convert to lists of integers/half-integers if possible
    # If not, return as is (though task says integer/half-integer)
    # The schoolbook solution is integers.
    # If we found a non-integer solution, we might need to round or reject.
    # For safety, if the found solution is not close to integers/half-integers, we might keep schoolbook.
    
    # Simple check: if best_rank < rank_schoolbook, try to snap to rationals
    if best_rank < rank_schoolbook:
        # Try to snap to nearest 0.5
        U_snapped = np.round(best_U * 2) / 2
        V_snapped = np.round(best_V * 2) / 2
        W_snapped = np.round(best_W * 2) / 2
        
        T_rec = np.zeros_like(T)
        for r in range(best_rank):
            T_rec += np.einsum('a,b,c->abc', U_snapped[:, r], V_snapped[:, r], W_snapped[:, r])
        
        residual = np.linalg.norm(T_rec - T)
        if residual < 1e-6:
            best_U = U_snapped
            best_V = V_snapped
            best_W = W_snapped
        else:
            # Revert to schoolbook
            best_U = U_school
            best_V = V_school
            best_W = W_school
            best_rank = rank_schoolbook

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
