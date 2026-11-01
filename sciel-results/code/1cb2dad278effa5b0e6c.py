import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_rank11_233():
    """
    Build a rank-11 decomposition for 2x3 x 3x3 matrix multiplication.
    
    The multiplication tensor T has components T[i,j,k] = 1 for all (i,j,k)
    where i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    A rank-1 term is u ⊗ v ⊗ w where:
      u is a vector in R^6 (indexed by (i,j))
      v is a vector in R^9 (indexed by (j,k))
      w is a vector in R^6 (indexed by (k,i) per trace convention W[k*n+i])
    
    The decomposition satisfies: for all (i,j,k),
      T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    
    where n=2, m=3, p=3.
    """
    n, m, p = 2, 3, 3
    
    # We need to find 11 triples (u_r, v_r, w_r) such that
    # for each (i,j,k): sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    # and for all other index combinations in the tensor structure, it works out.
    
    # Actually, the tensor for matrix multiplication M_{2,3,3} has a specific structure.
    # The tensor is T in R^{(n*m) x (m*p) x (p*n)} with
    # T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k' and i=i', else 0.
    # Wait, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor for A (n x m) times B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # As a tensor T in R^{(n*m) x (m*p) x (n*p)}, T[(i,j),(j',k),(i',k')] = 1 if j=j' and i=i' and k=k'
    # 
    # But the problem says W has shape (p*n, rank) with index W[k*n+i].
    # So W is indexed by (k,i) where k is the column index of C and i is the row index.
    # 
    # So the tensor is: T[(i,j), (j',k), (k,i')] = 1 if j=j' and i=i' and k=k'
    # Wait, that doesn't make sense dimensionally.
    
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i]"
    # So W has p*n entries, indexed as W[k*n + i] where k is in 0..p-1 and i is in 0..n-1.
    # This means W is indexed by (k, i) - the column and row of C.
    
    # For the decomposition to work:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum over the rank) U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # So: T[(i,j), (j',k), (k',i')] should satisfy:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # 
    # Wait, that's the Kronecker delta structure. So the tensor is:
    # T[a, b, c] where a=(i,j), b=(j',k), c=(k',i')
    # T[(i,j),(j',k),(k',i')] = 1 if j=j' AND k=k' AND i=i', else 0.
    
    # So for 2x3x3: n=2, m=3, p=3
    # U is 6x11, V is 9x11, W is 6x11
    # For each rank-1 term r, we have vectors u_r (6-dim), v_r (9-dim), w_r (6-dim)
    # such that sum_r u_r[a] * v_r[b] * w_r[c] = T[a,b,c] for all a,b,c.
    
    # This is a hard problem. Let me use a known construction.
    # 
    # One approach: use the fact that we can write the identity in a clever way.
    # 
    # Actually, let me try a different approach. I'll use optimization to find the rank-11 decomposition.
    
    from scipy.optimize import minimize
    
    # The tensor T has 6*9*6 = 324 components, but most are zero.
    # Non-zero components: for each (i,j,k) with i in 0..1, j in 0..2, k in 0..2:
    #   T[(i,j), (j,k), (k,i)] = 1
    # That's 18 non-zero components.
    
    # For a rank-r decomposition, we need:
    # For each of the 18 non-zero positions: sum_{s=1}^r U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    # For all other positions: sum_{s=1}^r U[i*m+j, s] * V[j'*p+k, s] * W[k'*n+i', s] = 0
    
    # This is a system of polynomial equations. Let me set up the optimization.
    
    # Total number of constraints: 6*9*6 = 324 (but many are automatically satisfied by structure)
    # Actually, the constraint is only that the tensor equals T.
    
    # Let me parameterize: U (6x11), V (9x11), W (6x11) -> 6*11 + 9*11 + 6*11 = 231 parameters
    
    # The objective is to minimize the squared error of the tensor reconstruction.
    
    # But this is a non-convex problem. Let me try a random restart approach.
    
    deadline = time.time() + 100  # Leave margin
    
    best_result = None
    best_err = float('inf')
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        np.random.seed(attempt)
        U = np.random.randn(6, 11) * 0.5
        V = np.random.randn(9, 11) * 0.5
        W = np.random.randn(6, 11) * 0.5
        
        # Build the target tensor
        T = np.zeros((6, 9, 6))
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    T[i*3+j, j*3+k, k*2+i] = 1.0
        
        # Define the error function
        def error(params):
            U_flat = params[0:66].reshape(6, 11)
            V_flat = params[66:165].reshape(9, 11)
            W_flat = params[165:231].reshape(6, 11)
            # Compute the tensor: sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s]
            # This is a 3-tensor product
            # T_recon[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
            # We can compute this as: for each s, outer(U[:,s], V[:,s]) * W[:,s]
            # More efficiently: einsum
            T_recon = np.einsum('as,bs,cs->abc', U_flat, V_flat, W_flat)
            return np.sum((T_recon - T)**2)
        
        def gradient(params):
            U_flat = params[0:66].reshape(6, 11)
            V_flat = params[66:165].reshape(9, 11)
            W_flat = params[165:231].reshape(6, 11)
            T_recon = np.einsum('as,bs,cs->abc', U_flat, V_flat, W_flat)
            diff = T_recon - T
            # dE/dU[a,s] = 2 * sum_{b,c} diff[a,b,c] * V[b,s] * W[c,s]
            dU = 2 * np.einsum('abc,bs,cs->as', diff, V_flat, W_flat)
            dV = 2 * np.einsum('abc,as,cs->bs', diff, U_flat, W_flat)
            dW = 2 * np.einsum('abc,as,bs->cs', diff, U_flat, V_flat)
            return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B
        result = minimize(error, params0, jac=gradient, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-10})
        
        if result.fun < best_err:
            best_err = result.fun
            best_result = result.x.copy()
        
        if best_err < 1e-6:
            break
    
    if best_result is None or best_err > 1e-4:
        # Fallback: return a schoolbook decomposition (rank 18)
        # But we need rank 11... Let me just return a valid but higher-rank decomposition
        # Actually, let me try a more structured initialization
        
        # Try with some structured initialization
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + attempt)
            U = np.zeros((6, 11))
            V = np.zeros((9, 11))
            W = np.zeros((6, 11))
            
            # Start with some of the "obvious" rank-1 terms
            # Term 0: A[0,0]*B[0,0]*C[0,0]
            U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
            # Term 1: A[0,1]*B[1,0]*C[0,0]
            U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
            # Term 2: A[0,2]*B[2,0]*C[0,0]
            U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
            # Term 3: A[1,0]*B[0,1]*C[1,0]
            U[3, 3] = 1; V[1, 3] = 1; W[2, 3] = 1
            # Term 4: A[1,1]*B[1,1]*C[1,1]
            U[4, 4] = 1; V[4, 4] = 1; W[3, 4] = 1
            # Term 5: A[1,2]*B[2,1]*C[1,1]
            U[5, 5] = 1; V[7, 5] = 1; W[3, 5] = 1
            
            # Add random perturbations to the remaining 5 terms
            for s in range(6, 11):
                U[:, s] = np.random.randn(6) * 0.3
                V[:, s] = np.random.randn(9) * 0.3
                W[:, s] = np.random.randn(6) * 0.3
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            result = minimize(error, params0, jac=gradient, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-10})
            
            if result.fun < best_err:
                best_err = result.fun
                best_result = result.x.copy()
            
            if best_err < 1e-6:
                break
    
    if best_result is not None and best_err < 1e-4:
        U_final = best_result[0:66].reshape(6, 11)
        V_final = best_result[66:165].reshape(9, 11)
        W_final = best_result[165:231].reshape(6, 11)
        
        # Round to nearest half-integer if close
        for mat in [U_final, V_final, W_final]:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    # Try rounding to nearest 0.5
                    val = mat[i, j]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-3:
                        mat[i, j] = rounded
                    else:
                        rounded = round(val)
                        if abs(val - rounded) < 1e-3:
                            mat[i, j] = rounded
        
        return U_final, V_final, W_final
    else:
        # Fallback: return schoolbook rank-18 decomposition (will be used as rank 18, total rank 36)
        # But we need to return something valid. Let me return a rank-18 decomposition.
        U_fb = np.zeros((6, 18))
        V_fb = np.zeros((9, 18))
        W_fb = np.zeros((6, 18))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_fb[i*3+j, col] = 1
                    V_fb[j*3+k, col] = 1
                    W_fb[k*2+i, col] = 1
                    col += 1
        return U_fb, V_fb, W_fb


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    deadline = time.time() + 110  # Safety margin
    
    # Try to get rank-11 decomposition for 2x3x3
    try:
        U3, V3, W3 = _build_rank11_233()
        rank3 = U3.shape[1]
        
        # Verify the decomposition
        T_recon = np.einsum('as,bs,cs->abc', U3, V3, W3)
        T_target = np.zeros((6, 9, 6))
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    T_target[i*3+j, j*3+k, k*2+i] = 1.0
        
        err = np.max(np.abs(T_recon - T_target))
        
        if err < 1e-4 and rank3 == 11:
            # Use two copies for 2x3x6
            # B is 3x6, split into B1 (3x3) and B2 (3x3)
            # C = A * [B1 B2] = [A*B1, A*B2]
            # 
            # For the first 3 columns of B (indices k=0,1,2):
            #   Use U3, V3 (with k in 0..2), W3
            # For the last 3 columns of B (indices k=3,4,5):
            #   Use U3, V3 (with k in 0..2 mapped to 3..5), W3 (with k in 0..2 mapped to 3..5)
            
            rank = 2 * rank3
            U = np.zeros((n * m, rank))  # 6 x 22
            V = np.zeros((m * p, rank))  # 18 x 22
            W = np.zeros((p * n, rank))  # 12 x 22
            
            # First copy: handles B columns 0,1,2 -> C columns 0,1,2
            U[:, 0:rank3] = U3
            # V3 is 9x11, indexed by (j,k) for j in 0..2, k in 0..2
            # For B columns 0,1,2: V[j*6+k, r] = V3[j*3+k, r]
            for j in range(3):
                for k in range(3):
                    V[j*6+k, 0:rank3] = V3[j*3+k, 0:rank3]
            # W3 is 6x11, indexed by (k,i) for k in 0..2, i in 0..1
            # For C columns 0,1,2: W[k*2+i, r] = W3[k*2+i, r]
            for k in range(3):
                for i in range(2):
                    W[k*2+i, 0:rank3] = W3[k*2+i, 0:rank3]
            
            # Second copy: handles B columns 3,4,5 -> C columns 3,4,5
            U[:, rank3:2*rank3] = U3
            # For B columns 3,4,5: V[j*6+(k+3), r] = V3[j*3+k, r]
            for j in range(3):
                for k in range(3):
                    V[j*6+(k+3), rank3:2*rank3] = V3[j*3+k, 0:rank3]
            # For C columns 3,4,5: W[(k+3)*2+i, r] = W3[k*2+i, r]
            for k in range(3):
                for i in range(2):
                    W[(k+3)*2+i, rank3:2*rank3] = W3[k*2+i, 0:rank3]
            
            # Final verification
            T_recon_full = np.einsum('as,bs,cs->abc', U, V, W)
            T_target_full = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target_full[i*m+j, j*p+k, k*n+i] = 1.0
            
            err_full = np.max(np.abs(T_recon_full - T_target_full))
            if err_full < 1e-3:
                # Round to half-integers
                for mat in [U, V, W]:
                    for i in range(mat.shape[0]):
                        for j in range(mat.shape[1]):
                            val = mat[i, j]
                            rounded = round(val * 2) / 2
                            if abs(val - rounded) < 1e-3:
                                mat[i, j] = rounded
                            else:
                                rounded = round(val)
                                if abs(val - rounded) < 1e-3:
                                    mat[i, j] = rounded
                
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If rank-11 didn't work, use whatever rank we got
        if err < 1e-4:
            rank = 2 * rank3
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            U[:, 0:rank3] = U3
            for j in range(3):
                for k in range(3):
                    V[j*6+k, 0:rank3] = V3[j*3+k, 0:rank3]
            for k in range(3):
                for i in range(2):
                    W[k*2+i, 0:rank3] = W3[k*2+i, 0:rank3]
            
            U[:, rank3:2*rank3] = U3
            for j in range(3):
                for k in range(3):
                    V[j*6+(k+3), rank3:2*rank3] = V3[j*3+k, 0:rank3]
            for k in range(3):
                for i in range(2):
                    W[(k+3)*2+i, rank3:2*rank3] = W3[k*2+i, 0:rank3]
            
            T_recon_full = np.einsum('as,bs,cs->abc', U, V, W)
            T_target_full = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target_full[i*m+j, j*p+k, k*n+i] = 1.0
            
            if np.max(np.abs(T_recon_full - T_target_full)) < 1e-3:
                for mat in [U, V, W]:
                    for i in range(mat.shape[0]):
                        for j in range(mat.shape[1]):
                            val = mat[i, j]
                            rounded = round(val * 2) / 2
                            if abs(val - rounded) < 1e-3:
                                mat[i, j] = rounded
                            else:
                                rounded = round(val)
                                if abs(val - rounded) < 1e-3:
                                    mat[i, j] = rounded
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook rank 36
    if time.time() > deadline - 10:
        pass
    
    rank = n * m * p  # 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
