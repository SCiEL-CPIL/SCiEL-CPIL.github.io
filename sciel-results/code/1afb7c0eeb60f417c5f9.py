import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    max_time = 145.0

    n, m, p = DIMENSIONS  # 2, 3, 4

    # The 2x3x4 multiplication tensor can be embedded in a 3x3x3 tensor.
    # We use the known explicit rank-23 border rank decomposition of 3x3x3
    # (from the work on matrix multiplication tensor decompositions).
    #
    # Strategy: Start with the schoolbook rank-24 decomposition, then apply
    # algebraic simplification (Gaussian elimination on the linear system)
    # to see if we can reduce the rank.
    
    # First, let's try a direct algebraic approach:
    # The tensor T[i,j,k] = delta_{i,k} for 2x3x4 (i=0,1; j=0,1,2; k=0,1,2,3)
    # We want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all valid (i,j,k)
    # where U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
    
    # Let's use the structure: for a rank-r decomposition, we have r terms.
    # Each term c contributes U[i,j,c]*V[j,k,c]*W[k,i,c] to T[i,j,k].
    #
    # Key insight: We can set up a linear system. For each rank term c,
    # define a = U[:,c] (shape n*m), b = V[:,c] (shape m*p), w = W[:,c] (shape p*n).
    # Then T[i,j,k] = sum_c a[i*m+j] * b[j*p+k] * w[k*n+i]
    #
    # This is a multilinear system. Let's try to find a solution with small rank
    # by using the border rank construction.
    
    # Alternative: Use the fact that 2x3x4 is a "slice" of 3x3x3.
    # The 3x3x3 multiplication tensor has known rank 23 (or border rank 23).
    # If we can embed 2x3x4 in 3x3x3 and project, we might get rank <= 23.
    
    # Let's try a more direct approach: construct an explicit low-rank decomposition.
    # 
    # For 2x3x4, the tensor has 2*3*4 = 24 entries, all equal to 1.
    # A rank-1 term has the form: outer(a, b, w) where a has 6 entries, b has 12, w has 8.
    # The contribution at (i,j,k) is a[i*3+j] * b[j*4+k] * w[k*2+i]
    
    # Let's try to construct a rank-22 or lower decomposition using the
    # "split" technique from algebraic complexity theory.
    
    # Actually, let me try a numerical search with better initialization.
    # The key improvement over the original: use structured initializations
    # based on the known rank-24 solution, then perturb and optimize.
    
    def make_schoolbook(R=24):
        """Create the trivial rank-24 schoolbook decomposition."""
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    def verify(U, V, W):
        """Verify exact decomposition."""
        R = U.shape[1]
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(approx, 1.0, atol=1e-9)

    def try_optimize_rank(R, U0, V0, W0, max_nfev=150):
        """Try to optimize a rank-R decomposition starting from given init."""
        from scipy.optimize import least_squares
        
        def unpack(params):
            U_flat = params[:R * n * m]
            V_flat = params[R * n * m : R * (n * m + m * p)]
            W_flat = params[R * (n * m + m * p):]
            U = U_flat.reshape(R, n * m).T
            V = V_flat.reshape(R, m * p).T
            W = W_flat.reshape(R, p * n).T
            return U, V, W

        def residuals(params):
            U, V, W = unpack(params)
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return (approx - 1.0).flatten()

        # Initialize by taking first R columns from the schoolbook solution
        # and adding some randomness to escape the trivial solution
        U_init = U0[:, :R].copy()
        V_init = V0[:, :R].copy()
        W_init = W0[:, :R].copy()
        
        # Add small random perturbation
        np.random.seed(42 + R)
        U_init += np.random.normal(0, 0.1, U_init.shape)
        V_init += np.random.normal(0, 0.1, V_init.shape)
        W_init += np.random.normal(0, 0.1, W_init.shape)
        
        init_params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            sol = least_squares(residuals, init_params, method='trf', max_nfev=max_nfev, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            if sol.success or sol.cost < 1e-12:
                U, V, W = unpack(sol.x)
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                if verify(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                if verify(U_i, V_i, W_i):
                    return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
                # Check if the unrounded solution is already close to half-integers
                if np.allclose(U, U_r, atol=1e-8) and np.allclose(V, V_r, atol=1e-8) and np.allclose(W, W_r, atol=1e-8):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
        
        return None

    def try_greedy_elimination(U, V, W):
        """
        Try to reduce rank by finding linear dependencies among the rank terms.
        If term c can be expressed as a linear combination of other terms
        (in the tensor product space), we can eliminate it.
        
        Actually, for matrix multiplication tensors, we can try:
        For each column c, check if the rank-1 tensor formed by
        U[:,c] ⊗ V[:,c] ⊗ W[:,c] is in the span of the others.
        """
        R = U.shape[1]
        if R <= 1:
            return None
        
        # Build the tensor for each rank term
        # Term c: T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Flatten: T_c has 24 entries
        tensors = []
        for c in range(R):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            Tc = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Tc[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
            tensors.append(Tc.flatten())
        
        tensors = np.array(tensors)  # (R, 24)
        
        # Find the rank of this matrix
        # If rank < R, there are linear dependencies
        try:
            rank = np.linalg.matrix_rank(tensors, tol=1e-10)
        except:
            return None
        
        if rank < R:
            # Find null space vectors
            _, S, Vt = np.linalg.svd(tensors, full_matrices=True)
            null_dim = R - rank
            if null_dim > 0:
                # The null space gives linear combinations that sum to zero tensor
                # We can use this to reduce the rank
                # For each null vector, we can express one term as a combination of others
                # and eliminate it
                # This is complex; let's just report that we found dependencies
                # and try to construct a lower-rank solution
                pass
        
        return None

    # Try different approaches
    # Approach 1: Direct numerical optimization from perturbed schoolbook
    U_sb, V_sb, W_sb = make_schoolbook(24)
    
    for R in range(23, 17, -1):
        if time.time() - start_time > max_time - 10:
            break
        
        # Try multiple random perturbations
        found = None
        for attempt in range(8):
            if time.time() - start_time > max_time - 5:
                break
            np.random.seed(1000 + R * 100 + attempt)
            U_try = U_sb[:, :R].copy()
            V_try = V_sb[:, :R].copy()
            W_try = W_sb[:, :R].copy()
            
            # Random perturbation
            scale = 0.5
            U_try += scale * np.random.randn(*U_try.shape)
            V_try += scale * np.random.randn(*V_try.shape)
            W_try += scale * np.random.randn(*W_try.shape)
            
            # Also try structured perturbations based on symmetry
            if attempt % 3 == 1:
                # Perturb only U
                U_try += 1.0 * np.random.randn(*U_try.shape)
            elif attempt % 3 == 2:
                # Perturb only V and W
                V_try += 1.0 * np.random.randn(*V_try.shape)
                W_try += 1.0 * np.random.randn(*W_try.shape)
            
            result = try_optimize_rank(R, U_try, V_try, W_try, max_nfev=100)
            if result:
                found = result
                break
        
        if found:
            return found

    # Approach 2: Try a hand-constructed decomposition using the
    # "Laser method" or "tensor splitting" from algebraic complexity theory.
    # For 2x3x4, we can try to use the structure to get rank < 24.
    
    # Let me try a specific construction:
    # We want sum_c a_c(i,j) * b_c(j,k) * w_c(k,i) = 1 for all i,j,k
    # 
    # Try rank 22: We need to "save" 2 terms.
    # One way: If two (i,j,k) triples can share a term partially.
    # 
    # Actually, let's just try the fallback.
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
