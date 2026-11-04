import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin

    n, m, p = 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Matricizations
    T_U = T.reshape(nm, mp * pn)
    T_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor T."""
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(T_recon - T))

    def schoolbook():
        """Schoolbook decomposition with rank n*m*p = 120."""
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # New Strategy: Use a different optimization approach.
    # Instead of ALS, we will use a tensor decomposition method that 
    # leverages the structure of matrix multiplication more directly.
    # We'll try to find a low-rank decomposition using a modified version
    # of the "subspace iteration" method or a direct search in the space
    # of possible rank-r decompositions.
    
    # For now, let's implement a more robust ALS with better initialization
    # and a different convergence criterion. We'll also try to use the
    # fact that the tensor has a specific structure to guide the search.
    
    def try_als_decomposition(rank, iterations=100, seed=0):
        """
        Attempt to find a rank-`rank` decomposition using Alternating Least Squares.
        Returns (U, V, W) if successful (low error), else None.
        """
        if time.time() > deadline - 5:
            return None
            
        np.random.seed(seed)
        
        # Better initialization: Use the top singular vectors of the matricizations
        # or start from a slightly perturbed schoolbook-like solution if rank is close.
        # For now, we'll use random initialization but with a different scale.
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        for it in range(iterations):
            if time.time() > deadline - 2:
                return None
            
            # Fix V, W. Solve for U.
            K_VW = np.zeros((rank, mp * pn))
            for t in range(rank):
                K_VW[t, :] = np.outer(V[:, t], W[:, t]).flatten()
            
            try:
                U, _, _, _ = np.linalg.lstsq(K_VW.T, T_U.T, rcond=None)
                U = U.T
            except np.linalg.LinAlgError:
                return None

            # Fix U, W. Solve for V.
            K_UW = np.zeros((rank, nm * pn))
            for t in range(rank):
                K_UW[t, :] = np.outer(U[:, t], W[:, t]).flatten()
            
            try:
                V, _, _, _ = np.linalg.lstsq(K_UW.T, T_V.T, rcond=None)
                V = V.T
            except np.linalg.LinAlgError:
                return None

            # Fix U, V. Solve for W.
            K_UV = np.zeros((rank, nm * mp))
            for t in range(rank):
                K_UV[t, :] = np.outer(U[:, t], V[:, t]).flatten()
            
            try:
                W, _, _, _ = np.linalg.lstsq(K_UV.T, T_W.T, rcond=None)
                W = W.T
            except np.linalg.LinAlgError:
                return None

        # Calculate error
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        
        if err < 1e-6:
            # Try to round to integers or half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            err_rounded = verify_decomposition(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-9:
                return U_rounded, V_rounded, W_rounded
            
            # If half-integer rounding fails, try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            err_int = verify_decomposition(U_int, V_int, W_int)
            if err_int < 1e-9:
                return U_int, V_int, W_int
            
            pass
            
        return None

    # Strategy: Try ranks from 108 down to 90, but with a different approach.
    # We'll use a more aggressive search with more seeds and a different 
    # initialization strategy. We'll also try to use the structure of the 
    # problem to guide the search more effectively.
    
    best_result = None
    best_rank = 120
    U_sb, V_sb, W_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    
    # List of ranks to try, prioritizing known good ones and lower ones
    ranks_to_try = [108, 106, 104, 102, 100, 98, 96, 94, 92, 90]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try multiple seeds for this rank
        for seed in range(10):  # Increase seeds
            if time.time() > deadline - 5:
                break
                
            res = try_als_decomposition(rank, iterations=50, seed=seed)
            if res is not None:
                U, V, W = res
                # Verify one last time
                if verify_decomposition(U, V, W) < 1e-9:
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
                    break # Break seed loop, try next lower rank
        
    # If no better rank was found via ALS, return schoolbook (or the best found)
    U, V, W = best_result
    
    # Final verification
    err = verify_decomposition(U, V, W)
    if err > 1e-9:
        # Fallback to schoolbook if something went wrong
        U, V, W = schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
