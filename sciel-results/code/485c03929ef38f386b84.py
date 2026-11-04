"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Schoolbook baseline
    rank_school = n * m * p  # 96
    U_school = np.zeros((nm, rank_school), dtype=np.int8)
    V_school = np.zeros((mp, rank_school), dtype=np.int8)
    W_school = np.zeros((pn, rank_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W decompose the tensor T."""
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Using einsum for efficiency
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_recon, T, atol=tol, rtol=0)
    
    def try_rank_reduction(U, V, W, current_rank, max_reductions=5):
        """
        Try to reduce rank by eliminating one term at a time.
        For each term c, check if we can adjust the remaining terms
        to still satisfy the tensor equation.
        
        Strategy: For a given term to remove, we look for linear combinations
        of other terms that can replace it. This is done by checking if the
        "contribution vector" of term c is in the span of other terms' contributions.
        """
        U = U.astype(np.float64).copy()
        V = V.astype(np.float64).copy()
        W = W.astype(np.float64).copy()
        current_rank = U.shape[1]
        
        for _ in range(max_reductions):
            if time.time() > deadline - 5.0:
                break
            
            found_removal = False
            # Try each column as a candidate for removal
            for c_remove in range(current_rank):
                if time.time() > deadline - 5.0:
                    break
                
                # The contribution of term c_remove is:
                # u_c ⊗ v_c ⊗ w_c (outer product)
                u_c = U[:, c_remove]
                v_c = V[:, c_remove]
                w_c = W[:, c_remove]
                
                # Check if this term is "redundant" by seeing if we can express
                # its contribution as a combination of other terms.
                # 
                # A simpler heuristic: if u_c, v_c, or w_c is zero, we can remove it.
                if np.allclose(u_c, 0) or np.allclose(v_c, 0) or np.allclose(w_c, 0):
                    # Remove this term
                    mask = np.ones(current_rank, dtype=bool)
                    mask[c_remove] = False
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    current_rank -= 1
                    found_removal = True
                    break
                
                # More sophisticated check: try to see if we can adjust nearby terms
                # to absorb this term's contribution.
                # This is expensive, so we only try for a few candidates.
                # 
                # For now, we'll use a simpler approach: try random perturbations
                # and check if the tensor is still satisfied.
                
                # Skip for now to save time
                continue
            
            if not found_removal:
                break
    
        return U, V, W, current_rank
    
    # Attempt to find a lower rank decomposition using a structured approach
    # based on the "sliding window" / convolutional idea.
    
    # The key insight: for matrix multiplication, the tensor has a specific structure.
    # We can try to find algebraic identities that reduce the rank.
    #
    # One approach: use the fact that for small matrices, there may be non-trivial
    # decompositions. For example, Strassen's algorithm reduces 2x2x2 from 8 to 7.
    #
    # For 3x4x8, we can try to apply block-wise Strassen-like decompositions.
    #
    # Let's try a different approach: use numerical linear algebra to find
    # a lower-rank approximation and then check if it's exact.
    
    # Approach: Start with schoolbook and try to find linear dependencies.
    # We can do this by looking at the "flattened" tensor and computing its rank
    # in different modes.
    
    # Mode-1 flattening: T as a matrix of shape (nm, mp*pn)
    T_mat1 = T.reshape(nm, mp * pn)
    rank_mode1 = np.linalg.matrix_rank(T_mat1)
    
    # Mode-2 flattening: T as a matrix of shape (mp, nm*pn)
    T_mat2 = T.reshape(mp, nm * pn)
    rank_mode2 = np.linalg.matrix_rank(T_mat2)
    
    # Mode-3 flattening: T as a matrix of shape (pn, nm*mp)
    T_mat3 = T.reshape(pn, nm * mp)
    rank_mode3 = np.linalg.matrix_rank(T_mat3)
    
    # The tensor rank is at least the maximum of these mode ranks.
    lower_bound = max(rank_mode1, rank_mode2, rank_mode3)
    
    # If lower_bound < best_rank, there might be a lower rank decomposition.
    # However, finding it is NP-hard, so we use heuristics.
    
    # Heuristic: Try to find a decomposition of rank = lower_bound + small_offset
    # by using alternating minimization.
    
    if lower_bound < best_rank:
        # Try alternating minimization starting from a random initialization
        # with rank = max(lower_bound, best_rank - 10)
        target_rank = max(lower_bound, best_rank - 10)
        
        if target_rank < best_rank:
            # Initialize with random vectors
            np.random.seed(42)
            U_rand = np.random.randn(nm, target_rank)
            V_rand = np.random.randn(mp, target_rank)
            W_rand = np.random.randn(pn, target_rank)
            
            # Normalize to avoid numerical issues
            for c in range(target_rank):
                norm = np.linalg.norm(U_rand[:, c])
                if norm > 0:
                    U_rand[:, c] /= norm
                norm = np.linalg.norm(V_rand[:, c])
                if norm > 0:
                    V_rand[:, c] /= norm
                norm = np.linalg.norm(W_rand[:, c])
                if norm > 0:
                    W_rand[:, c] /= norm
            
            # Alternating minimization
            for iteration in range(50):
                if time.time() > deadline - 10.0:
                    break
                
                # Update U: for each row a, solve for U[a, :]
                # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
                # For fixed a: T[a, :, :] = U[a, :] @ (V * W)  (element-wise product)
                # This is a linear system: T[a, b, c] = sum_r U[a, r] * (V[b, r] * W[c, r])
                
                for a in range(nm):
                    # Build the matrix M where M[b*pn + c, r] = V[b, r] * W[c, r]
                    M = np.zeros((mp * pn, target_rank))
                    for b in range(mp):
                        for c in range(pn):
                            M[b * pn + c, :] = V[b, :] * W[c, :]
                    
                    # Solve U[a, :] @ M^T = T[a, :].reshape(-1)
                    rhs = T[a, :, :].reshape(-1)
                    # Use least squares
                    try:
                        U[a, :], residuals, rank_M, sv = np.linalg.lstsq(M.T, rhs, rcond=None)
                    except:
                        pass
                
                # Update V: for each row b
                for b in range(mp):
                    M = np.zeros((nm * pn, target_rank))
                    for a in range(nm):
                        for c in range(pn):
                            M[a * pn + c, :] = U[a, :] * W[c, :]
                    
                    rhs = T[:, b, :].reshape(-1)
                    try:
                        V[b, :], residuals, rank_M, sv = np.linalg.lstsq(M.T, rhs, rcond=None)
                    except:
                        pass
                
                # Update W: for each row c
                for c in range(pn):
                    M = np.zeros((nm * mp, target_rank))
                    for a in range(nm):
                        for b in range(mp):
                            M[a * mp + b, :] = U[a, :] * V[b, :]
                    
                    rhs = T[:, :, c].reshape(-1)
                    try:
                        W[c, :], residuals, rank_M, sv = np.linalg.lstsq(M.T, rhs, rcond=None)
                    except:
                        pass
                
                # Check if we've converged to a valid solution
                if verify(U_rand, V_rand, W_rand, tol=1e-6):
                    # We found a valid decomposition!
                    # Try to clean up small values
                    U_rand[np.abs(U_rand) < 1e-8] = 0
                    V_rand[np.abs(V_rand) < 1e-8] = 0
                    W_rand[np.abs(W_rand) < 1e-8] = 0
                    
                    # Verify again with tighter tolerance
                    if verify(U_rand, V_rand, W_rand, tol=1e-8):
                        best_U, best_V, best_W = U_rand, V_rand, W_rand
                        best_rank = target_rank
                        break
    
    # Final verification
    if not verify(best_U, best_W, best_V, tol=1e-8):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_school
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
