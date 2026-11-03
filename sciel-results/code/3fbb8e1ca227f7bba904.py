"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Starts with schoolbook rank-27 and attempts to reduce rank.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    def idx_u(i, j):
        return i * m + j
        
    def idx_v(j, k):
        return j * p + k
        
    def idx_w(k, i):
        return k * n + i
    
    def build_schoolbook():
        """Build the schoolbook rank-27 decomposition."""
        R = n * m * p  # 27
        U = np.zeros((n * m, R), dtype=np.float64)
        V = np.zeros((m * p, R), dtype=np.float64)
        W = np.zeros((p * n, R), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        
        return U, V, W, R
    
    def verify(U, V, W, R):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_target = np.ones((n, m, p))
        
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    def round_to_half_int(arr):
        """Round array to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def try_remove_term(U, V, W, R, r_remove):
        """Try to remove term r_remove by expressing it as a combination of others."""
        if R <= 1:
            return None
        
        # Try to see if we can set the removed term to zero by adjusting others
        # This is a simplified approach - check if the term is redundant
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        
        # Zero out the term
        U_test[:, r_remove] = 0
        V_test[:, r_remove] = 0
        W_test[:, r_remove] = 0
        
        # Remove the column
        mask = [i for i in range(R) if i != r_remove]
        U_test = U_test[:, mask]
        V_test = V_test[:, mask]
        W_test = W_test[:, mask]
        
        R_new = R - 1
        if verify(U_test, V_test, W_test, R_new):
            return U_test, V_test, W_test, R_new
        
        return None
    
    def local_search(U, V, W, R, max_iterations=200):
        """Local search to try to reduce rank by removing redundant terms."""
        best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
        
        for it in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            found = False
            for r in range(best_R):
                result = try_remove_term(best_U, best_V, best_W, best_R, r)
                if result is not None:
                    best_U, best_V, best_W, best_R = result
                    found = True
                    break
            
            if not found:
                break
        
        return best_U, best_V, best_W, best_R
    
    # Start with schoolbook decomposition
    U, V, W, R = build_schoolbook()
    
    if not verify(U, V, W, R):
        # This should never happen, but handle it gracefully
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Local search to reduce rank
    U, V, W, R = local_search(U, V, W, R)
    
    # Round to half-integers and verify
    U_rounded = round_to_half_int(U)
    V_rounded = round_to_half_int(V)
    W_rounded = round_to_half_int(W)
    
    if verify(U_rounded, V_rounded, W_rounded, R):
        U, V, W = U_rounded, V_rounded, W_rounded
    
    # Final verification - if it fails, use the original unrounded version
    if not verify(U, V, W, R):
        U, V, W, R = build_schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
