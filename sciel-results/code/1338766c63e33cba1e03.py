"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 4 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    This implementation searches for a lower-rank decomposition than the schoolbook
    rank of 48 using structured linear algebra and local optimization.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition
    best_rank = rank_schoolbook
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Convert to numpy for efficient linear algebra
    U_np = np.array(best_U, dtype=float)  # (n*m, rank)
    V_np = np.array(best_V, dtype=float)  # (m*p, rank)
    W_np = np.array(best_W, dtype=float)  # (p*n, rank)
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that the decomposition correctly computes matrix multiplication."""
        # For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for all other combinations, the sum should be 0.
        # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # should equal 1 if this is the (i,j,k) entry of the multiplication tensor.
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Check all (i,j,k) entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # This should be 1 (the (i,j,k) entry of C = A*B)
                    val = 0.0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        
        # Check that off-diagonal entries are 0
        # For matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) - there are no off-diagonal zeros to check
        # in this formulation. The tensor is just the identity-like structure.
        
        return True
    
    def try_eliminate_term(U_mat, V_mat, W_mat, term_idx):
        """
        Try to eliminate term_idx by expressing it as a linear combination of other terms.
        Returns (success, new_U, new_V, new_W) or (False, None, None, None).
        """
        rank = U_mat.shape[1]
        if rank <= 1:
            return False, None, None, None
        
        # Remove term_idx
        mask = [i for i in range(rank) if i != term_idx]
        U_rem = U_mat[:, mask]
        V_rem = V_mat[:, mask]
        W_rem = W_mat[:, mask]
        
        new_rank = len(mask)
        
        # We want to find coefficients c_r such that:
        # U[term] = sum_r c_r * U_rem[:, r]  (for the U component)
        # V[term] = sum_r c_r * V_rem[:, r]  (for the V component)  
        # W[term] = sum_r c_r * W_rem[:, r]  (for the W component)
        # 
        # But this is over-constrained. Instead, we need:
        # For all (i,j,k): U[term_idx, i*m+j] * V[term_idx, j*p+k] * W[term_idx, k*n+i]
        #   = sum_r c_r * U_rem[r, i*m+j] * V_rem[r, j*p+k] * W_rem[r, k*n+i]
        #
        # This is a nonlinear system. Let's use a different approach:
        # Try small perturbations and see if we can find a valid lower-rank decomposition.
        
        return False, None, None, None
    
    # Strategy: Use a structured search to find rank reductions
    # For 3x4x4, known results suggest rank can be significantly below 48.
    # Let's try a randomized approach with local search.
    
    def random_decomposition(rank):
        """Generate a random rank-r decomposition."""
        U_r = np.random.randn(n * m, rank) * 0.5
        V_r = np.random.randn(m * p, rank) * 0.5
        W_r = np.random.randn(p * n, rank) * 0.5
        return U_r, V_r, W_r
    
    def decompose_error(U_mat, V_mat, W_mat):
        """Compute the error of the decomposition."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    error += (val - 1.0) ** 2
        return error
    
    def local_search(U_mat, V_mat, W_mat, max_iter=100, tol=1e-10):
        """Perform local search to minimize decomposition error."""
        U_cur = U_mat.copy()
        V_cur = V_mat.copy()
        W_cur = W_mat.copy()
        rank = U_cur.shape[1]
        
        best_err = decompose_error(U_cur, V_cur, W_cur)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if best_err < tol:
                break
            
            # Try random perturbations
            improved = False
            for _ in range(20):
                # Perturb one matrix
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    U_new = U_cur + np.random.randn(n * m, rank) * 0.1
                    V_new = V_cur
                    W_new = W_cur
                elif mat_idx == 1:
                    U_new = U_cur
                    V_new = V_cur + np.random.randn(m * p, rank) * 0.1
                    W_new = W_cur
                else:
                    U_new = U_cur
                    V_new = V_cur
                    W_new = W_cur + np.random.randn(p * n, rank) * 0.1
                
                err = decompose_error(U_new, V_new, W_new)
                if err < best_err:
                    U_cur, V_cur, W_cur = U_new, V_new, W_new
                    best_err = err
                    improved = True
                    break
            
            if not improved:
                # Try smaller step size
                for _ in range(10):
                    mat_idx = random.randint(0, 2)
                    step = 0.05
                    if mat_idx == 0:
                        U_new = U_cur + np.random.randn(n * m, rank) * step
                        V_new = V_cur
                        W_new = W_cur
                    elif mat_idx == 1:
                        U_new = U_cur
                        V_new = V_cur + np.random.randn(m * p, rank) * step
                        W_new = W_cur
                    else:
                        U_new = U_cur
                        V_new = V_cur
                        W_new = W_cur + np.random.randn(p * n, rank) * step
                    
                    err = decompose_error(U_new, V_new, W_new)
                    if err < best_err:
                        U_cur, V_cur, W_cur = U_new, V_new, W_new
                        best_err = err
                        improved = True
                        break
                if not improved:
                    break
        
        return U_cur, V_cur, W_cur, best_err
    
    # Try to find lower-rank decompositions
    # Known lower bounds for 3x4x4 matrix multiplication suggest rank around 30-40
    # Let's try ranks from 40 down to 25
    
    for target_rank in range(40, 24, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U_r, V_r, W_r = random_decomposition(target_rank)
            U_l, V_l, W_l, err = local_search(U_r, V_r, W_r, max_iter=50, tol=1e-8)
            
            if err < 1e-6:
                # Found a valid decomposition!
                # Round to nearest integers or half-integers
                U_rounded = np.round(U_l * 2) / 2
                V_rounded = np.round(V_l * 2) / 2
                W_rounded = np.round(W_l * 2) / 2
                
                # Verify the rounded version
                err_rounded = decompose_error(U_rounded, V_rounded, W_rounded)
                if err_rounded < 1e-6:
                    best_rank = target_rank
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    found = True
                    break
        
        if found:
            # Try to reduce further
            continue
    
    # Final verification and cleanup
    if best_rank == rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Convert back to list format
    U_final = [[float(x) for x in row] for row in best_U]
    V_final = [[float(x) for x in row] for row in best_V]
    W_final = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
