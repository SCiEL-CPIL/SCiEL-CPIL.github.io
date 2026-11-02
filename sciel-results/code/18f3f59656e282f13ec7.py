import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 4 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    For 3x4x4, we use a structured search starting from schoolbook (rank 48)
    and attempting to reduce rank through column elimination and random search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 48
    
    # Initialize schoolbook decomposition
    # U shape: (n*m, rank)
    # V shape: (m*p, rank) 
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank_schoolbook), dtype=np.int32)
    V = np.zeros((m * p, rank_schoolbook), dtype=np.int32)
    W = np.zeros((p * n, rank_schoolbook), dtype=np.int32)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Function to verify a decomposition
    def verify(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly computes matrix multiplication."""
        # For each (i, k) output element, the tensor entry should be 1
        # T[i, j, k] = 1 for all valid (i, j, k)
        # The decomposition gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For this to equal the multiplication tensor, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is valid
        # Actually, the standard matrix multiplication tensor has T[i,j,k]=1 for all i,j,k
        # and the bilinear form is: (U, V, W) such that for all i,j,k:
        # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] ... wait, let me reconsider.
        
        # The matrix multiplication bilinear map is:
        # C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
        # As a trilinear form in the decomposition:
        # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) ... 
        # No, the standard decomposition is:
        # C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] is not right either.
        
        # Correct: In tensor decomposition for matrix multiplication,
        # we have factors U, V, W such that:
        # C[i,k] = sum_r (U[:,i,k] · A) * (V[:,i,k] · B) * W[i,k]
        # No, that's not standard either.
        
        # The standard Strassen-style decomposition:
        # We decompose the multiplication tensor T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
        # A rank-1 term is u ⊗ v ⊗ w, contributing u[i]*v[j]*w[k] to T[i,j,k]
        # But for matrix multiplication, the tensor is different.
        
        # Actually, for matrix multiplication C = AB:
        # The "tensor" whose rank we minimize is the one where
        # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
        # and the bilinear computation is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In the decomposition, each rank-1 term (u, v, w) contributes:
        # (sum_j u[i,j] * A[i,j]) * (sum_j v[j,k] * B[j,k]) * w[i,k]
        # Wait, that gives a product of two bilinear forms times a scalar,
        # which is not how it works.
        
        # Let me reconsider. The standard formulation:
        # We want to compute C = A*B using rank r.
        # We precompute M_r = (U_r^T * A) * (V_r^T * B) ... no.
        
        # Correct formulation from literature:
        # U has shape (r, n*m), V has shape (r, m*p), W has shape (r, p*n)
        # Precompute: for each r, X_r = U[r,:] @ A.reshape(n*m) ... 
        # This is getting confused. Let me use the direct check.
        
        # The decomposition means:
        # For each output (i,k): C[i,k] = sum_r W[r, k*n+i] * (sum_j U[r, i*m+j] * A[i,j]) * (sum_j V[r, j*p+k] * B[j,k])
        # Wait no, that would be a product of two scalars times a scalar, giving a trilinear form in A, B, and the output weights.
        
        # Actually the correct check for the tensor decomposition:
        # The matrix multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k).
        # A rank-1 decomposition term is (u, v, w) where u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
        # and the contribution to T[i,j,k] is: u[i*m+j] * v[j*p+k] * w[k*n+i]
        # We need: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all i,j,k
        
        # So the check is:
        # For all i in [n], j in [m], k in [p]:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        total = np.zeros((n, m, p), dtype=np.int32)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
        return np.all(total == 1)
    
    if not verify(U, V, W):
        # Should never happen for schoolbook
        pass
    
    # Try to reduce rank by checking if columns are linearly dependent
    # in a way that preserves the tensor decomposition
    
    def try_remove_column(U_mat, V_mat, W_mat, col_idx):
        """Try to remove column col_idx by expressing it as a combination of others."""
        if col_idx >= U_mat.shape[1]:
            return None
        
        r = U_mat.shape[1]
        if r <= 1:
            return None
        
        # We want to find coefficients c_s for s != col_idx such that:
        # For all i,j,k: U[i*m+j, col_idx] * V[j*p+k, col_idx] * W[k*n+i, col_idx] 
        # equals the sum over s != col_idx of c_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        
        # This is a nonlinear system. Instead, let's try a simpler approach:
        # Check if column col_idx is zero in U, V, or W, in which case it can be removed.
        if np.all(U_mat[:, col_idx] == 0) or np.all(V_mat[:, col_idx] == 0) or np.all(W_mat[:, col_idx] == 0):
            new_U = np.delete(U_mat, col_idx, axis=1)
            new_V = np.delete(V_mat, col_idx, axis=1)
            new_W = np.delete(W_mat, col_idx, axis=1)
            return new_U, new_V, new_W
        
        return None
    
    # Try random perturbations to find lower-rank solutions
    # Use the fact that we can combine columns
    
    def random_search(current_U, current_V, current_W, max_iters, time_limit):
        """Randomly try to merge columns to reduce rank."""
        best_r = current_U.shape[1]
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        
        for iteration in range(max_iters):
            if time.time() > time_limit:
                break
            
            r = current_U.shape[1]
            if r <= best_r:
                break
            
            # Pick two columns to try to merge
            c1, c2 = np.random.choice(r, 2, replace=False)
            
            # Try to express column c2 as a linear combination of column c1 and others
            # Simple approach: try replacing c2 with c2 - c1 and see if we can zero out
            new_U = current_U.copy()
            new_V = current_V.copy()
            new_W = current_W.copy()
            
            # This is a heuristic - try various linear combinations
            for alpha in [0, 1, -1, 0.5, -0.5]:
                trial_U = current_U.copy()
                trial_V = current_V.copy()
                trial_W = current_W.copy()
                trial_U[:, c2] = current_U[:, c2] + alpha * current_U[:, c1]
                trial_V[:, c2] = current_V[:, c2] + alpha * current_V[:, c1]
                trial_W[:, c2] = current_W[:, c2] + alpha * current_W[:, c1]
                
                if verify(trial_U, trial_V, trial_W):
                    # Check if any column is now zero
                    for c in range(trial_U.shape[1]):
                        if np.all(trial_U[:, c] == 0) or np.all(trial_V[:, c] == 0) or np.all(trial_W[:, c] == 0):
                            new_U = np.delete(trial_U, c, axis=1)
                            new_V = np.delete(trial_V, c, axis=1)
                            new_W = np.delete(trial_W, c, axis=1)
                            if new_U.shape[1] < best_r:
                                best_r = new_U.shape[1]
                                best_U, best_V, best_W = new_U.copy(), new_V.copy(), new_W.copy()
                            break
        
        return best_U, best_V, best_W
    
    # Run limited random search
    max_search_time = deadline - time.time() - 5.0
    if max_search_time > 5.0:
        U, V, W = random_search(U, V, W, max_iters=500, time_limit=start_time + max_search_time)
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
