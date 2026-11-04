import time
import numpy as np
from itertools import combinations
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # Build the rank-42 Strassen decomposition
    rank_42 = 42
    U = np.zeros((n * m, rank_42), dtype=float)
    V = np.zeros((m * p, rank_42), dtype=float)
    W = np.zeros((p * n, rank_42), dtype=float)
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    # Verify the decomposition is correct
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Check if the tensor decomposition exactly reproduces matrix multiplication."""
        # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] should equal 
        # 1 if the (i,j) entry of A times (j,k) entry of B contributes to C[i,k]
        # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k), and the decomposition must satisfy:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Actually, the tensor for matrix multiplication M_{n,m,p} has:
        # T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p)
        # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify using random matrices
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomposed = np.zeros((n, p))
        for r in range(min(U_mat.shape[1], V_mat.shape[1], W_mat.shape[1])):
            a_vec = U_mat[:, r]  # length n*m
            b_vec = V_mat[:, r]  # length m*p
            c_vec = W_mat[:, r]  # length p*n
            
            A_r = a_vec.reshape(n, m)
            B_r = b_vec.reshape(m, p)
            C_r = c_vec.reshape(p, n)
            
            # C[i,k] += sum_j A_r[i,j] * B_r[j,k] * ... 
            # Actually: C_decomposed[i,k] = sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
            # Wait, let me think about this more carefully.
            
            # The standard tensor decomposition for matrix multiplication:
            # M_{n,m,p} = sum_r U_r ⊗ V_r ⊗ W_r
            # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
            
            # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U_r[i*m+j] * A[i,j]) * (sum_j V_r[j*p+k] * B[j,k]) * W_r[k*n+i]
            # Hmm, that's not quite right either.
            
            # Let me use the direct formula:
            # The matrix multiplication tensor has entries T[i,j,k] = 1
            # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # So for any A, B:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            #        = sum_j A[i,j] * B[j,k] * 1
            #        = sum_j A[i,j] * B[j,k] * sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            #        = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
            
            # So C_decomposed[i,k] = sum_r W[k*n+i, r] * (A @ U_r_col)[i] * (V_r_col @ B)[k]
            # where U_r_col is the vector U[:,r] reshaped as (n,m), etc.
            
            U_r = U_mat[:, r].reshape(n, m)
            V_r = V_mat[:, r].reshape(m, p)
            W_r = W_mat[:, r].reshape(p, n)
            
            # A @ U_r gives (n, m) @ (n, m)^T? No...
            # sum_j A[i,j] * U_r[i,j] = element-wise product sum
            # Actually: sum_j A[i,j] * U[i*m+j, r] = sum_j A[i,j] * U_r[i,j]
            
            term_A = np.sum(A * U_r, axis=1)  # shape (n,)
            term_B = np.sum(B * V_r, axis=0)  # shape (p,)
            
            # C_decomposed[i,k] += W_r[k,i] * term_A[i] * term_B[k]
            C_decomposed += np.outer(term_A, term_B) * W_r.T
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            return False
        return True
    
    if not verify_decomposition(U, V, W):
        # Fallback: return the schoolbook decomposition
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school), dtype=float)
        V_s = np.zeros((m * p, rank_school), dtype=float)
        W_s = np.zeros((p * n, rank_school), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, idx] = 1
                    V_s[j*p+k, idx] = 1
                    W_s[k*n+i, idx] = 1
                    idx += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Now try to find linear dependencies to reduce rank
    # Check if any column is a linear combination of others
    
    def find_minimal_rank(U_mat, V_mat, W_mat, max_rank=None):
        """Try to find a lower rank decomposition by checking linear dependencies."""
        current_rank = U_mat.shape[1]
        if max_rank is None:
            max_rank = current_rank - 1
        
        # For each target rank, check if we can find a valid decomposition
        # by identifying which columns can be expressed as combinations of others
        
        # Strategy: try removing columns one at a time and checking if the remaining
        # set still spans the same tensor
        
        # First, let's check the rank of the tensor itself
        # The matrix multiplication tensor M_{2,4,6} has a known tensor rank.
        # Lower bounds: 
        # - Trivial: n*m*p = 48 (but we have 42 from Strassen)
        # - Various algebraic geometry bounds
        
        # Let's try to see if rank 40 or 41 is possible by checking if
        # the 42 columns have linear dependencies in the tensor space
        
        # The tensor T can be flattened into a matrix. 
        # If we fix the first index, we get a matrix of size (n*m) x (m*p) for each k,
        # but that's not quite right.
        
        # Better approach: The tensor T[i,j,k] can be seen as a 3D array of shape (n, m, p).
        # A rank-r decomposition means T = sum_{r'} U_{r'} ⊗ V_{r'} ⊗ W_{r'}
        
        # To check if rank < 42 is possible, we need to see if the 42 rank-1 tensors
        # are linearly dependent in the tensor space.
        
        # Flatten the tensor space: dimension is n*m*p = 48
        # Each rank-1 tensor U_r ⊗ V_r ⊗ W_r flattens to a vector of length 48
        # If these 42 vectors are linearly dependent, we might be able to reduce rank
        
        # But linear dependence of the rank-1 tensors doesn't directly give a lower rank
        # decomposition. We need the specific structure.
        
        # Alternative: try to solve for a rank-40 decomposition directly
        # by selecting 40 of the 42 columns and checking if they span the same tensor
        
        # For each pair of columns to remove, check if the remaining 40 can reproduce the tensor
        # This is C(42,2) = 861 checks, each requiring solving a linear system
        
        # Actually, a simpler approach: check if the tensor has rank <= 40 by
        # trying to find 40 rank-1 tensors that sum to T
        
        # Let's use the fact that if the 42 vectors (flattened rank-1 tensors) span
        # a space of dimension < 42, then there's a linear relation, but that doesn't
        # directly help.
        
        # Instead, let's try: for each subset of 40 columns out of 42, check if they
        # can exactly reproduce the tensor. This requires solving:
        # T = sum_{r in subset} c_r * (U_r ⊗ V_r ⊗ W_r)
        # for coefficients c_r
        
        # The flattened equation is: vec(T) = sum_{r in subset} c_r * vec(U_r ⊗ V_r ⊗ W_r)
        # This is a linear system with 48 equations and 40 unknowns (if we fix the
        # structure of U, V, W to be the same as the Strassen decomposition for those columns)
        
        # Wait, but we're allowed to change the values, not just scale. So this approach
        # only works if we keep the same support pattern.
        
        # Let's try a different approach: use the current 42-term decomposition and
        # see if we can express 2 of the terms as linear combinations of the other 40,
        # keeping the U, V, W structure the same.
        
        # For each pair (a, b) to eliminate, solve for coefficients c_0, ..., c_{a-1}, 
        # c_{a+1}, ..., c_{b-1}, c_{b+1}, ..., c_{41} such that:
        # sum_{r != a,b} c_r * (U_r ⊗ V_r ⊗ W_r) = U_a ⊗ V_a ⊗ W_a + U_b ⊗ V_b ⊗ W_b
        
        # This is a linear system: A * c = d
        # where A is 48 x 40 (flattened rank-1 tensors as columns)
        # and d is the flattened sum of the two removed tensors
        
        # If a solution exists, we can absorb the removed terms into the remaining ones
        # by updating the coefficients. But wait, the coefficients are already 1 in the
        # Strassen decomposition. If we find that term_a + term_b = sum_{r != a,b} c_r * term_r,
        # then we can set the coefficients of the remaining terms to c_r and remove a and b.
        # But c_r might not be 1, so the new U, V, W would have different values.
        
        # Actually, in a tensor decomposition, the terms are U_r ⊗ V_r ⊗ W_r, and the
        # decomposition is T = sum_r U_r ⊗ V_r ⊗ W_r. The "coefficients" are absorbed
        # into U, V, W. So if we can write term_a + term_b = sum_{r != a,b} c_r * term_r,
        # we can define new U'_r = c_r * U_r, V'_r = V_r, W'_r = W_r (or distribute c_r
        # among the three factors).
        
        # Let's implement this:
        
        # First, build the matrix of flattened rank-1 tensors
        def flatten_tensor(U_r, V_r, W_r):
            """Flatten U_r ⊗ V_r ⊗ W_r into a vector of length n*m*p."""
            tensor = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
            return tensor.flatten()
        
        # Build the 48 x 42 matrix
        T_matrix = np.zeros((n * m * p, rank_42))
        for r in range(rank_42):
            T_matrix[:, r] = flatten_tensor(U[:, r], V[:, r], W[:, r])
        
        # The target tensor is the matrix multiplication tensor
        T_target = np.ones((n, m, p)).flatten()
        
        # Verify: sum of all 42 terms should equal T_target
        T_sum = T_matrix @ np.ones(rank_42)
        if not np.allclose(T_sum, T_target, atol=1e-10):
            # Something's wrong with our decomposition
            return None
        
        # Now try to eliminate pairs
        best_result = None
        best_rank = rank_42
        
        # For each pair to remove, check if the remaining 40 can span the target
        for remove_a, remove_b in combinations(range(rank_42), 2):
            if time.time() > deadline:
                break
            
            remaining = [r for r in range(rank_42) if r != remove_a and r != remove_b]
            A_sub = T_matrix[:, remaining]  # 48 x 40
            d = T_target  # 48
            
            # Check if d is in the column space of A_sub
            # Use least squares and check residual
            try:
                coeffs, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, d, rcond=None)
                residual = np.linalg.norm(A_sub @ coeffs - d)
                
                if residual < 1e-10:
                    # Found a valid rank-40 decomposition!
                    new_U = U[:, remaining].copy()
                    new_V = V[:, remaining].copy()
                    new_W = W[:, remaining].copy()
                    
                    # Apply the coefficients
                    for idx, r in enumerate(remaining):
                        c = coeffs[idx]
                        # Distribute coefficient: put it all in U
                        new_U[:, idx] *= c
                    
                    # Verify
                    if verify_decomposition(new_U, new_V, new_W):
                        if 40 < best_rank:
                            best_rank = 40
                            best_result = (new_U, new_V, new_W)
            except:
                continue
        
        # Also try single column elimination (rank 41)
        for remove_a in range(rank_42):
            if time.time() > deadline:
                break
            
            remaining = [r for r in range(rank_42) if r != remove_a]
            A_sub = T_matrix[:, remaining]  # 48 x 41
            d = T_target  # 48
            
            try:
                coeffs, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, d, rcond=None)
                residual = np.linalg.norm(A_sub @ coeffs - d)
                
                if residual < 1e-10:
                    new_U = U[:, remaining].copy()
                    new_V = V[:, remaining].copy()
                    new_W = W[:, remaining].copy()
                    
                    for idx, r in enumerate(remaining):
                        c = coeffs[idx]
                        new_U[:, idx] *= c
                    
                    if verify_decomposition(new_U, new_V, new_W):
                        if 41 < best_rank:
                            best_rank = 41
                            best_result = (new_U, new_V, new_W)
            except:
                continue
        
        if best_result is not None:
            return best_result
        return None
    
    # Try to find a lower rank decomposition
    result = find_minimal_rank(U, V, W)
    
    if result is not None:
        U_final, V_final, W_final = result
    else:
        U_final, V_final, W_final = U, V, W
    
    # Final check for deadline
    if time.time() > deadline - 2:
        # Return what we have
        pass
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
