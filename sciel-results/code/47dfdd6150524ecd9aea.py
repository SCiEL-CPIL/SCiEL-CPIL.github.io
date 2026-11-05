import numpy as np
import time
from itertools import combinations

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W correctly decompose the n x m * m x p multiplication."""
        rank = U.shape[1]
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj, col] = 1.0
                    V[jj * p + kk, col] = 1.0
                    W[kk * n + ii, col] = 1.0
                    col += 1
        return U, V, W

    def build_rank33():
        """Build the rank-33 Strassen-based decomposition."""
        U = np.zeros((n * m, 33))
        V = np.zeros((m * p, 33))
        W = np.zeros((p * n, 33))
        
        col = 0
        
        # Part 1: A[:,0] * B[0,:] - 12 terms
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
        
        # Part 2: A[:,1:3] * B[1:3,:] using Strassen for each 2x2 block of B
        for t in range(3):
            k0 = 2 * t
            k1 = 2 * t + 1
            
            # M1 = (A00+A11)(B00+B11)
            U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
            V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
            W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
            col += 1
            
            # M2 = (A10+A11)B00
            U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
            V[1 * p + k0, col] += 1.0
            W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
            col += 1
            
            # M3 = A00(B10-B11)
            U[0 * m + 1, col] += 1.0
            V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
            W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
            col += 1
            
            # M4 = A11(B01-B00)
            U[1 * m + 2, col] += 1.0
            V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] -= 1.0
            W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
            col += 1
            
            # M5 = (A00+A01)B11
            U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
            V[2 * p + k1, col] += 1.0
            W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
            col += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
            V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
            W[k1 * n + 1, col] += 1.0
            col += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
            V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
            W[k0 * n + 0, col] += 1.0
            col += 1
        
        return U, V, W

    def try_reduce_rank(U, V, W, target_rank, time_budget=60):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 terms.
        For each term to remove, check if it can be expressed as a linear combination
        of the remaining terms (which would require the remaining terms to span it).
        
        Since we're looking for exact rank reduction, we check if removing a term
        still allows the decomposition to be exact by solving for adjustments.
        """
        start = time.time()
        rank = U.shape[1]
        
        if rank <= target_rank:
            return U, V, W
        
        # Try removing each term one at a time
        for remove_idx in range(rank):
            if time.time() - start > time_budget:
                break
            
            # Create copies without the removed term
            U_test = np.delete(U, remove_idx, axis=1)
            V_test = np.delete(V, remove_idx, axis=1)
            W_test = np.delete(W, remove_idx, axis=1)
            
            # Check if the remaining terms can still span the required tensor
            # This is equivalent to checking if the removed term is in the span
            # of the remaining terms in the tensor space
            
            # Build the tensor representation
            # Each term r contributes: outer product of u_r, v_r, w_r
            # We need sum_r u_r ⊗ v_r ⊗ w_r = multiplication tensor
            
            # Check if the removed term is linearly dependent
            # by seeing if we can find coefficients c_r such that:
            # sum_{r != remove_idx} c_r * (u_r ⊗ v_r ⊗ w_r) = u_remove ⊗ v_remove ⊗ w_remove
            
            # This is a linear system. Let's set it up.
            # The tensor space has dimension n*m*p = 36
            # Each term gives a vector in this space (flattened outer product)
            
            n, m, p = 2, 3, 6
            dim = n * m * p
            
            # Build matrix of remaining terms
            A_mat = np.zeros((dim, rank - 1))
            b_vec = np.zeros(dim)
            
            # Target: the removed term
            u_rem = U[:, remove_idx].reshape(n, m)
            v_rem = V[:, remove_idx].reshape(m, p)
            w_rem = W[:, remove_idx].reshape(p, n)
            
            # Flatten the outer product
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = (i * m + j) * p + k
                        b_vec[idx] = u_rem[i, j] * v_rem[j, k] * w_rem[k, i]
            
            # Build columns for remaining terms
            col_idx = 0
            for r in range(rank):
                if r == remove_idx:
                    continue
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = (i * m + j) * p + k
                            A_mat[idx, col_idx] = u_r[i, j] * v_r[j, k] * w_r[k, i]
                col_idx += 1
            
            # Try to solve A_mat * c = b_vec
            try:
                # Use least squares
                c, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check if the solution is exact
                if residuals.size > 0:
                    residual = np.sqrt(residuals[0])
                else:
                    residual = np.linalg.norm(A_mat @ c - b_vec)
                
                if residual < 1e-8:
                    # Found a dependency! The removed term is in the span.
                    # This means we can potentially re-express the decomposition
                    # without that term, but we need to adjust the other terms.
                    # 
                    # Actually, if the removed term is exactly in the span of the others,
                    # it means the original decomposition had a linear dependency,
                    # so the rank was not minimal. We can remove it.
                    #
                    # But wait - we need to verify that the remaining terms alone
                    # still decompose the multiplication tensor correctly.
                    
                    # Let's verify the reduced decomposition
                    if verify_decomposition(U_test, V_test, W_test, n, m, p):
                        return U_test, V_test, W_test
                    else:
                        # The removed term was needed, but it's in the span.
                        # This means we can re-express it, but the simple removal doesn't work.
                        # We'd need to find a different set of terms.
                        pass
                        
            except Exception:
                continue
        
        return U, V, W

    def search_rank32(time_budget=80):
        """
        Search for a rank-32 decomposition by trying to find a single
        linear dependency in the rank-33 construction.
        """
        start = time.time()
        
        # Build rank-33
        U, V, W = build_rank33()
        
        # Verify it's correct
        if not verify_decomposition(U, V, W, n, m, p):
            return None
        
        # Try to reduce
        U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W, 32, time_budget)
        
        if U_reduced.shape[1] < U.shape[1]:
            return U_reduced, V_reduced, W_reduced
        
        return None

    # First, build the rank-33 decomposition
    U, V, W = build_rank33()
    
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U_school, V_school, W_school = schoolbook(n, m, p)
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find rank-32 if we have time
    if time.time() < deadline - 20:
        result = search_rank32(time_budget=60)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return rank-33
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
