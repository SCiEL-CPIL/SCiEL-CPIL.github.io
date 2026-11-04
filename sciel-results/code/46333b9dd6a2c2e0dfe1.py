"""Matrix multiplication 4x5x5 - structured rank reduction search."""
DIMENSIONS = [4, 5, 5]
import time
import numpy as np
from itertools import product, combinations

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    schoolbook_rank = n * m * p  # 100
    
    deadline = time.time() + 140  # Safety margin
    
    # Build schoolbook as baseline using NumPy for speed
    def make_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    
    # Fast verification using NumPy
    def verify(U, V, W, rank):
        """Vectorized verification: check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)"""
        # Build the full tensor product
        # For each r, the contribution to (i,j,k) is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We need this to equal 1 for all (i,j,k)
        
        # Efficient approach: for each r, compute the outer product structure
        # U_r has shape (nm,) = (n*m,)
        # V_r has shape (mp,) = (m*p,)
        # W_r has shape (pn,) = (p*n,)
        
        # The constraint is: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        # We can check this by building the full n*m*p tensor
        result = np.zeros((n, m, p))
        for r in range(rank):
            Ur = U[:, r]  # shape (n*m,)
            Vr = V[:, r]  # shape (m*p,)
            Wr = W[:, r]  # shape (p*n,)
            
            # Reshape to 3D
            Ur_3d = Ur.reshape(n, m)
            Vr_3d = Vr.reshape(m, p)
            Wr_3d = Wr.reshape(p, n)
            
            # The contribution to (i,j,k) is Ur_3d[i,j] * Vr_3d[j,k] * Wr_3d[k,i]
            # This is a rank-1 tensor in (i,j,k) space
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += Ur_3d[i, j] * Vr_3d[j, k] * Wr_3d[k, i]
        
        return np.allclose(result, 1.0, atol=1e-9)
    
    # Even faster verification using einsum
    def verify_fast(U, V, W, rank):
        """Use einsum for fast verification"""
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all (i,j,k)
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is: np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
        
        return np.allclose(result, 1.0, atol=1e-9)
    
    # Verify schoolbook is valid
    assert verify_fast(U_sb, V_sb, W_sb, schoolbook_rank), "Schoolbook verification failed"
    
    import random
    random.seed(42)
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Strategy: Try to find a lower-rank decomposition by eliminating terms
    # and solving for coefficient adjustments
    
    # For each potential term to eliminate, we'll try to find coefficient
    # adjustments that maintain the constraints
    
    for col_to_remove in range(schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        # Determine which (i,j,k) this column corresponds to
        i0 = col_to_remove // (m*p)
        j0 = (col_to_remove // p) % m
        k0 = col_to_remove % p
        
        # Create a copy
        U = U_sb.copy()
        V = V_sb.copy()
        W = W_sb.copy()
        
        # Zero out the column to remove
        U[:, col_to_remove] = 0
        V[:, col_to_remove] = 0
        W[:, col_to_remove] = 0
        
        # Find candidate terms that share at least one index
        candidates = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) == (i0, j0, k0):
                        continue
                    shares = 0
                    if i == i0: shares += 1
                    if j == j0: shares += 1
                    if k == k0: shares += 1
                    if shares > 0:
                        col_idx = i*m*p + j*p + k
                        candidates.append((i, j, k, col_idx, shares))
        
        # Sort by number of shared indices (prefer more shared)
        candidates.sort(key=lambda x: -x[4])
        
        # Try different numbers of candidate terms to adjust
        for num_candidates in [1, 2, 3]:
            if time.time() > deadline - 3:
                break
            if len(candidates) < num_candidates:
                continue
            
            # Try subsets of candidates
            for subset in combinations(candidates[:min(15, len(candidates))], num_candidates):
                if time.time() > deadline - 2:
                    break
                
                # Set up the problem: we want to find coefficients a_c for each candidate c
                # such that the constraint at (i0,j0,k0) is satisfied
                # and all other constraints remain satisfied
                
                # The constraint at (i0,j0,k0) requires:
                # sum_c a_c * U[i0*m+j0, c_col] * V[j0*p+k0, c_col] * W[k0*n+i0, c_col] = 1
                # (since we removed the original term)
                
                # For other (i,j,k), the constraint is:
                # sum_c a_c * U[i*m+j, c_col] * V[j*p+k, c_col] * W[k*n+i, c_col] = 0
                # (since those were already satisfied by the schoolbook)
                
                # This is a linear system in the a_c variables
                
                num_vars = num_candidates
                num_constraints = n * m * p - 1  # all except (i0,j0,k0)
                
                # Build the linear system
                A = np.zeros((num_constraints, num_vars))
                b = np.zeros(num_constraints)
                
                constraint_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i, j, k) == (i0, j0, k0):
                                continue
                            for var_idx, (ci, cj, ck, ccol, _) in enumerate(subset):
                                A[constraint_idx, var_idx] = (U[i*m+j, ccol] * V[j*p+k, ccol] * W[k*n+i, ccol])
                            constraint_idx += 1
                
                # The constraint at (i0,j0,k0):
                # sum_c a_c * U[i0*m+j0, ccol] * V[j0*p+k0, ccol] * W[k0*n+i0, ccol] = 1
                target = 0
                for var_idx, (ci, cj, ck, ccol, _) in enumerate(subset):
                    target += U[i0*m+j0, ccol] * V[j0*p+k0, ccol] * W[k0*n+i0, ccol]
                
                # Add this as an additional constraint (or modify b)
                # Actually, we need the sum to equal 1, so:
                # sum_c a_c * (U[i0*m+j0, ccol] * V[j0*p+k0, ccol] * W[k0*n+i0, ccol]) = 1
                
                # Let's add this as the last row of A and set b appropriately
                # But we already have num_constraints rows. Let's add one more.
                A_ext = np.vstack([A, np.array([U[i0*m+j0, ccol] * V[j0*p+k0, ccol] * W[k0*n+i0, ccol] for (_, _, _, ccol, _) in subset])])
                b_ext = np.concatenate([b, [1.0]])
                
                # Solve the linear system
                try:
                    solution = np.linalg.lstsq(A_ext, b_ext, rcond=None)[0]
                except:
                    continue
                
                # Check if the solution is valid (integer/half-integer and small)
                valid_solution = True
                for val in solution:
                    if abs(val - round(val*2)/2) > 1e-6:
                        valid_solution = False
                        break
                    if abs(val) > 10:  # Check for reasonable magnitude
                        valid_solution = False
                        break
                
                if not valid_solution:
                    continue
                
                # Apply the solution
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                
                for var_idx, (ci, cj, ck, ccol, _) in enumerate(subset):
                    a = solution[var_idx]
                    # The original coefficient was 1, so we add a to it
                    # Actually, we need to be careful: the schoolbook had coefficient 1
                    # We're modifying the coefficient from 1 to 1 + a
                    U_new[ci*m+cj, ccol] = 1 + a
                    V_new[cj*p+ck, ccol] = 1 + a
                    W_new[ck*n+ci, ccol] = 1 + a
                
                # Verify
                if verify_fast(U_new, V_new, W_new, schoolbook_rank):
                    # Remove the zero column
                    mask = np.ones(schoolbook_rank, dtype=bool)
                    mask[col_to_remove] = False
                    U_final = U_new[:, mask]
                    V_final = V_new[:, mask]
                    W_final = W_new[:, mask]
                    
                    if verify_fast(U_final, V_final, W_final, schoolbook_rank - 1):
                        best_rank = schoolbook_rank - 1
                        best_U, best_V, best_W = U_final, V_final, W_final
                        break
            
            if best_rank < schoolbook_rank:
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # If no improvement found, return schoolbook
    if best_rank == schoolbook_rank:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
