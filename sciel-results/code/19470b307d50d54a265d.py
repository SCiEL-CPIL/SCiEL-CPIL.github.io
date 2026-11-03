"""Matrix multiplication 2x3 x 3x6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 36
    
    # Build the tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    diag_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                diag_indices.append((a, b, c))
    
    def verify_full(U, V, W):
        """Full verification of the decomposition."""
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_decomp, T, atol=1e-8):
            return False
        return True
    
    def verify_sparse(U, V, W):
        """Check only the 36 nonzero entries and a subset of zero entries."""
        diag_set = set(diag_indices)
        for (a, b, c) in diag_indices:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-8:
                return False
        # Check all off-diagonal entries
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in diag_set:
                        continue
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return False
        return True
    
    # Create the schoolbook solution as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    def to_lists(U, V, W):
        r = U.shape[1]
        U_list = [[float(U[i,j]) for j in range(r)] for i in range(nm)]
        V_list = [[float(V[i,j]) for j in range(r)] for i in range(mp)]
        W_list = [[float(W[i,j]) for j in range(r)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    best_rank = schoolbook_rank
    best_result = to_lists(U_school, V_school, W_school)
    
    def als_optimize(U, V, W, max_iters=200, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each row a, solve U[a,:] such that
            # sum_c U[a,c] * V[b,c] * W[c,c] = T[a,b,c] for all b,c
            # This is a linear system: U[a,:] @ (V[b,:] * W[c,:])^T = T[a,b,c]
            # For fixed a, we need: sum_c U[a,c] * (V[b,c]*W[c,c]) = T[a,b,c]
            # Let M_a be a matrix where M_a[b,c] = V[b,c] * W[c,c]
            # Then U[a,:] @ M_a^T should equal T[a,:,:] flattened
            # Actually: for each (b,c), sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # So for fixed a: U[a,:] @ (V * W) = T[a,:,:] where (V*W)[b,c] = sum_r V[b,r]*W[c,r]
            
            # Compute V @ W^T: shape (mp, pn) where (V@W^T)[b,c] = sum_r V[b,r]*W[c,r]
            VW = np.einsum('br,cr->bc', V, W)  # shape (mp, pn)
            
            # For each a, solve: U[a,:] @ VW = T[a,:,:]
            # This is: U[a,:] = T[a,:,:] @ pinv(VW)
            # But this is a least squares problem: minimize ||U[a,:] @ VW - T[a,:,:]||^2
            # Solution: U[a,:] = T[a,:,:] @ VW^T @ (VW @ VW^T)^{-1} if invertible
            
            # Use lstsq for numerical stability
            for a in range(nm):
                target = T[a, :, :]  # shape (mp, pn)
                # Solve U[a,:] @ VW = target
                # Equivalent to VW^T @ U[a,:]^T = target^T
                sol, residuals, rank, sv = np.linalg.lstsq(VW.T, target.T, rcond=None)
                U[a, :] = sol
            
            # Update V: for each row b, solve V[b,:] such that
            # sum_c U[a,c] * V[b,c] * W[c,c] = T[a,b,c] for all a,c
            UW = np.einsum('ar,cr->ac', U, W)  # shape (nm, pn)
            
            for b in range(mp):
                target = T[:, b, :]  # shape (nm, pn)
                # Solve V[b,:] @ UW = target
                sol, residuals, rank, sv = np.linalg.lstsq(UW.T, target.T, rcond=None)
                V[b, :] = sol
            
            # Update W: for each row c, solve W[c,:] such that
            # sum_c U[a,c] * V[b,c] * W[c,c] = T[a,b,c] for all a,b
            UV = np.einsum('ar,br->ab', U, V)  # shape (nm, mp)
            
            for c in range(pn):
                target = T[:, :, c]  # shape (nm, mp)
                # Solve W[c,:] @ UV = target
                sol, residuals, rank, sv = np.linalg.lstsq(UV.T, target.T, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_decomp - T
            loss = np.sum(diff**2)
            if loss < tol:
                break
        
        return U, V, W
    
    # Try ranks from 25 to 35
    for rank in range(25, schoolbook_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Multiple random restarts with ALS
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank * 1000 + attempt)
            
            # Initialize with some structure: use small random values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Run ALS
            U, V, W = als_optimize(U, V, W, max_iters=150)
            
            # Check if we found a valid solution
            if verify_full(U, V, W):
                found = True
                best_rank = rank
                best_result = to_lists(U, V, W)
                break
        
        if found:
            break
    
    # If we found something better, try to polish it with more iterations
    if best_rank < schoolbook_rank:
        if time.time() < deadline - 20:
            U_polish = np.array(best_result['U'], dtype=np.float64)
            V_polish = np.array(best_result['V'], dtype=np.float64)
            W_polish = np.array(best_result['W'], dtype=np.float64)
            
            U_polish, V_polish, W_polish = als_optimize(U_polish, V_polish, W_polish, max_iters=300, tol=1e-14)
            
            if verify_full(U_polish, V_polish, W_polish):
                best_result = to_lists(U_polish, V_polish, W_polish)
    
    return best_result
