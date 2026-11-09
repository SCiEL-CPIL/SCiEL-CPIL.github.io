import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110

    def verify(U, V, W):
        """Verify tensor decomposition exactly using vectorized ops."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)  # (n*m, rank)
        V_arr = np.array(V, dtype=float)  # (m*p, rank)
        W_arr = np.array(W, dtype=float)  # (p*n, rank)
        
        # Reshape: U_r is (n, m, rank), V_r is (m, p, rank), W_r is (p, n, rank)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # For each (i,j,k): dot product of U3[i,j,:], V3[j,k,:], W3[k,i,:]
        # Vectorize: for fixed (i,j), sum over r of U3[i,j,r]*V3[j,k,r] gives (p,) array
        # Then multiply by W3[k,i,r] and sum over r
        
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # = sum_r (U3[i,j,r] * V3[j,k,r]) * W3[k,i,r]
        
        # Build T efficiently:
        # For each r: outer product structure
        # T[i*m+j, j*p+k, k*n+i] += U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
        
        # Use einsum: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # But W3[k,i,r] has indices (k,i,r) and we need (i,j,k)
        # Let's use explicit indexing with broadcasting
        
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # = sum_r U3[i,j,r] * (V3[j,k,r] * W3[k,i,r])
        
        # Compute for all (i,j,k):
        # temp[j,k,i] = sum_r V3[j,k,r] * W3[k,i,r]  -> shape (m, p, n)
        temp = np.einsum('jkr,kir->jki', V3, W3)  # (m, p, n)
        # T[i,j,k] = sum_r U3[i,j,r] * temp[j,k,i]
        # = sum_r U3[i,j,r] * temp[j,k,i]
        # temp[j,k,i] indexed as [j,k,i], U3[i,j,r] indexed as [i,j,r]
        # T[i,j,k] = sum_r U3[i,j,r] * temp[j,k,i]
        T = np.einsum('ijr,jki->ijk', U3, temp)
        
        # Check against expected: T[i,j,k] should be 1 for all i,j,k
        return np.allclose(T, 1.0, atol=1e-9)

    def build_strassen_36():
        """
        Build rank-36 decomposition:
        - 4 Strassen 2x2x2 blocks for B columns 0-3 (4*7 = 28 terms)
        - 8 schoolbook terms for B column 4 (2*4*1 = 8 terms)
        Total: 36 terms
        """
        # Strassen 2x2x2 factor matrices (rank 7)
        strassen_U = np.array([
            [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
            [1,1,0,0],[-1,0,1,0],[0,1,0,-1],
        ], dtype=float)
        strassen_V = np.array([
            [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
            [0,0,0,1],[1,1,0,0],[0,0,1,1],
        ], dtype=float)
        strassen_W = np.array([
            [1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],
            [-1,0,1,0],[0,0,0,1],[1,0,0,0],
        ], dtype=float)
        
        rank = 4*7 + 8  # 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        # Four Strassen blocks: j-blocks (0,2) x k-blocks (0,2)
        for j0 in (0, 2):
            for k0 in (0, 2):
                for r in range(7):
                    # Fill U: 2x2 block at rows 0,1 and cols j0,j0+1
                    U[0:m, col] = strassen_U[r, :].reshape(2, 2).T.ravel()  # careful with indexing
                    # Actually let's do it explicitly
                    for li in range(2):
                        for lj in range(2):
                            U[li*m + (j0+lj), col] = strassen_U[r, li*2+lj]
                    # Fill V: 2x2 block at rows j0,j0+1 and cols k0,k0+1
                    for lj in range(2):
                        for lk in range(2):
                            V[(j0+lj)*p + (k0+lk), col] = strassen_V[r, lj*2+lk]
                    # Fill W: 2x2 block at rows k0,k0+1 and cols 0,1
                    for lk in range(2):
                        for li in range(2):
                            W[(k0+lk)*n + li, col] = strassen_W[r, lk*2+li]
                    col += 1
        
        # Schoolbook for B column 4: 2*4*1 = 8 terms
        for i in range(n):
            for j in range(m):
                U[i*m+j, col] = 1
                V[j*p+4, col] = 1
                W[4*n+i, col] = 1
                col += 1
        
        assert col == rank, f"Expected {rank}, got {col}"
        return U.tolist(), V.tolist(), W.tolist()

    def try_reduce(U, V, W, budget_s=30.0):
        """
        Try to reduce rank by:
        1. Removing zero columns
        2. Trying to merge/remove columns via least-squares fit
        """
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Step 1: Remove zero columns
        keep = []
        for r in range(rank):
            if not (np.all(U_arr[:, r] == 0) and np.all(V_arr[:, r] == 0) and np.all(W_arr[:, r] == 0)):
                keep.append(r)
        
        if len(keep) < rank:
            U2 = U_arr[:, keep].tolist()
            V2 = V_arr[:, keep].tolist()
            W2 = W_arr[:, keep].tolist()
            if verify(U2, V2, W2):
                return U2, V2, W2
        
        # Step 2: Try removing one column at a time and refitting
        # For each column to remove, project its contribution onto remaining columns
        for r_remove in range(rank):
            if time.time() - start_time > deadline - 5:
                break
            others = [r for r in range(rank) if r != r_remove]
            # Try to express column r_remove as combination of others
            # U[:, r_remove] ≈ sum_{r in others} alpha_r * U[:, r]
            # This is a 3-way problem; approximate by checking if the tensor contribution
            # of column r_remove can be absorbed.
            
            # Build the tensor contribution of column r_remove
            U_r = U_arr[:, r_remove].reshape(n, m)
            V_r = V_arr[:, r_remove].reshape(m, p)
            W_r = W_arr[:, r_remove].reshape(p, n)
            
            # For each (i,j,k), the contribution is U_r[i,j]*V_r[j,k]*W_r[k,i]
            # We need to find coefficients such that this equals sum_{r in others} c_r * U_r'[i,j]*V_r'[j,k]*W_r'[k,i]
            
            # This is a bilinear system; use least squares on a subset of entries
            # Sample some (i,j,k) entries
            samples = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target = U_r[i,j] * V_r[j,k] * W_r[k,i]
                        row = np.zeros(len(others))
                        for idx, r in enumerate(others):
                            row[idx] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                        samples.append((row, target))
            
            if not samples:
                continue
            
            A = np.array([s[0] for s in samples])
            b = np.array([s[1] for s in samples])
            
            # Check if b is in the column space of A
            # Use lstsq
            coeffs, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
            residual = np.linalg.norm(A @ coeffs - b)
            
            if residual < 1e-8:
                # Column r_remove is redundant; remove it
                U2 = U_arr[:, others].tolist()
                V2 = V_arr[:, others].tolist()
                W2 = W_arr[:, others].tolist()
                if verify(U2, V2, W2):
                    return U2, V2, W2
        
        return U, V, W

    def try_optimize_rank(U, V, W, target_rank, budget_s=20.0):
        """
        Try to find a decomposition of target_rank using alternating least squares.
        """
        U0 = np.array(U, dtype=float)
        V0 = np.array(V, dtype=float)
        W0 = np.array(W, dtype=float)
        
        # Initialize with the current decomposition's first target_rank columns
        U_opt = U0[:, :target_rank].copy()
        V_opt = V0[:, :target_rank].copy()
        W_opt = W0[:, :target_rank].copy()
        
        # Add small random perturbation to avoid local minima
        np.random.seed(42)
        U_opt += 0.01 * np.random.randn(n*m, target_rank)
        V_opt += 0.01 * np.random.randn(m*p, target_rank)
        W_opt += 0.01 * np.random.randn(p*n, target_rank)
        
        # Reshape for convenience
        U3 = U_opt.reshape(n, m, target_rank)
        V3 = V_opt.reshape(m, p, target_rank)
        W3 = W_opt.reshape(p, n, target_rank)
        
        # ALS: fix two, solve for third
        for iteration in range(200):
            if time.time() - start_time > deadline - 5:
                break
            
            # Solve for U: for each (i,j), U3[i,j,:] should satisfy
            # sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] = 1 for all k
            # This is: U3[i,j,:] @ (V3[j,:,:] * W3[:,i,:]) = ones(p)
            # Let M[j,k] = V3[j,k,r] * W3[k,i,r] summed over r... no.
            # For fixed (i,j): sum_r U3[i,j,r] * (V3[j,k,r] * W3[k,i,r]) = 1 for all k
            # Define B[j,k,r] = V3[j,k,r] * W3[k,i,r]
            # Then: sum_r U3[i,j,r] * B[j,k,r] = 1 for all k
            # This is: (B[j,:,:] @ U3[i,j,:]) = ones(p)
            # B[j,:,:] is (p, target_rank), U3[i,j,:] is (target_rank,)
            
            for i in range(n):
                for j in range(m):
                    # B[k, r] = V3[j,k,r] * W3[k,i,r] for k in range(p)
                    B = V3[j, :, :] * W3[:, i, :]  # (p, target_rank)
                    # Solve B @ x = ones(p)
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    U3[i, j, :] = x
            
            # Solve for V: for each (j,k), sum_r V3[j,k,r] * U3[i,j,r] * W3[k,i,r] = 1 for all i
            # B[i, r] = U3[i,j,r] * W3[k,i,r]
            for j in range(m):
                for k in range(p):
                    B = U3[:, j, :] * W3[k, :, :]  # (n, target_rank)
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V3[j, k, :] = x
            
            # Solve for W: for each (k,i), sum_r W3[k,i,r] * U3[i,j,r] * V3[j,k,r] = 1 for all j
            # B[j, r] = U3[i,j,r] * V3[j,k,r]
            for k in range(p):
                for i in range(n):
                    B = U3[i, :, :] * V3[:, k, :]  # (m, target_rank)
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    W3[k, i, :] = x
            
            # Check convergence
            U_opt = U3.reshape(n*m, target_rank)
            V_opt = V3.reshape(m*p, target_rank)
            W_opt = W3.reshape(p*n, target_rank)
            
            U3 = U_opt.reshape(n, m, target_rank)
            V3 = V_opt.reshape(m, p, target_rank)
            W3 = W_opt.reshape(p, n, target_rank)
            
            # Compute error
            temp = np.einsum('jkr,kir->jki', V3, W3)
            T = np.einsum('ijr,jki->ijk', U3, temp)
            error = np.max(np.abs(T - 1.0))
            
            if error < 1e-6:
                U_list = U_opt.tolist()
                V_list = V_opt.tolist()
                W_list = W_opt.tolist()
                if verify(U_list, V_list, W_list):
                    return U_list, V_list, W_list
            
            if error < 1e-3 and iteration % 20 == 0:
                # Try to polish
                pass
        
        return None

    # Build the rank-36 decomposition
    U, V, W = build_strassen_36()
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook rank 40
        rank_schoolbook = n * m * p
        U_school = np.zeros((n*m, rank_schoolbook))
        V_school = np.zeros((m*p, rank_schoolbook))
        W_school = np.zeros((p*n, rank_schoolbook))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, column] = 1
                    V_school[j*p+k, column] = 1
                    W_school[k*n+i, column] = 1
                    column += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to reduce rank
    if time.time() - start_time < deadline - 10:
        U_red, V_red, W_red = try_reduce(U, V, W, budget_s=20.0)
        if len(U_red[0]) < len(U[0]):
            U, V, W = U_red, V_red, W_red
    
    # Try to find rank-35 via optimization
    if time.time() - start_time < deadline - 5:
        result = try_optimize_rank(U, V, W, 35, budget_s=15.0)
        if result is not None:
            U, V, W = result
    
    return {"U": U, "V": V, "W": W}
