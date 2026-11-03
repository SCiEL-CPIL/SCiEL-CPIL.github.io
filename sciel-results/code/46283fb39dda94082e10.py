import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    def compute_flattening_ranks():
        """Compute the three-mode flattening ranks of the tensor to determine theoretical minimum rank."""
        # Build the full tensor T[i,j,k] = 1 for the multiplication tensor
        # Actually, the multiplication tensor has T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor is defined by: for each (i,j,k), the coefficient is 1
        # But wait - the tensor for matrix multiplication M_{n,m,p} has entries:
        # T[i,j,k] = 1 (it's the identity tensor in the appropriate sense)
        
        # Actually, let me think about this more carefully.
        # The multiplication tensor M_{n,m,p} is defined such that:
        # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
        # As a tensor, M[i,j,k] = delta (the Kronecker delta structure)
        
        # For the standard matrix multiplication tensor, T[i,j,k] = 1 for all valid indices
        # This is because each product term A[i,j]*B[j,k] contributes to output (i,k)
        
        # The three mode flattening matrices:
        # Mode-1 flattening: T_1 has shape (n, m*p), T_1[i, j*p+k] = T[i,j,k]
        # Mode-2 flattening: T_2 has shape (m, n*p), T_2[j, i*p+k] = T[i,j,k]  
        # Mode-3 flattening: T_3 has shape (p, n*m), T_3[k, i*m+j] = T[i,j,k]
        
        T = np.ones((n, m, p))
        
        # Mode-1 flattening
        T1 = T.reshape(n, m*p)
        # Mode-2 flattening
        T2 = T.transpose(1,0,2).reshape(m, n*p)
        # Mode-3 flattening
        T3 = T.transpose(2,0,1).reshape(p, n*m)
        
        # Compute ranks via SVD
        r1 = np.linalg.matrix_rank(T1)
        r2 = np.linalg.matrix_rank(T2)
        r3 = np.linalg.matrix_rank(T3)
        
        # The tensor rank is at least max(r1, r2, r3)
        theoretical_min_rank = max(r1, r2, r3)
        
        return theoretical_min_rank, r1, r2, r3

    def check_solution(U, V, W, R):
        """Check if the decomposition is exact. Returns max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def linearized_rank_check(U, V, R):
        """
        Given U and V with R columns, check if there exists W such that
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
        
        This is linear in W. For each (i,j), we have a system:
        sum_k V[j*p+k, r] * W[k*n+i, r] = 1 / U[i*m+j, r] for r where U[i*m+j,r] != 0
        
        Actually, let's reformulate. The constraint is:
        For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        Let's define for each r: u_r = U[:,r], v_r = V[:,r], w_r = W[:,r]
        Then: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
        
        This is linear in w_r for fixed u_r, v_r.
        For each output position (k*n+i), we have:
        sum_r w_r[k*n+i] * (u_r[i*m+j] * v_r[j*p+k]) = 1
        
        For fixed (i,j), as k varies, we get p equations in R unknowns (w_r[k*n+i] for r=0..R-1).
        But w_r[k*n+i] appears in multiple equations (for different j).
        
        Let's set up the full linear system. Unknowns: W[k*n+i, r] for k=0..p-1, i=0..n-1, r=0..R-1.
        Total unknowns: p*n*R
        
        Constraints: for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        Total constraints: n*m*p
        
        We can write this as M * w_flat = b where w_flat is the vectorized W.
        """
        # Build the linear system M * w_flat = b
        # Unknowns indexed by (k, i, r) -> index = (k*n + i)*R + r
        # Constraints indexed by (i, j, k)
        
        num_unknowns = p * n * R
        num_constraints = n * m * p
        
        # Use sparse representation for efficiency
        # For each constraint (i,j,k), the equation is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # The unknowns involved are W[k*n+i, r] for r=0..R-1
        # Coefficient for W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
        
        # Build system
        rows = []
        cols = []
        vals = []
        rhs = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        coeff = U[i*m + j, r] * V[j*p + k, r]
                        if abs(coeff) > 1e-15:
                            unknown_idx = (k * n + i) * R + r
                            rows.append(len(rhs))
                            cols.append(unknown_idx)
                            vals.append(coeff)
                    rhs.append(1.0)
        
        if len(rows) == 0:
            return None, None
        
        from scipy.sparse import csr_matrix
        M = csr_matrix((vals, (rows, cols)), shape=(num_constraints, num_unknowns))
        
        # Solve least squares
        try:
            # Use scipy's sparse least squares
            from scipy.sparse.linalg import lsqr
            w_flat, info = lsqr(M, np.array(rhs), atol=1e-12, btol=1e-12, itermax=1000)
            
            # Check residual
            residual = M @ w_flat - np.array(rhs)
            max_err = np.max(np.abs(residual))
            
            if max_err < 1e-10:
                # Reshape w_flat back to W
                W = w_flat.reshape(p, n, R).transpose(1, 0, 2).reshape(p*n, R)
                return W, max_err
            else:
                return None, max_err
        except:
            return None, float('inf')
    
    def find_rank27_by_linearization(U28, V28, W28, rank28, deadline):
        """
        Try to find rank-27 solution by fixing U and V supports.
        For each subset of columns of U28 and V28 of size 27, check if W exists.
        """
        R = 27
        
        # Strategy: try removing one column from each of U28, V28 (keeping 27 columns)
        # For each pair of removals, check if the linear system has a solution
        
        best_W = None
        best_err = float('inf')
        
        for remove_u in range(rank28):
            if time.time() > deadline - 5:
                break
            
            u_cols = [c for c in range(rank28) if c != remove_u]
            U = U28[:, u_cols]
            
            for remove_v in range(rank28):
                if time.time() > deadline - 3:
                    break
                
                v_cols = [c for c in range(rank28) if c != remove_v]
                V = V28[:, v_cols]
                
                W, err = linearized_rank_check(U, V, R)
                if W is not None and err < best_err:
                    best_err = err
                    best_W = W
                    if err < 1e-10:
                        return U, V, W, R, err
        
        if best_W is not None and best_err < 1e-10:
            return None, None, best_W, R, best_err
        return None, None, None, R, best_err
    
    def svd_based_initialization(R):
        """Use SVD of flattening matrices to get good initial basis for factor matrices."""
        T = np.ones((n, m, p))
        
        # Mode-1 flattening: shape (n, m*p)
        T1 = T.reshape(n, m*p)
        U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
        
        # Mode-2 flattening: shape (m, n*p)
        T2 = T.transpose(1,0,2).reshape(m, n*p)
        U2, S2, Vt2 = np.linalg.svd(T2, full_matrices=False)
        
        # Mode-3 flattening: shape (p, n*m)
        T3 = T.transpose(2,0,1).reshape(p, n*m)
        U3, S3, Vt3 = np.linalg.svd(T3, full_matrices=False)
        
        # Initialize U, V, W using the left singular vectors from the SVDs
        # U should have shape (n*m, R)
        # V should have shape (m*p, R)
        # W should have shape (p*n, R)
        
        # For U: use columns from Vt1 (right singular vectors of T1)
        # Vt1 has shape (m*p, min(n, m*p)), we need n*m rows
        # This is tricky - let me think differently
        
        # Actually, let's use a different approach:
        # Initialize with random but structured values based on the SVD spectra
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Use the singular values to weight the initialization
        # For each factor, mix the SVD components
        
        # Simple approach: use the first min(R, rank) right singular vectors
        for r in range(R):
            # U initialization from T1's right singular vectors
            if r < Vt1.shape[1]:
                U[:, r] = Vt1[:, r].reshape(n, m*p).reshape(-1)
            else:
                U[:, r] = np.random.randn(n*m)
            
            # V initialization from T2's right singular vectors
            if r < Vt2.shape[1]:
                V[:, r] = Vt2[:, r].reshape(m, n*p).reshape(-1)
            else:
                V[:, r] = np.random.randn(m*p)
            
            # W initialization from T3's right singular vectors
            if r < Vt3.shape[1]:
                W[:, r] = Vt3[:, r].reshape(p, n*m).reshape(-1)
            else:
                W[:, r] = np.random.randn(p*n)
        
        # Normalize to avoid numerical issues
        for r in range(R):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10:
                U[:, r] /= norm_u
            if norm_v > 1e-10:
                V[:, r] /= norm_v
            if norm_w > 1e-10:
                W[:, r] /= norm_w
        
        return U, V, W

    def build_strassen_rank28():
        """Build explicit rank-28 solution using two Strassen 2x2x2 blocks."""
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        c = 0

        def add_strassen_block(j_off, k_off):
            nonlocal c
            # m1 = (a00+a11)*(b00+b11)
            U[0*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
            V[j_off*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
            c += 1

            # m2 = (a10+a11)*b00
            U[1*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
            V[j_off*p + k_off, c] = 1
            W[k_off*n + 1, c] = 1; W[(k_off+1)*n + 1, c] = -1
            c += 1

            # m3 = a00*(b01-b11)
            U[0*m + j_off, c] = 1
            V[j_off*p + (k_off+1), c] = 1; V[(j_off+1)*p + (k_off+1), c] = -1
            W[(k_off+1)*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
            c += 1

            # m4 = a11*(b10-b00)
            U[1*m + (j_off+1), c] = 1
            V[(j_off+1)*p + k_off, c] = 1; V[j_off*p + k_off, c] = -1
            W[k_off*n + 0, c] = 1; W[k_off*n + 1, c] = 1
            c += 1

            # m5 = (a00+a01)*b11
            U[0*m + j_off, c] = 1; U[0*m + (j_off+1), c] = 1
            V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = -1; W[(k_off+1)*n + 0, c] = 1
            c += 1

            # m6 = (a10-a00)*(b00+b01)
            U[1*m + j_off, c] = 1; U[0*m + j_off, c] = -1
            V[j_off*p + k_off, c] = 1; V[j_off*p + (k_off+1), c] = 1
            W[(k_off+1)*n + 1, c] = 1
            c += 1

            # m7 = (a01-a11)*(b10+b11)
            U[0*m + (j_off+1), c] = 1; U[1*m + (j_off+1), c] = -1
            V[(j_off+1)*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
            W[k_off*n + 0, c] = 1
            c += 1

        # Block 1: j in {0,1}, k in {0,1}
        add_strassen_block(0, 0)

        # Block 2: j in {0,1}, k in {2,3}
        add_strassen_block(0, 2)

        # Remaining 14 schoolbook terms (j=2, all k; plus j=0,1 with k=4)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j in (0, 1) and k in (0, 1, 2, 3):
                        continue
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1

        assert c == rank, f"Expected rank {rank}, got {c}"
        return U, V, W, rank

    def solve_for_U(V, W, R):
        U = np.zeros((n*m, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                b = np.ones(p)
                try:
                    U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return U

    def solve_for_V(U, W, R):
        V = np.zeros((m*p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                b = np.ones(n)
                try:
                    V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return V

    def solve_for_W(U, V, R):
        W = np.zeros((p*n, R))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                b = np.ones(m)
                try:
                    W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return W

    def als_optimize(U, V, W, R, max_iters=200, tol=1e-12):
        """Run ALS optimization. Returns (U, V, W, err)."""
        for it in range(max_iters):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            err = check_solution(U, V, W, R)
            if err < tol:
                break
        return U, V, W, err

    def polish(U, V, W, R, max_iters=500, tol=1e-14):
        """High-precision polish pass."""
        for it in range(max_iters):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            err = check_solution(U, V, W, R)
            if err < tol:
                break
        return U, V, W, err

    # Step 1: Build guaranteed rank-28 solution
    U28, V28, W28, rank28 = build_strassen_rank28()
    err28 = check_solution(U28, V28, W28, rank28)
    
    # Step 1.5: Compute theoretical minimum rank from flattening matrices
    if time.time() < deadline - 30:
        theo_min, r1, r2, r3 = compute_flattening_ranks()
        # Use this to guide our search - if theo_min < 28, there might be room for improvement

    # Verify rank-28 is correct
    if err28 > 1e-10:
        # Fallback to schoolbook rank 30
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Step 2: Try to find rank-27 solution
    best_rank = rank28
    best_U, best_V, best_W = U28, V28, W28

    if time.time() < deadline - 10:
        R = 27

        # Strategy D: Linearized tensor rank search
        # Fix U and V supports, solve linear system for W
        U_lin, V_lin, W_lin, R_lin, err_lin = find_rank27_by_linearization(
            U28, V28, W28, rank28, deadline
        )
        
        if W_lin is not None and err_lin < 1e-10:
            # Found a valid rank-27 solution
            # Polish it for higher precision
            U_lin, V_lin, W_lin, err_lin = polish(U_lin, V_lin, W_lin, R_lin, max_iters=500, tol=1e-14)
            if err_lin < 1e-8:
                best_rank = R_lin
                best_U, best_V, best_W = U_lin, V_lin, W_lin

        # Strategy A: Perturb rank-28 by zeroing each column, then ALS
        if best_rank == rank28 and time.time() < deadline - 10:
            for col_to_zero in range(rank28):
                if time.time() > deadline - 5:
                    break
                U = U28.copy()
                V = V28.copy()
                W = W28.copy()
                U[:, col_to_zero] = 0
                V[:, col_to_zero] = 0
                W[:, col_to_zero] = 0
                # Remove that column to get rank 27
                U = np.delete(U, col_to_zero, axis=1)
                V = np.delete(V, col_to_zero, axis=1)
                W = np.delete(W, col_to_zero, axis=1)
                U, V, W, err = als_optimize(U, V, W, R, max_iters=300, tol=1e-12)
                if err < 1e-10:
                    U, V, W, err = polish(U, V, W, R, max_iters=500, tol=1e-14)
                    if err < 1e-8:
                        best_rank = R
                        best_U, best_V, best_W = U, V, W
                        break

        # Strategy B: Structured perturbations - zero pairs of related columns
        if best_rank == rank28 and time.time() < deadline - 5:
            # Try zeroing pairs of columns that might be redundant
            for i in range(rank28):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, rank28):
                    if time.time() > deadline - 4:
                        break
                    U = U28.copy()
                    V = V28.copy()
                    W = W28.copy()
                    # Remove columns i and j
                    cols_to_keep = [c for c in range(rank28) if c != i and c != j]
                    U = U[:, cols_to_keep]
                    V = V[:, cols_to_keep]
                    W = W[:, cols_to_keep]
                    U, V, W, err = als_optimize(U, V, W, R, max_iters=200, tol=1e-12)
                    if err < 1e-10:
                        U, V, W, err = polish(U, V, W, R, max_iters=500, tol=1e-14)
                        if err < 1e-8:
                            best_rank = R
                            best_U, best_V, best_W = U, V, W
                            break
                if best_rank == R:
                    break

        # Strategy C: Random restarts with diverse seeds
        if best_rank == rank28 and time.time() < deadline - 5:
            n_trials = 800
            for trial in range(n_trials):
                if time.time() > deadline - 3:
                    break
                np.random.seed(trial * 7919 + 13)
                
                # Use SVD-based initialization for some trials, random for others
                if trial % 3 == 0:
                    U, V, W = svd_based_initialization(R)
                else:
                    U = np.random.randn(n*m, R)
                    V = np.random.randn(m*p, R)
                    W = np.random.randn(p*n, R)
                
                U, V, W, err = als_optimize(U, V, W, R, max_iters=150, tol=1e-12)
                if err < 1e-10:
                    U, V, W, err = polish(U, V, W, R, max_iters=300, tol=1e-14)
                    if err < 1e-8:
                        best_rank = R
                        best_U, best_V, best_W = U, V, W
                        break

    # Final polish on the best solution found
    if best_rank < rank28 and time.time() < deadline - 2:
        best_U, best_V, best_W, _ = polish(best_U, best_V, best_W, best_rank, max_iters=200, tol=1e-15)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
