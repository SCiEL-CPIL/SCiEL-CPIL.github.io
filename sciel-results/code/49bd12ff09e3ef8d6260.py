import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T as a flat vector of non-zero entries and indices
    # T is 1 at (i,j), (j,k), (i,k)
    # Indices:
    # u_idx = i * m + j
    # v_idx = j * p + k
    # w_idx = k * n + i
    
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                nonzero_indices.append((u_idx, v_idx, w_idx))
                
    # Total number of entries in the tensor product space
    total_entries = dim_u * dim_v * dim_w
    # We only care about the non-zero ones for the "1" constraint
    # and the zero ones for the "0" constraint.
    # Actually, we can just check the non-zero ones equal 1 and a random subset of zero ones equal 0?
    # No, for exactness, ALL must be satisfied.
    # But ALS will minimize the global error.
    
    def check_solution(U, V, W, tol=1e-9):
        # Check non-zero entries
        for u_idx, v_idx, w_idx in nonzero_indices:
            val = sum(U[u_idx, c] * V[v_idx, c] * W[w_idx, c] for c in range(U.shape[1]))
            if abs(val - 1.0) > tol:
                return False
        # Check a sample of zero entries to ensure they are close to 0
        # Full check is expensive, but we can do it if r is small
        # Let's do a full check for small r
        if U.shape[1] <= 24:
            for u_idx in range(dim_u):
                for v_idx in range(dim_v):
                    for w_idx in range(dim_w):
                        # Skip non-zero
                        is_nonzero = False
                        for nz in nonzero_indices:
                            if nz[0] == u_idx and nz[1] == v_idx and nz[2] == w_idx:
                                is_nonzero = True
                                break
                        if not is_nonzero:
                            val = sum(U[u_idx, c] * V[v_idx, c] * W[w_idx, c] for c in range(U.shape[1]))
                            if abs(val) > tol:
                                return False
        return True

    def snap_to_half_integers(U, V, W):
        # Try to snap to k/2
        def snap_matrix(M):
            M_snapped = np.zeros_like(M)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    val = M[i, j]
                    # Try k/2 for k in -10..10
                    best_k = 0
                    best_err = abs(val - 0)
                    for k in range(-20, 21):
                        candidate = k / 2.0
                        err = abs(val - candidate)
                        if err < best_err:
                            best_err = err
                            best_k = k
                    M_snapped[i, j] = best_k / 2.0
            return M_snapped
        
        U_s = snap_matrix(U)
        V_s = snap_matrix(V)
        W_s = snap_matrix(W)
        return U_s, V_s, W_s

    def run_als(U, V, W, max_iters=500, tol=1e-10):
        r = U.shape[1]
        prev_err = float('inf')
        for it in range(max_iters):
            # Update U
            # Fix V, W. Minimize sum (T - sum U_c V_c W_c)^2
            # This is linear in U.
            # For each u_idx, we want to find U[u_idx, :]
            # The term involving U[u_idx, c] is U[u_idx, c] * (V[v_idx, c] * W[w_idx, c])
            # We want to match T[u_idx, v_idx, w_idx]
            
            # Construct matrix A_u for each u_idx?
            # A_u has rows for each (v_idx, w_idx) pair.
            # A_u[row, c] = V[v_idx, c] * W[w_idx, c]
            # b_u[row] = T[u_idx, v_idx, w_idx]
            
            # Since T is sparse, we can be efficient.
            # But dim_v * dim_w = 12 * 8 = 96.
            # dim_u = 6.
            # So we solve 6 systems of size 96 x r.
            
            for u_idx in range(dim_u):
                A = np.zeros((dim_v * dim_w, r))
                b = np.zeros(dim_v * dim_w)
                row = 0
                for v_idx in range(dim_v):
                    for w_idx in range(dim_w):
                        # Determine T value
                        t_val = 0.0
                        # Check if (u_idx, v_idx, w_idx) is in nonzero_indices
                        # Optimization: precompute a set of nonzero tuples
                        # For now, linear search is slow. Use a dict.
                        # Let's use a set of tuples
                        if (u_idx, v_idx, w_idx) in nonzero_set:
                            t_val = 1.0
                        b[row] = t_val
                        for c in range(r):
                            A[row, c] = V[v_idx, c] * W[w_idx, c]
                        row += 1
                # Solve A U_col = b
                # Use least squares
                U[u_idx, :], residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                
            # Update V
            for v_idx in range(dim_v):
                A = np.zeros((dim_u * dim_w, r))
                b = np.zeros(dim_u * dim_w)
                row = 0
                for u_idx in range(dim_u):
                    for w_idx in range(dim_w):
                        if (u_idx, v_idx, w_idx) in nonzero_set:
                            b[row] = 1.0
                        else:
                            b[row] = 0.0
                        for c in range(r):
                            A[row, c] = U[u_idx, c] * W[w_idx, c]
                        row += 1
                V[v_idx, :], residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                
            # Update W
            for w_idx in range(dim_w):
                A = np.zeros((dim_u * dim_v, r))
                b = np.zeros(dim_u * dim_v)
                row = 0
                for u_idx in range(dim_u):
                    for v_idx in range(dim_v):
                        if (u_idx, v_idx, w_idx) in nonzero_set:
                            b[row] = 1.0
                        else:
                            b[row] = 0.0
                        for c in range(r):
                            A[row, c] = U[u_idx, c] * V[v_idx, c]
                        row += 1
                W[w_idx, :], residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                
            # Calculate error
            err = 0.0
            for u_idx, v_idx, w_idx in nonzero_indices:
                val = sum(U[u_idx, c] * V[v_idx, c] * W[w_idx, c] for c in range(r))
                err += (val - 1.0)**2
            # Sample zero errors
            # Skip for speed, rely on convergence
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W

    # Precompute nonzero set for O(1) lookup
    nonzero_set = set(nonzero_indices)

    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Try ranks 19, 20, 21, 22
    for r in [19, 20, 21, 22]:
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts depends on time
        # Estimate time per restart
        # For r=19, ALS is fast.
        # Let's do 20 restarts.
        restarts = 20
        
        for restart in range(restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart)
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)
            
            U, V, W = run_als(U, V, W)
            
            # Check if close to solution
            # Calculate max error on non-zeros
            max_err = 0
            for u_idx, v_idx, w_idx in nonzero_indices:
                val = sum(U[u_idx, c] * V[v_idx, c] * W[w_idx, c] for c in range(r))
                max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-5:
                # Try to snap
                U_s, V_s, W_s = snap_to_half_integers(U, V, W)
                if check_solution(U_s, V_s, W_s):
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
                # Try snapping to integers
                def snap_int(M):
                    M_snapped = np.zeros_like(M)
                    for i in range(M.shape[0]):
                        for j in range(M.shape[1]):
                            M_snapped[i, j] = round(M[i, j])
                    return M_snapped
                
                U_si, V_si, W_si = snap_int(U), snap_int(V), snap_int(W)
                if check_solution(U_si, V_si, W_si):
                    return {"U": U_si.tolist(), "V": V_si.tolist(), "W": W_si.tolist()}

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
