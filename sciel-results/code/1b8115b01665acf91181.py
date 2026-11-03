import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    
    # Indices for the tensor T
    # T is a linear map A (n x m) x B (m x p) -> C (n x p)
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor indices: (i, j), (j, k), (i, k)
    # Shape of U: (n*m, r)
    # Shape of V: (m*p, r)
    # Shape of W: (p*n, r) -- Note: W[k*n + i, r] in prompt
    
    # We want to find U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is valid, 0 otherwise.
    # Actually, the tensor T has 1s at (i*m+j, j*p+k, k*n+i) for all i,j,k.
    # And 0s elsewhere.
    
    # Let's create the list of non-zero entries
    non_zero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                non_zero_entries.append((idx_u, idx_v, idx_w))
    
    num_entries = len(non_zero_entries)
    
    # Helper to verify a decomposition
    def verify(U, V, W):
        r = U.shape[1]
        # Check all non-zero entries
        for idx_u, idx_v, idx_w in non_zero_entries:
            val = 0.0
            for c in range(r):
                val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
            if abs(val - 1.0) > 1e-9:
                return False
        # Check a sample of zero entries to ensure no spurious non-zeros
        # Total possible indices: (n*m) * (m*p) * (p*n)
        # We can't check all, but we can check random ones
        # However, if the rank is minimal and the system is determined, 
        # matching the non-zeros might imply the zeros are handled if the 
        # decomposition is unique or if we use the full tensor structure.
        # To be safe, we should check the full tensor reconstruction.
        
        # Reconstruct the full tensor from U, V, W
        # T_recon[i, j, k, l, m, n] ... no, it's 3-way.
        # Let's check a few random zero entries.
        # Total size: 6 * 27 * 18 = 2916 entries.
        # Non-zeros: 54.
        # Zeros: 2862.
        # We can check all zeros if it's fast enough.
        # 2862 * r operations. If r=30, 85k ops. Very fast.
        
        # Create a set of non-zero indices for O(1) lookup
        non_zero_set = set(non_zero_entries)
        
        for idx_u in range(n * m):
            for idx_v in range(m * p):
                for idx_w in range(p * n):
                    if (idx_u, idx_v, idx_w) not in non_zero_set:
                        val = 0.0
                        for c in range(r):
                            val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                        if abs(val) > 1e-9:
                            return False
        return True

    # Helper to snap to integers or half-integers
    def snap(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    # ALS Solver for a given rank r
    def solve_als(r, max_iters=500, tol=1e-12):
        # Initialize with random integers in {-1, 0, 1} to help with snapping
        # Or small random floats
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Precompute indices for efficiency
        u_idx, v_idx, w_idx = zip(*non_zero_entries)
        u_idx = np.array(u_idx)
        v_idx = np.array(v_idx)
        w_idx = np.array(w_idx)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # For each u, minimize sum_k (T[u, v_k, w_k] - sum_c U[u, c] V[v_k, c] W[w_k, c])^2
            # This is a linear least squares problem for U[u, :].
            # Let A_u be a matrix of size (num_entries_for_u, r)
            # A_u[k, c] = V[v_k, c] * W[w_k, c]
            # b_u[k] = 1 (since T is 1 at these spots)
            # We need to handle that different u's have different sets of (v, w).
            
            # Actually, it's easier to do block updates.
            # Fix V, W. Solve for U.
            # The error is sum_{u,v,w} (T_{uvw} - sum_c U_{uc} V_{vc} W_{wc})^2
            # For a fixed u, the term is sum_{v,w} (T_{uvw} - sum_c U_{uc} (V_{vc} W_{wc}))^2
            # Let M_u be the matrix where M_u[v, w, c] = V[v, c] * W[w, c].
            # We want to find U[u, :] to best fit the slice T[u, :, :].
            
            # Since T is sparse, we only care about the non-zero entries.
            # But wait, the zero entries also contribute to the error!
            # If we only minimize on non-zeros, we might get spurious non-zeros.
            # So we must include zero entries in the least squares problem.
            
            # This is expensive if we do it for all zeros.
            # However, n*m=6, m*p=27, p*n=18.
            # Total entries 2916.
            # For each u (6 values), we have 27*18 = 486 entries.
            # We can build the design matrix for each u.
            
            U_new = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    # Collect all (v, w) pairs
                    # v ranges 0..26, w ranges 0..17
                    # Design matrix A of size (486, r)
                    A = np.zeros((m * p * p * n, r))
                    b = np.zeros(m * p * p * n)
                    
                    idx = 0
                    for v in range(m * p):
                        for w in range(p * n):
                            # Calculate V[v, :] * W[w, :]
                            A[idx, :] = V[v, :] * W[w, :]
                            # Target T
                            # Is (u, v, w) a non-zero entry?
                            # u = i*m+j, v = j'*p+k, w = k'*n+i'
                            # Non-zero if j==j' and i==i' and k==k'?
                            # u = i*m+j
                            # v = j*p+k
                            # w = k*n+i
                            # So for fixed u (i,j), non-zero if v starts with j and w ends with i.
                            # v = j*p+k. So v // p must be j.
                            # w = k*n+i. So w % n must be i.
                            if (v // p == j) and (w % n == i):
                                b[idx] = 1.0
                            else:
                                b[idx] = 0.0
                            idx += 1
                    
                    # Solve A x = b
                    U_new[u, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            U = U_new
            
            # Update V
            V_new = np.zeros((m * p, r))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n * m * p * n, r))
                    b = np.zeros(n * m * p * n)
                    
                    idx = 0
                    for u in range(n * m):
                        for w in range(p * n):
                            A[idx, :] = U[u, :] * W[w, :]
                            # Non-zero if u corresponds to (i, j) and w to (k, i)
                            # u = i*m+j => u % m == j
                            # w = k*n+i => w // n == k
                            if (u % m == j) and (w // n == k):
                                b[idx] = 1.0
                            else:
                                b[idx] = 0.0
                            idx += 1
                            
                    V_new[v, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            V = V_new
            
            # Update W
            W_new = np.zeros((p * n, r))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((n * m * m * p, r))
                    b = np.zeros(n * m * m * p)
                    
                    idx = 0
                    for u in range(n * m):
                        for v in range(m * p):
                            A[idx, :] = U[u, :] * V[v, :]
                            # Non-zero if u=(i,j) and v=(j,k)
                            # u = i*m+j => u // m == i
                            # v = j*p+k => v % p == k
                            if (u // m == i) and (v % p == k):
                                b[idx] = 1.0
                            else:
                                b[idx] = 0.0
                            idx += 1
                            
                    W_new[w, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            W = W_new
            
            # Check convergence
            # Calculate error on non-zeros
            err = 0
            for idx_u, idx_v, idx_w in non_zero_entries:
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                err += (val - 1.0) ** 2
            if err < tol:
                break
                
        return U, V, W

    # Search for minimum rank
    # Start with a reasonable guess. 54 is schoolbook.
    # Try 20 to 54.
    for r in range(20, 55):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Multiple restarts
        for restart in range(20):
            if time.time() > deadline - 10:
                break
                
            random.seed() # Re-seed for different randoms
            np.random.seed()
            
            U, V, W = solve_als(r, max_iters=100)
            
            # Snap and verify
            U_s, V_s, W_s = snap(U, V, W)
            
            if verify(U_s, V_s, W_s):
                # Found a valid decomposition
                return {
                    "U": U_s.tolist(),
                    "V": V_s.tolist(),
                    "W": W_s.tolist()
                }
                
    # Fallback to schoolbook if no better found
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
# EVOLVE-BLOCK-END
