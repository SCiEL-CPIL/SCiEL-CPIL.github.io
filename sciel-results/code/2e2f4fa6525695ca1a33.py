import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Construct the target tensor T
    # T[i, j, k] = 1 if i = i1*m + i2, j = i2*p + i3, k = i3*n + i1
    # i in [0, nm), j in [0, mp), k in [0, pn]
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    
    for i1 in range(n):
        for i2 in range(m):
            for i3 in range(p):
                idx_u = i1 * m + i2
                idx_v = i2 * p + i3
                idx_w = i3 * n + i1
                T[idx_u, idx_v, idx_w] = 1.0

    # Try to find a lower rank decomposition
    # Schoolbook rank is n*m*p
    schoolbook_rank = n * m * p
    
    # Define a helper to check if a decomposition is valid and exact
    def is_valid(U, V, W):
        # Check if entries are close to half-integers
        # U, V, W are (rows, rank)
        for M in [U, V, W]:
            # Check if 2*M is close to integer
            if not np.allclose(2 * M, np.round(2 * M), atol=1e-5):
                return False
        
        # Reconstruct tensor
        # T_recon[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Use einsum for efficiency
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Check if it matches T
        if not np.allclose(T_recon, T, atol=1e-5):
            return False
        
        return True

    # Define ALS decomposition
    def als_decompose(rank, iterations=50, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        for it in range(iterations):
            # Update U: for each i, solve V^T W^T u_i = T_i
            # T_i is the matrix T[i, :, :] of shape (m*p, p*n)
            # We want U[:, r] such that sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k]
            # This is linear in U[i, :] if V, W are fixed.
            # Let M_jk_r = V[j, r] W[k, r].
            # We can vectorize this.
            # T_flat[i] is a vector of length (m*p)*(p*n).
            # Design matrix A_i is ((m*p)*(p*n)) x rank.
            # A_i[x, r] = V[j, r] W[k, r] where x maps to (j, k).
            
            # It is faster to loop over rows of U if dimensions are small.
            # n*m is 6. So we solve 6 systems of size (12*8=96) x rank.
            for i in range(n * m):
                # Construct the linear system for U[i, :]
                # T_i is T[i, :, :] reshaped to (-1,)
                t_vec = T[i, :, :].flatten()
                
                # A matrix: rows are (j, k), cols are r
                # A[j, k, r] = V[j, r] * W[k, r]
                # We can form this using outer products or broadcasting
                # V is (mp, rank), W is (pn, rank)
                # We need a matrix of size (mp*pn, rank)
                # A = np.einsum('jr,kr->jkr', V, W).reshape(-1, rank)
                A = np.einsum('jr,kr->jkr', V, W).reshape(-1, rank)
                
                # Solve A * u_i = t_vec
                # Use least squares
                u_i, residuals, rank_ls, sv = np.linalg.lstsq(A, t_vec, rcond=None)
                U[i, :] = u_i

            # Update V: for each j
            for j in range(m * p):
                t_vec = T[:, j, :].flatten() # Shape (nm * pn)
                # A[i, k, r] = U[i, r] * W[k, r]
                A = np.einsum('ir,kr->ikr', U, W).reshape(-1, rank)
                v_j, residuals, rank_ls, sv = np.linalg.lstsq(A, t_vec, rcond=None)
                V[j, :] = v_j

            # Update W: for each k
            for k in range(p * n):
                t_vec = T[:, :, k].flatten() # Shape (nm * mp)
                # A[i, j, r] = U[i, r] * V[j, r]
                A = np.einsum('ir,jr->ijr', U, V).reshape(-1, rank)
                w_k, residuals, rank_ls, sv = np.linalg.lstsq(A, t_vec, rcond=None)
                W[k, :] = w_k

        return U, V, W

    # Try to find a decomposition with rank < schoolbook_rank
    # We iterate from schoolbook_rank - 1 down to a limit
    # Given time constraints, we try a few ranks and a few seeds.
    
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # We will try ranks 23, 22, 21
    # And a few random seeds
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for rank in range(schoolbook_rank - 1, 19, -1):
        for seed in seeds:
            if time.time() - start_time > 110:
                break
            U, V, W = als_decompose(rank, iterations=20, seed=seed)
            
            # Snap to half-integers
            U_snap = np.round(2 * U) / 2.0
            V_snap = np.round(2 * V) / 2.0
            W_snap = np.round(2 * W) / 2.0
            
            if is_valid(U_snap, V_snap, W_snap):
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                best_rank = rank
                break
        if best_rank < schoolbook_rank:
            break

    if best_U is not None:
        # Convert to lists of lists as required by interface
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Return schoolbook
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
