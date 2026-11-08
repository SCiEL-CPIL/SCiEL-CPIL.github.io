import numpy as np
import time
import scipy.optimize as opt
from numpy.linalg import svd

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    
    # Base solution (Rank 22)
    U_base = np.zeros((n * m, rank))
    V_base = np.zeros((m * p, rank))
    W_base = np.zeros((p * n, rank))
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U_base[idx][col] += val
        for idx, val in v_indices:
            V_base[idx][col] += val
        for idx, val in w_indices:
            W_base[idx][col] += val

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Outer: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    # Convert to lists for return if search fails
    def to_lists(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Target Tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                # C[i, l] += A[i, j] * B[j, l]
                # W index: k*n+i where k=l, i=i -> l*n + i
                T[i * m + j, j * p + l, l * n + i] = 1

    def verify(U, V, W, tol=1e-9):
        recon = np.zeros(T.shape)
        for c in range(U.shape[1]):
            recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(recon, T, atol=tol, rtol=0):
            return False
        for M in (U, V, W):
            for v in M.flat:
                if v != int(v) and 2 * v != int(2 * v):
                    return False
        return True

    def greedy_construct(rank_target, time_limit, seed=42):
        """
        Greedy iterative rank-reduction:
        For each rank position, select the next rank-1 component by solving
        a least-squares problem on the residual tensor's mode-3 flattening.
        """
        rng = np.random.RandomState(seed)
        U = np.zeros((n * m, rank_target))
        V = np.zeros((m * p, rank_target))
        W = np.zeros((p * n, rank_target))
        
        # Residual tensor
        residual = T.copy()
        
        for c in range(rank_target):
            # Check time
            if time.time() > time_limit:
                break
                
            # Flatten residual to mode-3: shape (p*n, n*m * m*p)
            # residual[i,j,k] -> flatten to (k, i*m + j)
            res_flat = residual.reshape(p * n, n * m * m * p)
            
            # For rank-1 tensor u ⊗ v ⊗ w:
            # res_flat[k, ij] = w[k] * (u[i] * v[j])
            # This is equivalent to: for each k, res_flat[k,:] = w[k] * outer(u, v).flatten()
            # So res_flat = w[:,None] * (outer(u,v)).flatten()[:,None] * ones
            # Actually: res_flat[k, ij] = w[k] * u[i] * v[j]
            # Let M = outer(u,v).flatten() (size n*m*m*p)
            # Then res_flat = w[:, None] * M[None, :]
            # This is a rank-1 matrix factorization of res_flat!
            
            # Use SVD to find best rank-1 approximation
            try:
                U_svd, s, Vt_svd = svd(res_flat, full_matrices=False)
            except np.linalg.LinAlgError:
                # Fallback: use largest entry
                idx = np.unravel_index(np.argmax(np.abs(res_flat)), res_flat.shape)
                w_k = res_flat[idx]
                # Initialize u and v from first row/col
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                u[idx[1] // m] = 1.0
                v[idx[1] % m] = 1.0
                # Scale
                if np.abs(w_k) > 1e-12:
                    scale = np.sqrt(np.abs(w_k))
                    u[idx[1] // m] = scale
                    v[idx[1] % m] = w_k / scale if scale != 0 else 0
                U[:, c] = u
                V[:, c] = v
                W[:, c] = w_k
                residual -= np.einsum('i,j,k->ijk', u, v, np.zeros(p*n) + w_k)
                # Actually need to be more careful
                w_vec = np.zeros(p * n)
                w_vec[idx[0]] = w_k
                residual -= np.einsum('i,j,k->ijk', u, v, w_vec)
                continue
            
            # Take first left and right singular vectors
            if len(U_svd.shape) < 2 or U_svd.shape[1] == 0 or Vt_svd.shape[1] == 0:
                break
                
            u_flat = U_svd[:, 0]  # size p*n
            v_flat = Vt_svd[0, :]  # size n*m*m*p
            s_val = s[0]
            
            # u_flat is w (size p*n)
            # v_flat is outer(u,v).flatten() (size n*m*m*p)
            w_vec = u_flat * s_val
            outer_uv = v_flat / s_val if abs(s_val) > 1e-15 else v_flat
            
            # Reshape outer_uv to (n*m, m*p)
            outer_mat = outer_uv.reshape(n * m, m * p)
            
            # We need to factor outer_mat = u * v^T
            # Use SVD on outer_mat
            try:
                U2, s2, V2t = svd(outer_mat, full_matrices=False)
            except np.linalg.LinAlgError:
                # Fallback
                idx2 = np.unravel_index(np.argmax(np.abs(outer_mat)), outer_mat.shape)
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                u_vec[idx2[0]] = 1.0
                v_vec[idx2[1]] = 1.0
                if abs(outer_mat[idx2]) > 1e-12:
                    scale = np.sqrt(abs(outer_mat[idx2]))
                    u_vec[idx2[0]] = scale
                    v_vec[idx2[1]] = outer_mat[idx2] / scale if scale != 0 else 0
                U[:, c] = u_vec
                V[:, c] = v_vec
                W[:, c] = w_vec
                residual -= np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
                continue
            
            if len(s2) > 0 and s2[0] > 1e-15:
                u_vec = U2[:, 0] * s2[0]
                v_vec = V2t[0, :]
            else:
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
            
            # Refine with least squares: given u and v, solve for w
            # residual[k, ij] ≈ w[k] * u[i] * v[j]
            # For each k: residual[k,:] = w[k] * outer(u,v).flatten()
            # Let M = outer(u,v).flatten()
            # Then residual_flat = w[:,None] * M[None,:]
            # w = argmin ||residual_flat - w[:,None] * M[None,:]||
            # This is: for each k, w[k] = <residual_flat[k,:], M> / ||M||^2
            # But this assumes M is fixed. Let's do a proper LS.
            
            # Actually, let's refine all three vectors using alternating LS
            # Given V and W, solve for U:
            # residual[i,j,k] ≈ U[i,c] * V[j,c] * W[k,c]
            # For each i: residual[i,:,:] = U[i,c] * outer(V[:,c], W[:,c])
            # residual_reshaped: (n*m, m*p, p*n)
            # For fixed i: residual[i,:,:] should be U[i,c] * np.outer(V[:,c], W[:,c])
            # So U[i,c] = <residual[i,:,:], outer(V[:,c], W[:,c])> / ||outer(V[:,c], W[:,c])||^2
            
            # Let's do alternating minimization to refine
            for _ in range(5):
                # Solve for U given V, W
                outer_VW = np.outer(V[:, c], W[:, c])  # (m*p, p*n)
                norm_VW = np.sum(outer_VW ** 2)
                if norm_VW > 1e-15:
                    for i in range(n * m):
                        U[i, c] = np.sum(residual[i, :, :] * outer_VW) / norm_VW
                else:
                    U[i, c] = 0.0
                
                # Solve for V given U, W
                outer_UW = np.outer(U[:, c], W[:, c])  # (n*m, p*n)
                norm_UW = np.sum(outer_UW ** 2)
                if norm_UW > 1e-15:
                    for j in range(m * p):
                        V[j, c] = np.sum(residual[:, j, :] * outer_UW) / norm_UW
                else:
                    V[j, c] = 0.0
                
                # Solve for W given U, V
                outer_UV = np.outer(U[:, c], V[:, c])  # (n*m, m*p)
                norm_UV = np.sum(outer_UV ** 2)
                if norm_UV > 1e-15:
                    for k in range(p * n):
                        W[k, c] = np.sum(residual[:, :, k] * outer_UV) / norm_UV
                else:
                    W[k, c] = 0.0
            
            # Update residual
            residual -= np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        return U, V, W

    # Try to find rank 21 solution using greedy approach
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_solution = to_lists(U_base, V_base, V_base, W_base) if False else to_lists(U_base, V_base, W_base)
    best_rank = 22
    
    # Try greedy construction for various ranks
    for r_target in [21, 20, 19]:
        if time.time() > deadline - 5:
            break
        
        for seed in [42, 123, 999, 7, 2024]:
            if time.time() > deadline - 5:
                break
            
            try:
                U_g, V_g, W_g = greedy_construct(r_target, deadline - 5, seed)
                
                # Round to half-integers
                U_round = np.round(U_g * 2) / 2
                V_round = np.round(V_g * 2) / 2
                W_round = np.round(W_g * 2) / 2
                
                if verify(U_round, V_round, W_round):
                    best_solution = to_lists(U_round, V_round, W_round)
                    best_rank = r_target
                    print(f"Found rank {r_target} solution with seed {seed}")
                    break  # Found valid solution for this rank, try lower
            except Exception as e:
                pass
        
        if best_rank <= r_target:
            break
    
    # Also try the original least_squares approach if time permits
    if time.time() < deadline - 10 and best_rank >= 21:
        try:
            r_target = 21
            U_init = U_base[:, :r_target].flatten()
            V_init = V_base[:, :r_target].flatten()
            W_init = W_base[:, :r_target].flatten()
            x0 = np.concatenate([U_init, V_init, W_init])
            
            def residual_func(x):
                U = x[:n*m*r_target].reshape(n*m, r_target)
                V = x[n*m*r_target : n*m*r_target + m*p*r_target].reshape(m*p, r_target)
                W = x[n*m*r_target + m*p*r_target :].reshape(p*n, r_target)
                recon = np.zeros(T.shape)
                for c in range(r_target):
                    recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                return (recon - T).flatten()
            
            result = opt.least_squares(residual_func, x0, method='lm', max_nfev=100)
            
            if time.time() < deadline - 5:
                x_opt = result.x
                U_opt = x_opt[:n*m*r_target].reshape(n*m, r_target)
                V_opt = x_opt[n*m*r_target : n*m*r_target + m*p*r_target].reshape(m*p, r_target)
                W_opt = x_opt[n*m*r_target + m*p*r_target :].reshape(p*n, r_target)
                
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify(U_round, V_round, W_round) and r_target < best_rank:
                    best_solution = to_lists(U_round, V_round, W_round)
                    best_rank = r_target
        except Exception as e:
            pass

    return best_solution
