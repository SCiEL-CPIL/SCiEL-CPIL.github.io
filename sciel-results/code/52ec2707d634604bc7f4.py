"""Matrix multiplication 3x3 x 3x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p  # 36
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                U_school[alpha, col] = 1
                V_school[beta, col] = 1
                W_school[gamma, col] = 1
                col += 1
    
    def to_lists(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
                
    T_flat = T.reshape(n*m, m*p * p*n)
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        # Reconstruct tensor
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_recon, T, atol=tol)

    def polish(U, V, W):
        """Snap values to nearest integer or half-integer if close."""
        def snap(x):
            # Check if close to integer or half-integer
            # Multipliers: 1, 0.5
            # We want x to be close to k/2
            # 2x should be close to integer k
            x2 = x * 2
            k = np.round(x2)
            if np.all(np.abs(x2 - k) < 1e-5):
                return k / 2.0
            return x
        return snap(U), snap(V), snap(W)

    def try_rank(rank, seed=0, max_iter=100):
        """Try to find a decomposition of given rank using ALS."""
        if rank >= 36:
            return None
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random factors
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Normalize initial columns to unit norm to help convergence
        for r in range(rank):
            nu = np.linalg.norm(U[:,r])
            nv = np.linalg.norm(V[:,r])
            nw = np.linalg.norm(W[:,r])
            if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                U[:,r] /= nu
                V[:,r] /= nv
                W[:,r] /= nw
                # Scale back to keep product magnitude reasonable
                scale = (nu * nv * nw) ** (1.0/3.0)
                U[:,r] *= scale
                V[:,r] *= scale
                W[:,r] *= scale

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U
            # T_flat (nm x D) approx U (nm x R) * Z^T (R x D)
            # Z has columns A_r = V[:,r] outer W[:,r] flattened
            Z = np.zeros((m*p * p*n, rank))
            for r in range(rank):
                A_r = np.outer(V[:,r], W[:,r]).flatten()
                Z[:, r] = A_r
            try:
                # Solve U * Z^T = T_flat => U = T_flat * Z * (Z^T * Z)^-1
                # Using lstsq: U * Z^T = T_flat
                # Equivalent to: (Z * U^T)^T = T_flat^T ? No.
                # We want X such that X * A = B. X = B * A^+
                U_new, residuals, rank_u, s = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
                U = U_new.T
            except np.linalg.LinAlgError:
                break
                
            # Update V
            # T_flat (nm x D) approx Y (nm x R) * Z^T (R x D) where Y is effectively W part?
            # T = sum U_r outer V_r outer W_r
            # Flatten over U and W: T_flat[:, :] = sum U_r (V_r outer W_r)
            # Let's group U and W.
            # T_flat = (U * W^T) * ... ? No.
            # T_flat[i, j] where i is U index, j is (V,W) index.
            # T_flat = sum_r U[:,r] * (V[:,r] outer W[:,r])^T
            # Let M = U * W^T ? No.
            # Let's use the standard ALS formulation for V.
            # Fix U and W.
            # T_flat (nm x D) = sum_r U[:,r] * (V[:,r] outer W[:,r])^T
            # Let A_r = U[:,r] outer W[:,r] flattened? No, that's not right.
            # T = sum_r U_r \otimes V_r \otimes W_r
            # Reshape T to (nm, mp*pn).
            # T_flat = sum_r U[:,r] * (V[:,r] \otimes W[:,r])^T
            # This is linear in V if we consider the structure?
            # (V[:,r] \otimes W[:,r]) is a vector.
            # Let's define matrices:
            # U (nm x R)
            # W (pn x R)
            # We want V (mp x R).
            # T_flat (nm x mp*pn)
            # Let's reshape T_flat to (nm, mp, pn).
            # T_reshaped = sum_r U[:,r] * V[:,r] * W[:,r] (outer products)
            # This is equivalent to:
            # T_reshaped = U * (V * W^T) ? No.
            # T_reshaped[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
            # This is a linear system for V if U and W are fixed.
            # Let's vectorize T_reshaped to (nm*pn, mp).
            # T_vec (nm*pn, mp)
            # T_vec = (W * U^T) * V ?
            # Let M = W (pn x R) and N = U (nm x R).
            # T_vec[a, j] where a indexes (i,k) and j indexes j.
            # T_vec[a, j] = sum_r W[k,r] V[j,r] U[i,r]
            # Let P = W * U^T ? No.
            # Let's construct the matrix A for V.
            # T_vec = A * V
            # A (nm*pn, R) ? No, A should be (nm*pn, R) and V is (R, mp)?
            # Let's stick to lstsq.
            # We want V such that sum_r V[:,r] * (U[:,r] outer W[:,r])^T = T_flat
            # Let B_r = U[:,r] outer W[:,r] flattened (size nm*pn)
            # T_flat (nm, mp*pn) -> We need to be careful with dimensions.
            # T_flat is (nm, D) where D = mp*pn.
            # The term for V is: sum_r (V[:,r] * (U[:,r] outer W[:,r])_resized)
            # This is tricky. Let's use the property:
            # T_reshaped (nm, mp, pn)
            # T_reshaped = U * (V * W^T) is not standard.
            # Let's use: T_reshaped = np.einsum('ir,jr,kr->ijk', U, V, W)
            # We want to solve for V.
            # Reshape T_reshaped to (nm*pn, mp).
            # T_mat (nm*pn, mp)
            # T_mat[a, j] = sum_r U[i,r] W[k,r] V[j,r]
            # Let M = U * W^T ? No.
            # Let's form the matrix C (nm*pn, R) where C[a, r] = U[i,r] * W[k,r]
            # Then T_mat = C * V^T ?
            # T_mat[a, j] = sum_r C[a, r] V[j, r] = (C * V^T)[a, j]
            # So T_mat = C * V^T
            # V^T = T_mat * C^+
            # V = (T_mat * C^+)^T
            
            # Construct C
            C = np.zeros((n*m * p*n, rank))
            for r in range(rank):
                C[:, r] = np.outer(U[:,r], W[:,r]).flatten()
            
            T_mat = T.reshape(n*m * p*n, m*p)
            
            try:
                # Solve C * V^T = T_mat
                # V^T = T_mat * C^+
                V_T, residuals, rank_v, s = np.linalg.lstsq(C, T_mat, rcond=None)
                V = V_T.T
            except np.linalg.LinAlgError:
                break

            # Update W
            # Similar to V, but swapping roles.
            # T_mat (nm*pn, mp) = (U * V^T) * W^T ?
            # T_mat[a, j] = sum_r U[i,r] V[j,r] W[k,r]
            # Let D (nm*pn, R) where D[a, r] = U[i,r] * V[j,r] ? No, j is fixed in T_mat columns?
            # T_mat is (nm*pn, mp).
            # Let's reshape T to (mp, nm*pn) for W update?
            # T_mat2 (mp, nm*pn)
            # T_mat2[j, a] = sum_r V[j,r] U[i,r] W[k,r]
            # Let E (mp, R) where E[j, r] = V[j,r]
            # Let F (R, nm*pn) where F[r, a] = U[i,r] * W[k,r] ? No.
            # T_mat2 = V * (U * W^T) ?
            # Let's use: T_mat2 = V * G where G (R, nm*pn)
            # G[r, a] = U[i,r] * W[k,r]
            # Then T_mat2[j, a] = sum_r V[j,r] G[r, a]
            # So T_mat2 = V * G
            # We want W.
            # G depends on W. This is not linear in W directly if we fix U,V.
            # Wait, G[r, a] = U[i,r] * W[k,r].
            # T_mat2[j, a] = sum_r V[j,r] U[i,r] W[k,r]
            # Let H (mp, R) where H[j, r] = V[j,r] * U[i,r] ? No, i varies with a.
            # Let's go back to C for W.
            # T_mat (nm*pn, mp)
            # T_mat[a, j] = sum_r (U[i,r] * V[j,r]) * W[k,r]
            # Let Q (nm*pn, R) where Q[a, r] = U[i,r] * V[j,r]
            # Then T_mat = Q * W^T
            # W^T = T_mat * Q^+
            
            Q = np.zeros((n*m * p*n, rank))
            for r in range(rank):
                # a indexes (i,k). r is rank.
                # We need U[i,r] * V[j,r] for all i,k,j?
                # Q[a, r] = U[i,r] * V[j,r]
                # But j is not in a. a is (i,k).
                # So Q cannot be formed simply as outer product of U and V.
                # T_mat[a, j] = sum_r U[i,r] V[j,r] W[k,r]
                # This is linear in W.
                # Let's fix a (i,k) and j.
                # T_mat[i*m*p + k, j] = sum_r (U[i,r] V[j,r]) W[k,r]
                # Let R_mat (nm*pn, R) where R_mat[a, r] = U[i,r] * V[j,r] ?
                # No, j is the column index of T_mat.
                # We want to solve for W (pn x R).
                # T_mat (nm*pn, mp)
                # Let's reshape T to (mp, nm, pn) -> T3
                # T3[j, i, k] = sum_r V[j,r] U[i,r] W[k,r]
                # T3 = V * (U * W^T) ?
                # Let's use lstsq on W.
                # T3 (mp, nm*pn)
                # T3[j, a] = sum_r V[j,r] * (U[i,r] * W[k,r])
                # Let M (mp, R) where M[j, r] = V[j,r]
                # Let N (R, nm*pn) where N[r, a] = U[i,r] * W[k,r]
                # T3 = M * N
                # N depends on W.
                # N[r, a] = U[i,r] * W[k,r]
                # This is not linear in W if we treat N as variable?
                # Actually, T3[j, a] = sum_r V[j,r] U[i,r] W[k,r]
                # Let's define S (mp, R) where S[j, r] = V[j,r]
                # Let's define T (R, nm*pn) where T[r, a] = U[i,r] * W[k,r]
                # This is not right.
                
                # Let's use the same trick as V.
                # T_mat (nm*pn, mp)
                # T_mat[a, j] = sum_r (U[i,r] * V[j,r]) * W[k,r]
                # Let P (nm*pn, R) where P[a, r] = U[i,r] * V[j,r]
                # But j is not in a.
                # We need to be careful.
                # Let's flatten T to (mp, nm*pn) for W.
                # T_W (mp, nm*pn)
                # T_W[j, a] = sum_r V[j,r] U[i,r] W[k,r]
                # Let A_W (mp, R) where A_W[j, r] = V[j,r]
                # Let B_W (R, nm*pn) where B_W[r, a] = U[i,r] * W[k,r]
                # T_W = A_W * B_W
                # B_W depends on W.
                # B_W[r, a] = U[i,r] * W[k,r]
                # This is linear in W?
                # B_W[r, a] = U[i,r] * W[k,r]
                # T_W[j, a] = sum_r A_W[j, r] * U[i,r] * W[k,r]
                # Let C_W (mp, R) where C_W[j, r] = A_W[j, r] * U[i,r] ? No, i depends on a.
                
                # Let's use the matrix form:
                # T_W = V * (U \otimes I) * W^T ? No.
                
                # Let's just use the residual and gradient for W if ALS is too complex?
                # Or use the fact that the problem is small.
                # We can solve for W by fixing U, V and treating the problem as:
                # Minimize || T - sum U_r outer V_r outer W_r ||
                # This is linear in W.
                # Let's construct the system for W.
                # T_flat (nm, mp*pn)
                # T_flat[i, j*pn + k] = sum_r U[i,r] V[j,r] W[k,r]
                # Let's group by k.
                # For each k, we have a vector of size nm*mp.
                # T_k (nm*mp) = sum_r (U * V^T)_{:,r} * W[k,r] ?
                # Let M (nm*mp, R) where M[a, r] = U[i,r] * V[j,r]
                # Then T_k = M * W[k, :]^T
                # So T_stack (pn, nm*mp) = W * M^T
                # W = T_stack * (M^T)^+
                
                M = np.zeros((n*m * m*p, rank))
                for r in range(rank):
                    M[:, r] = np.outer(U[:,r], V[:,r]).flatten()
                
                T_stack = T.reshape(p*n, n*m * m*p).T # (nm*mp, pn)
                
                try:
                    # Solve W * M^T = T_stack
                    # W = T_stack * (M^T)^+
                    W_T, residuals, rank_w, s = np.linalg.lstsq(M.T, T_stack, rcond=None)
                    W = W_T.T
                except np.linalg.LinAlgError:
                    break

            # Check convergence
            T_recon = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            if np.max(np.abs(T_recon - T)) < 1e-8:
                if verify_decomposition(U, V, W, tol=1e-6):
                    return U, V, W
                break
        
        # Final check
        if verify_decomposition(U, V, W, tol=1e-6):
            return U, V, W
        return None

    best_result = to_lists(U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Try various ranks with multiple seeds
    for rank in range(35, 19, -1):
        if time.time() > deadline - 10:
            break
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            result = try_rank(rank, seed=seed)
            if result is not None:
                U, V, W = result
                U, V, W = polish(U, V, W)
                if verify_decomposition(U, V, W, tol=1e-5):
                    best_rank = rank
                    best_result = to_lists(U, V, W)
                    break  # Found this rank, try lower
    
    return best_result
# EVOLVE-BLOCK-END
