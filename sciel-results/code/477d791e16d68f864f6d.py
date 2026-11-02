import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # Build the target tensor
    nm, mp, pn = n * m, m * p, p * n
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T_target."""
        U_arr = np.asarray(U)
        V_arr = np.asarray(V)
        W_arr = np.asarray(W)
        T_full = np.einsum('ur,vr,wr->uvw', U_arr, V_arr, W_arr)
        return np.allclose(T_full, T_target, atol=tol)

    def check_half_integer(U, V, W, tol=1e-9):
        """Check if entries are close to half-integers (k/2 for integer k)."""
        for M in [np.asarray(U), np.asarray(V), np.asarray(W)]:
            doubled = 2.0 * M
            if not np.allclose(doubled, np.round(doubled), atol=tol):
                return False
        return True

    def project_to_half_integer(U, V, W):
        """Project each entry to nearest half-integer."""
        U_h = np.round(2.0 * np.asarray(U)) / 2.0
        V_h = np.round(2.0 * np.asarray(V)) / 2.0
        W_h = np.round(2.0 * np.asarray(W)) / 2.0
        return U_h, V_h, W_h

    def try_ho_svd_peeling(max_rank, max_iter=200, n_restarts=3):
        """
        Use HO SVD-inspired rank-1 peeling with multiple restarts.
        """
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 10:
                return None

            residual = T_target.copy()
            U_list = []
            V_list = []
            W_list = []

            np.random.seed(42 + restart * 1000)

            for rank_idx in range(max_rank):
                if time.time() - start_time > time_limit - 10:
                    return None

                # Initialize u, v, w from residual slices with some randomness
                u = np.random.randn(nm) * 0.1 + np.mean(residual, axis=(1, 2))
                v = np.random.randn(mp) * 0.1 + np.mean(residual, axis=(0, 2))
                w = np.random.randn(pn) * 0.1 + np.mean(residual, axis=(0, 1))

                # Alternating maximization to find best rank-1 approximation
                best_alpha = 0.0
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()

                for it in range(max_iter):
                    # Update u
                    S_u = np.einsum('ijk,j,k->i', residual, v, w)
                    u = S_u / (np.linalg.norm(S_u) + 1e-15)

                    # Update v
                    S_v = np.einsum('ijk,i,k->j', residual, u, w)
                    v = S_v / (np.linalg.norm(S_v) + 1e-15)

                    # Update w
                    S_w = np.einsum('ijk,i,j->k', residual, u, v)
                    w = S_w / (np.linalg.norm(S_w) + 1e-15)

                    alpha = np.einsum('ijk,i,j,k', residual, u, v, w)
                    if abs(alpha) > abs(best_alpha):
                        best_alpha = alpha
                        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()

                    # Early stopping if converged
                    if abs(alpha) < 1e-12:
                        break

                if abs(best_alpha) < 1e-10:
                    break

                # Subtract the rank-1 component
                residual = residual - best_alpha * np.einsum('i,j,k->ijk', best_u, best_v, best_w)

                U_list.append(best_alpha * best_u)
                V_list.append(best_v)
                W_list.append(best_w)

                # Check if residual is essentially zero
                if np.linalg.norm(residual) < 1e-9:
                    break

            # Form the matrices
            R = len(U_list)
            if R == 0:
                continue

            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)

            # Try direct verification
            if check_decomposition(U, V, W, tol=1e-6):
                # Try to project to half-integers
                U_h, V_h, W_h = project_to_half_integer(U, V, W)
                if check_decomposition(U_h, V_h, W_h, tol=1e-9):
                    return {"U": U_h.tolist(), "V": V_h.tolist(), "W": W_h.tolist()}
                # If projection doesn't work, return as-is if valid
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    def try_recursive_block_partition(max_rank, max_iter=150, n_restarts=5):
        """
        Recursive block partitioning strategy.
        Treats the multiplication tensor as a sum of block interactions.
        For n=2, m=3, p=7, we can partition the index spaces and solve
        sub-problems, combining solutions.
        
        This exploits the additive structure: if we can decompose the tensor
        into blocks where each block has a known low-rank decomposition,
        the total rank is the sum of block ranks.
        
        We try to find a partition of the (i, j, k) space into blocks
        such that each block's restriction of T_target has rank < size.
        """
        if time.time() - start_time > time_limit - 15:
            return None

        # The full tensor has support on (i, j, k) for i in [0,2), j in [0,3), k in [0,7).
        # Total 42 non-zero entries.
        
        # Strategy: Try to partition the k-index (size 7) into groups,
        # and for each group, find a low-rank decomposition of the sub-tensor.
        # Then combine.
        
        # However, a more effective approach for this specific problem:
        # Use the fact that the tensor is sparse and has a specific structure.
        # We can try to find a set of "basis" tensors that span the space.
        
        # Let's try a different angle: Randomized SVD on the flattened tensor.
        # Flatten T_target to a matrix of shape (nm, mp*pn).
        # The rank of this matrix is an upper bound on the tensor rank.
        # Actually, the tensor rank is >= max(matricization rank).
        
        # Let's compute the matricization ranks.
        T_2d_1 = T_target.reshape(nm, mp * pn)
        T_2d_2 = T_target.reshape(mp, nm * pn)
        T_2d_3 = T_target.reshape(pn, nm * mp)
        
        r1 = np.linalg.matrix_rank(T_2d_1)
        r2 = np.linalg.matrix_rank(T_2d_2)
        r3 = np.linalg.matrix_rank(T_2d_3)
        
        lower_bound = max(r1, r2, r3)
        
        # If lower bound is close to schoolbook, we might not gain much.
        # For 2x3x7, let's see what the matricization ranks are.
        # T_target has 1 at (i*m+j, j*p+k, k*n+i).
        
        # Let's try to find a decomposition of rank R starting from lower_bound.
        # We will use a local search / optimization approach.
        
        # We will try to optimize U, V, W to minimize ||T - sum u_r v_r w_r||_F
        # using gradient descent or alternating least squares.
        
        # Given time constraints, we will stick to the peeling method but
        # initialize it with the structure of the matricizations.
        
        # For now, this function acts as a wrapper that calls try_ho_svd_peeling
        # with parameters tuned based on the matricization ranks.
        
        target_rank = min(max_rank, lower_bound + 2)
        if target_rank < 40:
             target_rank = 40 # Don't go too low, it's unlikely
        
        return try_ho_svd_peeling(target_rank, max_iter=max_iter, n_restarts=n_restarts)

    # Try ranks from 41 down to 35
    # Theoretical lower bounds suggest rank is at least max(nm, mp, pn) = 21
    # and at most nm*mp*pn / max(nm, mp, pn) but that's not tight.
    # Known results: for 2x3 x 3x7, the rank is likely 42 (schoolbook) or slightly less.
    
    # First, try to verify if schoolbook can be improved
    # Try HO SVD peeling for rank 41
    result = try_ho_svd_peeling(41, max_iter=200, n_restarts=10)
    if result is not None:
        return result

    # Try rank 40
    if time.time() - start_time < time_limit - 20:
        result = try_ho_svd_peeling(40, max_iter=150, n_restarts=5)
        if result is not None:
            return result

    # Try rank 39
    if time.time() - start_time < time_limit - 15:
        result = try_ho_svd_peeling(39, max_iter=100, n_restarts=3)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()
