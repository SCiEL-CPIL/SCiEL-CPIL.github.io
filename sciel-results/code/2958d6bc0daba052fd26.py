"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]
_START_TIME = None

# EVOLVE-BLOCK-START
def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    DEADLINE = 118.0  # safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = n * m * p

    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def flatten_tensor(U, V, W):
        """Given U (nm x r), V (mp x r), W (pn x r), compute the tensor
        tensor[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        and flatten to 1D."""
        # Efficient: use einsum
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # = sum_r (U[i,r] * V[j,r]) * W[k,r]
        # We can compute as: for each r, outer product of U[:,r] and V[:,r], then element-wise with W[:,r]
        # Better: use einsum
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def tensor_to_vec(T3):
        return T3.reshape(-1)

    target_vec = tensor_to_vec(T)
    target_len = target_vec.size  # 90

    def check_exact(U, V, W, tol=1e-9):
        """Check if the decomposition is exact (within tolerance)."""
        T3 = flatten_tensor(U, V, W)
        err = np.max(np.abs(T3 - T))
        return err < tol, err

    def round_to_half_int(M):
        """Round each entry to nearest half-integer (i.e., 2*x rounded to int, then divide by 2)."""
        return np.round(2.0 * M) / 2.0

    def verify_rounded(U_r, V_r, W_r):
        """Verify that the rounded matrices give an exact decomposition."""
        T3 = flatten_tensor(U_r, V_r, W_r)
        # Check exactness: all entries should be 0 or 1 exactly (within tight tol)
        err = np.max(np.abs(T3 - T))
        return err < 1e-9

    def als_rank(r, n_iters=200, n_restarts=3, seed=0):
        """Try to find rank-r decomposition via ALS."""
        best_err = float('inf')
        best_U = best_V = best_W = None
        rng = np.random.RandomState(seed)
        for restart in range(n_restarts):
            # Initialize
            U = rng.randn(nm, r)
            V = rng.randn(mp, r)
            W = rng.randn(pn, r)
            # Normalize to avoid divergence
            for r_idx in range(r):
                norm = 1.0
                U[:, r_idx] /= norm
                V[:, r_idx] /= norm
                W[:, r_idx] /= norm

            for it in range(n_iters):
                # Check time
                if time.time() - _START_TIME > DEADLINE:
                    break

                # Update U: for each i, U[i,:] solves (V^T * W^T * T_i) = U[i,:] * (something)
                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # For fixed i: T[i,:,:] = sum_r U[i,r] (V[:,r] outer W[:,r])
                # Flatten: vec(T[i,:,:]) = sum_r U[i,r] vec(V[:,r] outer W[:,r])
                # = sum_r U[i,r] kron(W[:,r], V[:,r])  (ordering matters)
                # Let's use the einsum approach:
                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # For fixed i: T[i,j,k] * W[k,r] summed over k gives: sum_k T[i,j,k] W[k,r] = sum_r' U[i,r'] V[j,r'] sum_k W[k,r] W[k,r']
                # This is getting complicated. Let's use a simpler ALS approach.

                # Simpler: treat as matrix factorization on each slice
                # T[i,:,:] = sum_r U[i,r] (V[:,r] * W[:,r]^T)  -- element-wise? No, it's outer product.
                # Actually T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # For fixed i: T[i,:,:] has entries T[i,j,k].
                # Let's define M_i as a mp x pn matrix with M_i[j,k] = T[i,j,k]
                # M_i = sum_r U[i,r] * (V[:,r] * W[:,r]^T)  -- element-wise product of V[:,r] and W[:,r]^T? No.
                # V[:,r] is mp x 1, W[:,r]^T is 1 x pn, so their outer product is mp x pn.
                # But the formula is element-wise: V[j,r] * W[k,r], which IS the outer product.
                # So M_i = sum_r U[i,r] * outer(V[:,r], W[:,r])
                # This is a rank-1 update structure.

                # ALS for U:
                # M_i = sum_r U[i,r] * outer(V[:,r], W[:,r])
                # Flatten: vec(M_i) = sum_r U[i,r] * vec(outer(V[:,r], W[:,r]))
                # vec(outer(a,b)) = kron(b, a)  (column-major) or kron(a,b) (row-major)
                # numpy uses C-order (row-major), so vec(outer(V[:,r], W[:,r])) in row-major is:
                # outer is mp x pn, row-major flatten: first row of outer, then second, etc.
                # Row j of outer(V[:,r], W[:,r]) is V[j,r] * W[:,r]
                # So the flattened vector is: for each j, V[j,r] * W[:,r]
                # = kron(V[:,r], W[:,r]) in some sense... let's just compute it directly.

                # Let's build the design matrix for U:
                # D_U is nm x (mp*pn) where D_U[i, j*pn+k] = ... no, that's not right either.
                # Let me think differently.

                # T is nm x mp x pn.
                # For ALS on U: we want to find U[i,:] for each i.
                # T[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r]^T)  where * is element-wise... no, it's outer.
                # Let A_i be the matrix with A_i[r, j*pn+k] = V[j,r] * W[k,r] for r in range(r_rank), j in range(mp), k in range(pn)
                # Then vec(T[i,:,:]) = A_i @ U[i,:]
                # So U[i,:] = pinv(A_i) @ vec(T[i,:,:])

                # But A_i is r x (mp*pn) = r x 540. That's manageable.

                # Let's precompute for each i the design matrix.
                # Actually, the design matrix is the same for all i! Because it only depends on V and W.
                # D[r, j*pn+k] = V[j,r] * W[k,r]
                # This is mp*pn x r matrix (transposed from above).

                # Let's define D as mp*pn x r: D[j*pn+k, r] = V[j,r] * W[k,r]
                # Then vec(T[i,:,:]) = D^T @ U[i,:]  (r x r times r x 1... no)
                # vec(T[i,:,:]) is mp*pn x 1.
                # U[i,:] is r x 1.
                # D^T is r x (mp*pn).
                # So vec(T[i,:,:]) = D^T @ U[i,:] doesn't work dimensionally.
                # It should be: vec(T[i,:,:]) = D @ U[i,:] where D is (mp*pn) x r.
                # D[j*pn+k, r] = V[j,r] * W[k,r]
                # Yes!

                # Build D (mp*pn x r)
                D = np.outer(V.T, W.T).T.reshape(mp*pn, r)  # Hmm, let me be careful
                # V is mp x r, W is pn x r
                # We want D[j*pn+k, r] = V[j,r] * W[k,r]
                # Let's compute it element-wise:
                # V_rep = np.repeat(V, pn, axis=0)  # mp*pn x r
                # W_rep = np.tile(W, mp, axis=0)    # mp*pn x r
                # D = V_rep * W_rep                 # mp*pn x r

                V_rep = np.repeat(V, pn, axis=0)
                W_rep = np.tile(W, mp, axis=0)
                D = V_rep * W_rep  # mp*pn x r

                # Update U:
                # For each i: U[i,:] = pinv(D) @ T[i,:,:].flatten()
                # T[i,:,:].flatten() is mp*pn x 1
                # pinv(D) is r x (mp*pn)
                # U[i,:] is r x 1
                D_pinv = np.linalg.pinv(D)
                for i in range(nm):
                    U[i, :] = D_pinv @ T[i, :, :].flatten()

                # Update V:
                # T[:,j,k] = sum_r U[i,r] V[j,r] W[k,r] for all i,k
                # For fixed j: T[:,j,:] has entries T[i,j,k] for all i,k
                # T[i,j,k] = sum_r (U[i,r] * W[k,r]) * V[j,r]
                # Let E[j*pn+k... no, for fixed j, we have a nm x pn matrix T[:,j,:]
                # vec(T[:,j,:]) = sum_r V[j,r] * vec(U[:,r] * W[:,r]^T)
                # vec(U[:,r] * W[:,r]^T) where U[:,r] is nm x 1, W[:,r]^T is 1 x pn
                # This is the outer product, flattened row-major: for each i, U[i,r]*W[:,r]
                # = np.repeat(U[:, r].reshape(-1,1), pn, axis=1) * np.tile(W[:,r].reshape(1,-1), nm, axis=0)
                # Then flatten.
                # Let's build the design matrix for V:
                # E is (nm*pn) x r, where E[i*pn+k, r] = U[i,r] * W[k,r]
                U_rep = np.repeat(U, pn, axis=0)  # nm*pn x r
                W_rep2 = np.tile(W, nm, axis=0)   # nm*pn x r
                E = U_rep * W_rep2
                E_pinv = np.linalg.pinv(E)
                for j in range(mp):
                    # T[:,j,:] flattened: T[:,j,k] for all i,k
                    # In our array, T is indexed [i, j, k], so T[:, j, :] is a slice
                    V[j, :] = E_pinv @ T[:, j, :].flatten()

                # Update W:
                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # For fixed k: T[:,:,k] is nm x mp
                # vec(T[:,:,k]) = sum_r W[k,r] * vec(U[:,r] * V[:,r]^T)
                # vec(U[:,r] * V[:,r]^T): U[:,r] is nm x 1, V[:,r]^T is 1 x mp
                # F is (nm*mp) x r, where F[i*mp+j, r] = U[i,r] * V[j,r]
                U_rep2 = np.repeat(U, mp, axis=0)  # nm*mp x r
                V_rep2 = np.tile(V, nm, axis=0)    # nm*mp x r
                F = U_rep2 * V_rep2
                F_pinv = np.linalg.pinv(F)
                for k in range(pn):
                    W[k, :] = F_pinv @ T[:, :, k].flatten()

            # After iterations, check error
            err, _ = check_exact(U, V, W)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()

            if best_err < 1e-12:
                break

        return best_err, best_U, best_V, best_W

    # Try ranks from low to high, but start with a reasonable guess
    # Theoretical lower bound: max(nm, mp, pn) = max(15, 30, 18) = 30
    # But we know schoolbook is 90. Let's try to do better.
    # Known results: for 3x5x6, the rank is likely between 40 and 60 or so.
    # Let's try from 40 up to 89.

    best_rank = rank_schoolbook
    best_result = None

    # Schoolbook as fallback
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    # Try ranks
    for r in range(35, min(90, rank_schoolbook)):
        if time.time() - _START_TIME > DEADLINE:
            break

        # Try a few restarts
        found = False
        for attempt in range(2):
            if time.time() - _START_TIME > DEADLINE:
                break
            err, U, V, W = als_rank(r, n_iters=150, n_restarts=2, seed=42 + attempt)
            if err < 1e-9:
                # Round to half-integers
                U_r = round_to_half_int(U)
                V_r = round_to_half_int(V)
                W_r = round_to_half_int(W)
                if verify_rounded(U_r, V_r, W_r):
                    best_rank = r
                    best_result = (U_r, V_r, W_r)
                    found = True
                    break
                else:
                    # Try without rounding first? The problem says integer/half-integer entries.
                    # Maybe we need to round more carefully or the solution is already close to half-integers.
                    pass

        if found:
            break

    if best_result is not None:
        U_r, V_r, W_r = best_result
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        # Convert to lists of lists for the return
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    U_list = U_sb.tolist()
    V_list = V_sb.tolist()
    W_list = W_sb.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
