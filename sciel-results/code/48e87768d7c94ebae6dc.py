"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 7
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14

    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])

        if np.max(np.abs(R - target)) > tol:
            return False

        # Check integer/half-integer vectorized
        for mat in [U, V, W]:
            if not np.all(np.abs(mat * 2 - np.round(mat * 2)) < tol):
                return False
        return True

    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def compute_R(U, V, W):
        """Compute the reconstructed tensor."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return R

    def tensor_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        R = compute_R(U, V, W)
        return np.sum((R - target) ** 2)

    def als_optimize(U, V, W, max_iters=80):
        """
        Optimize U, V, W using Alternating Least Squares with time guard.
        """
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()

        prev_error = tensor_error(U, V, W)

        for it in range(max_iters):
            if time.time() > deadline - 8:
                break

            # Update U: for each row a, solve (V*W^T) @ U[a,:] = target[a,:,:].flatten()
            # M_U[b,c] = sum_r V[b,r]*W[c,r] -> shape (mp*pn, r)
            M_U = np.zeros((mp * pn, r))
            for c in range(r):
                M_U[:, c] = np.outer(V[:, c], W[:, c]).flatten()

            for a in range(nm):
                rhs = target[a, :, :].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                except:
                    U[a, :] = 0.0

            # Update V
            M_V = np.zeros((nm * pn, r))
            for c in range(r):
                M_V[:, c] = np.outer(U[:, c], W[:, c]).flatten()

            for b in range(mp):
                rhs = target[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                except:
                    V[b, :] = 0.0

            # Update W
            M_W = np.zeros((nm * mp, r))
            for c in range(r):
                M_W[:, c] = np.outer(U[:, c], V[:, c]).flatten()

            for c in range(pn):
                rhs = target[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                except:
                    W[c, :] = 0.0

            current_error = tensor_error(U, V, W)
            if abs(current_error - prev_error) < 1e-12:
                break
            prev_error = current_error

        return U, V, W, prev_error

    def polish_pass(U, V, W, max_iters=200):
        """
        High-precision polish: alternate between rounding to half-integers
        and doing a tight local solve. This snaps the solution onto the lattice.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Round to half-integers
            U_r = round_to_half_integer(U)
            V_r = round_to_half_integer(V)
            W_r = round_to_half_integer(W)

            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r

            # Use rounded values as starting point for a tight local solve
            # Fix W, solve for U and V via least squares with tight tolerance
            M_U = np.zeros((mp * pn, r))
            for c in range(r):
                M_U[:, c] = np.outer(V_r[:, c], W_r[:, c]).flatten()

            for a in range(nm):
                rhs = target[a, :, :].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                except:
                    pass

            M_V = np.zeros((nm * pn, r))
            for c in range(r):
                M_V[:, c] = np.outer(U_r[:, c], W_r[:, c]).flatten()

            for b in range(mp):
                rhs = target[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                except:
                    pass

            M_W = np.zeros((nm * mp, r))
            for c in range(r):
                M_W[:, c] = np.outer(U_r[:, c], V_r[:, c]).flatten()

            for c in range(pn):
                rhs = target[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                except:
                    pass

        return U, V, W, tensor_error(U, V, W)

    def structured_init(rank):
        """
        Create a structured initialization based on the tensor's support.
        The tensor has 1s at (i*m+j, j*p+k, k*n+i) for i,j,k.
        We initialize each rank-1 component to cover a small subset of these.
        """
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        # List of all (a, b, c) positions where target is 1
        positions = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    positions.append((i * m + j, j * p + k, k * n + i))

        # Assign each position to a rank component (round-robin)
        for idx, (a, b, c) in enumerate(positions):
            comp = idx % rank
            # Add small random perturbation to avoid degenerate solutions
            U[a, comp] += 1.0 + 0.1 * np.random.randn()
            V[b, comp] += 1.0 + 0.1 * np.random.randn()
            W[c, comp] += 1.0 + 0.1 * np.random.randn()

        return U, V, W

    def random_sparse_init(rank):
        """Random sparse initialization with structure-aware sparsity."""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        for c in range(rank):
            # Each component touches a few entries
            nz_u = np.random.choice(nm, size=np.random.randint(1, 4), replace=False)
            nz_v = np.random.choice(mp, size=np.random.randint(1, 5), replace=False)
            nz_w = np.random.choice(pn, size=np.random.randint(1, 4), replace=False)
            U[nz_u, c] = np.random.choice([-1, 1], size=len(nz_u)) * 0.5
            V[nz_v, c] = np.random.choice([-1, 1], size=len(nz_v)) * 0.5
            W[nz_w, c] = np.random.choice([-1, 1], size=len(nz_w)) * 0.5

        return U, V, W

    def try_rank(rank, num_attempts=10):
        """Try to find a valid decomposition for a given rank."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None

        for attempt in range(num_attempts):
            if time.time() > deadline - 12:
                break

            # Alternate between structured and random initialization
            if attempt % 3 == 0:
                U, V, W = structured_init(rank)
            elif attempt % 3 == 1:
                U, V, W = random_sparse_init(rank)
            else:
                # Dense random with small scale
                U = np.random.randn(nm, rank) * 0.3
                V = np.random.randn(mp, rank) * 0.3
                W = np.random.randn(pn, rank) * 0.3

            U_opt, V_opt, W_opt, final_error = als_optimize(U, V, W, max_iters=60)

            if final_error < best_error:
                best_error = final_error
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()

            # Try polishing
            if final_error < 1.0:  # Only polish if we're getting close
                U_pol, V_pol, W_pol = polish_pass(U_opt, V_opt, W_opt, max_iters=50)
                if verify(U_pol, V_pol, W_pol):
                    return U_pol, V_pol, W_pol

            # Also try rounding the ALS result directly
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)

            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded

        # Final polish on best result
        if best_U is not None and best_error < 5.0:
            U_pol, V_pol, W_pol = polish_pass(best_U, best_V, best_W, max_iters=100)
            if verify(U_pol, V_pol, W_pol):
                return U_pol, V_pol, W_pol

        return None, None, None

    best_result = None
    best_rank = 42

    # Try ranks from low to high, with more attempts for lower ranks
    for rank in range(24, 42):
        if time.time() > deadline - 30:
            break

        # More attempts for promising lower ranks
        num_attempts = 8 if rank <= 30 else 5
        U, V, W = try_rank(rank, num_attempts=num_attempts)

        if U is not None:
            best_result = (U, V, W)
            best_rank = rank
            break

    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
