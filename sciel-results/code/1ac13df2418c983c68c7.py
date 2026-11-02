"""Tensor-rank search for 3x3 * 3x5 matrix multiplication.

Approach family: tensor network / CP-decomposition rank minimization.
We build the multiplication tensor T with T[(i,j),(j',k),(k',i')] = 1 iff i=i', j=j', k=k',
then search for a CP decomposition of minimal rank with half-integer entries via
alternating least squares + local rounding, verifying exactness before reporting.
"""
import numpy as np

DIMENSIONS = [3, 3, 5]


def _build_tensor():
    n, m, p = DIMENSIONS
    a_dim, b_dim, c_dim = n * m, m * p, p * n
    T = np.zeros((a_dim, b_dim, c_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T, a_dim, b_dim, c_dim


def _round_to_half(x):
    return np.round(2.0 * x) / 2.0


def _verify(T, U, V, W, tol=1e-9):
    r = U.shape[1]
    for c in range(r):
        T -= np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(T)) <= tol


def _als_cp(T, rank, iters=800, tol=1e-10):
    """Alternating least squares for CP decomposition of a 3-tensor."""
    a_dim, b_dim, c_dim = T.shape
    rng = np.random.default_rng(12345)
    U = rng.standard_normal((a_dim, rank))
    V = rng.standard_normal((b_dim, rank))
    W = rng.standard_normal((c_dim, rank))
    # Normalize columns to reduce scaling ambiguity
    for c in range(rank):
        nu = np.linalg.norm(U[:, c])
        nv = np.linalg.norm(V[:, c])
        nw = np.linalg.norm(W[:, c])
        s = (nu * nv * nw) ** (1.0 / 3.0)
        if s > 1e-12:
            U[:, c] /= s
            V[:, c] /= s
            W[:, c] /= s

    prev_err = np.inf
    for _ in range(iters):
        # Update U
        for c in range(rank):
            # U[:,c] = argmin sum_{a} (T[a,:,:] - sum_c' U[a,c'] V[c',:] W[:,c']^T)
            # => U[:,c] = (T x (V[c,:] * W[c,:])) / sum_a ...
            # Standard ALS: fix V,W, solve for each column
            pass
        # Simpler: use standard CP-ALS via mode products
        # U = T x2 (V * W^T) ... using Khatri-Rao
        Vkr = np.einsum('bc,dc->bd', V, W)  # (b_dim, rank)
        U_new = np.einsum('abc,dc->ad', T, Vkr)
        # Normalize scale: divide by norm of each column to keep stable
        for c in range(rank):
            nrm = np.linalg.norm(U_new[:, c])
            if nrm > 1e-12:
                U_new[:, c] /= nrm
        U = U_new

        # Update V
        Wkr = np.einsum('ac,dc->ad', W, U)
        V_new = np.einsum('abc,ac->bc', T, Wkr)
        for c in range(rank):
            nrm = np.linalg.norm(V_new[:, c])
            if nrm > 1e-12:
                V_new[:, c] /= nrm
        V = V_new

        # Update W
        Ukr = np.einsum('ac,bc->ab', U, V)
        W_new = np.einsum('abc,ab->ac', T, Ukr)
        for c in range(rank):
            nrm = np.linalg.norm(W_new[:, c])
            if nrm > 1e-12:
                W_new[:, c] /= nrm
        W = W_new

        err = np.max(np.abs(T - np.einsum('ac,bc,dc->abc', U, V, W)))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W


def _refine_rounded(U, V, W):
    """Round to half-integers and try small local perturbations to fix exactness."""
    Ur = _round_to_half(U)
    Vr = _round_to_half(V)
    Wr = _round_to_half(W)
    # Try small flips: for each column, try flipping one element by +/- 0.5
    best = (Ur, Vr, Wr)
    best_err = np.inf
    for name, mat in (('U', Ur), ('V', Vr), ('W', Wr)):
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                for delta in (-0.5, +0.5):
                    trial = mat.copy()
                    trial[r, c] = mat[r, c] + delta
                    if name == 'U':
                        err = np.max(np.abs(np.einsum('ac,bc,dc->abc', trial, Vr, Wr)
                                            - np.einsum('ac,bc,dc->abc', U, V, W)))
                    elif name == 'V':
                        err = np.max(np.abs(np.einsum('ac,bc,dc->abc', Ur, trial, Wr)
                                            - np.einsum('ac,bc,dc->abc', U, V, W)))
                    else:
                        err = np.max(np.abs(np.einsum('ac,bc,dc->abc', Ur, Vr, trial)
                                            - np.einsum('ac,bc,dc->abc', U, V, W)))
                    if err < best_err:
                        best_err = err
                        if name == 'U':
                            best = (trial, Vr, Wr)
                        elif name == 'V':
                            best = (Ur, trial, Wr)
                        else:
                            best = (Ur, Vr, trial)
    return best


def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 45

    T, a_dim, b_dim, c_dim = _build_tensor()

    best = None
    best_rank = rank_school

    # Search ranks from below schoolbook down to a floor.
    for rank in range(20, rank_school):
        # Try multiple random inits
        found = False
        for trial in range(4):
            U, V, W = _als_cp(T, rank)
            Ur, Vr, Wr = _refine_rounded(U, V, W)
            # exact check
            Tcopy = T.copy()
            if _verify(Tcopy, Ur, Vr, Wr, tol=1e-7):
                # Convert to list of ints/half-ints
                U_list = [[float(x) for x in row] for row in Ur.tolist()]
                V_list = [[float(x) for x in row] for row in Vr.tolist()]
                W_list = [[float(x) for x in row] for row in Wr.tolist()]
                # Clean up -0.0
                for L in (U_list, V_list, W_list):
                    for row in L:
                        for i in range(len(row)):
                            if abs(row[i]) < 1e-12:
                                row[i] = 0.0
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                found = True
                break
        if found:
            break

    if best is not None and best_rank < rank_school:
        return best

    # Fallback: schoolbook (rank 45)
    U = [[0.0] * rank_school for _ in range(n * m)]
    V = [[0.0] * rank_school for _ in range(m * p)]
    W = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
