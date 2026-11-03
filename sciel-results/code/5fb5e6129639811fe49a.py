import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0


def _remaining():
    return _DEADLINE - time.time()


def _build_rank36(n=2, m=4, p=5):
    """Known rank-36 decomposition: 4x Strassen 2x2x2 blocks + rank-8 schoolbook tail."""
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        c = col
        # M1 = (A00+A11)*(B00+B11)
        U[i0*m+j0][c] += 1; U[(i0+1)*m+(j0+1)][c] += 1
        V[j0*p+k0][c] += 1; V[(j0+1)*p+(k0+1)][c] += 1
        W[k0*n+i0][c] += 1; W[(k0+1)*n+(i0+1)][c] += 1
        c += 1
        # M2 = (A10+A11)*B00
        U[(i0+1)*m+j0][c] += 1; U[(i0+1)*m+(j0+1)][c] += 1
        V[j0*p+k0][c] += 1
        W[k0*n+(i0+1)][c] += 1; W[(k0+1)*n+(i0+1)][c] -= 1
        c += 1
        # M3 = A00*(B01-B11)
        U[i0*m+j0][c] += 1
        V[j0*p+(k0+1)][c] += 1; V[(j0+1)*p+(k0+1)][c] -= 1
        W[(k0+1)*n+i0][c] += 1; W[(k0+1)*n+(i0+1)][c] += 1
        c += 1
        # M4 = A11*(B10-B00)
        U[(i0+1)*m+(j0+1)][c] += 1
        V[(j0+1)*p+k0][c] += 1; V[j0*p+k0][c] -= 1
        W[k0*n+i0][c] += 1; W[k0*n+(i0+1)][c] += 1
        c += 1
        # M5 = (A00+A01)*B11
        U[i0*m+j0][c] += 1; U[i0*m+(j0+1)][c] += 1
        V[(j0+1)*p+(k0+1)][c] += 1
        W[k0*n+i0][c] -= 1; W[(k0+1)*n+i0][c] += 1
        c += 1
        # M6 = (A10-A00)*(B00+B01)
        U[(i0+1)*m+j0][c] += 1; U[i0*m+j0][c] -= 1
        V[j0*p+k0][c] += 1; V[j0*p+(k0+1)][c] += 1
        W[(k0+1)*n+(i0+1)][c] += 1
        c += 1
        # M7 = (A01-A11)*(B10+B11)
        U[i0*m+(j0+1)][c] += 1; U[(i0+1)*m+(j0+1)][c] -= 1
        V[(j0+1)*p+k0][c] += 1; V[(j0+1)*p+(k0+1)][c] += 1
        W[k0*n+i0][c] += 1
        c += 1
        return c

    def add_schoolbook_2x4x1(i0, j0, k0):
        nonlocal col
        c = col
        for di in range(2):
            for dj in range(4):
                U[(i0+di)*m+(j0+dj)][c] = 1
                V[(j0+dj)*p+k0][c] = 1
                W[k0*n+(i0+di)][c] = 1
                c += 1
        return c

    col = add_strassen(0, 0, 0)
    col = add_strassen(0, 2, 0)
    col = add_strassen(0, 0, 2)
    col = add_strassen(0, 2, 2)
    col = add_schoolbook_2x4x1(0, 0, 4)
    assert col == rank, f"Expected {rank}, got {col}"
    return U, V, W


def _verify(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check that the factors exactly reconstruct the matmul tensor T[i,j,k]=1."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reconstruct T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Build by summing over rank: T = sum_r u_r (n*m) x v_r (m*p) x w_r (p*n)
    # We need T[i,j,k] for all i in [0,n), j in [0,m), k in [0,p).
    # For each r, u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # Sum over r should equal 1 for all (i,j,k).
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]   # length n*m
        v = V[:, r]   # length m*p
        w = W[:, r]   # length p*n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True


def _reduce_rank(U, V, W, n=2, m=4, p=5):
    """Try to reduce rank by finding linear dependencies among the rank-1 terms."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    nm, m_p, p_n, rank = U.shape[0], V.shape[0], W.shape[0], U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Check for zero columns (trivially removable)
    for r in range(rank - 1, -1, -1):
        if _remaining() < 5.0:
            break
        if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
            if rank < best_rank:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = rank

    if rank < best_rank:
        U, V, W, rank = best_U, best_V, best_W, best_rank

    # Try greedy removal: for each column r, check if T can be reconstructed without it
    # by solving a linear system for the other columns' contributions.
    # This is expensive, so only try if we have time.
    if _remaining() < 20.0:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

    # For each pair of columns (r, s), check if one can be expressed as a linear combination
    # of the others in a way that preserves T. This is equivalent to checking if the
    # rank-1 tensor u_r ⊗ v_r ⊗ w_r is in the span of the others.
    # A cheaper heuristic: check if removing column r and re-solving gives a valid solution.
    # We'll try removing one column at a time and checking if the residual is rank-1.
    for r in range(rank - 1, -1, -1):
        if _remaining() < 10.0:
            break
        # Remove column r
        U_r = np.delete(U, r, axis=1)
        V_r = np.delete(V, r, axis=1)
        W_r = np.delete(W, r, axis=1)
        rank_r = rank - 1
        # Compute residual: T - sum_{s != r} u_s ⊗ v_s ⊗ w_s
        # Check if residual is a rank-1 tensor (i.e., equals u ⊗ v ⊗ w for some u,v,w)
        # A tensor is rank-1 iff all its 2x2 minors are zero (for the appropriate flattening).
        # Flatten T as (n*m) x (m*p) matrix: T_flat[i*m+j, j*p+k] ... this doesn't work directly.
        # Instead, use the unfolding: for fixed i, T[i,:,:] should be v ⊗ w scaled.
        # Compute T_r = sum_{s != r} u_s ⊗ v_s ⊗ w_s
        T_r = np.zeros((n, m, p))
        for s in range(rank_r):
            u_s = U_r[:, s]
            v_s = V_r[:, s]
            w_s = W_r[:, s]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_r[i, j, k] += u_s[i*m+j] * v_s[j*p+k] * w_s[k*n+i]
        residual = np.ones((n, m, p)) - T_r
        # Check if residual is rank-1: residual[i,j,k] = a[i]*b[j]*c[k]
        # Try to find a, b, c. Set a[0,0,0] = residual[0,0,0] (if nonzero).
        # Actually, let's use SVD on unfoldings.
        # Unfold 1: reshape to (n, m*p) -> matrix R1 where R1[i, j*p+k] = residual[i,j,k]
        R1 = residual.reshape(n, m*p)
        # If residual is rank-1, R1 should have rank 1.
        sv = np.linalg.svd(R1, compute_uv=False)
        if len(sv) > 0 and sv[0] > 1e-12:
            if len(sv) > 1 and sv[1] > 1e-12 * sv[0]:
                continue  # Not rank-1
        # Check consistency with other unfoldings
        R2 = residual.reshape(m, n*p)  # R2[j, i*p+k] ... need to be careful
        # Actually let's just verify by trying to reconstruct
        if sv[0] < 1e-12:
            continue  # Residual is zero, so we can remove column r!
        # Try to extract the rank-1 factors
        U_new = np.linalg.svd(R1, compute_uv=True)[0]
        u_vec = U_new[:, 0] * sv[0]
        # residual[i, j*p+k] = u_vec[i] * v_vec[j*p+k]
        # v_vec should be consistent: residual[0, j*p+k] / u_vec[0] = v_vec[j*p+k]
        if abs(u_vec[0]) < 1e-12:
            continue
        v_vec = residual[0, :] / u_vec[0]
        # Verify: residual[i, j*p+k] == u_vec[i] * v_vec[j*p+k]
        recon = np.outer(u_vec, v_vec).reshape(n, m, p)
        if not np.allclose(recon, residual, atol=1e-9):
            continue
        # Now we need v_vec[j*p+k] = b[j] * c[k] for some b, c
        # v_vec has shape (m*p), reshape to (m, p)
        vp = v_vec.reshape(m, p)
        sv2 = np.linalg.svd(vp, compute_uv=False)
        if len(sv2) > 1 and sv2[1] > 1e-12 * (sv2[0] if sv2[0] > 0 else 1.0):
            continue  # v_vec is not separable
        # Extract b and c
        U2, S2, V2t = np.linalg.svd(vp, compute_uv=True)
        b_vec = U2[:, 0] * S2[0]
        c_vec = V2t[0, :]
        # Verify: vp[j,k] = b_vec[j] * c_vec[k]
        if not np.allclose(np.outer(b_vec, c_vec), vp, atol=1e-9):
            continue
        # Now we have: residual[i,j,k] = u_vec[i*m+j]... wait, u_vec is indexed by i only.
        # We need u_full[i*m+j] such that u_full[i*m+j] * b_vec[j] * c_vec[k] = residual[i,j,k]
        # residual[i,j,k] = u_vec[i] * v_vec[j*p+k] = u_vec[i] * b_vec[j] * c_vec[k]
        # So u_full[i*m+j] = u_vec[i] (independent of j)? Let's check.
        # u_full[i*m+j] * b_vec[j] * c_vec[k] = u_vec[i] * b_vec[j] * c_vec[k]
        # So u_full[i*m+j] = u_vec[i] for all j.
        u_full = np.zeros(n*m)
        for i in range(n):
            for j in range(m):
                u_full[i*m+j] = u_vec[i]
        # Now add this new rank-1 term to the existing ones
        U_new = np.column_stack([U_r, u_full])
        V_new = np.column_stack([V_r, b_vec])
        W_new = np.column_stack([W_r, c_vec])
        # Verify the full reconstruction
        if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            U, V, W = U_new, V_new, W_new
            rank = rank_r + 1  # Removed one, added one, net same? No: removed r, added 1 new
            # Actually rank stays the same: we replaced column r with a new column.
            # Wait, we removed column r (rank-1 terms) and added 1 new rank-1 term.
            # So rank is the same. This doesn't help.
            # The point is: if residual is rank-1, we can replace column r with the new term.
            # But that doesn't reduce rank. We need to check if the new term is redundant
            # with existing terms, which would reduce rank.
            # For now, this approach doesn't reduce rank. Skip.
            pass
        else:
            continue

    return U.tolist(), V.tolist(), W.tolist(), rank


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.
    """
    n, m, p = 2, 4, 5

    # Start with known rank-36 solution
    U, V, W = _build_rank36(n, m, p)

    # Verify it works
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but fallback
        raise RuntimeError("Rank-36 solution failed verification")

    # Try to reduce rank
    if _remaining() > 15.0:
        U, V, W, rank = _reduce_rank(U, V, W, n, m, p)
        if rank < 36:
            # Verify the reduced solution
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Return the rank-36 solution
    return {"U": U, "V": V, "W": W}
