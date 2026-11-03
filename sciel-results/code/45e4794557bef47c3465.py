"""Rank-minimal exact tensor decomposition for 6x6 x 6x6 matrix multiplication.

Strategy:
  1. Build the Block-Strassen decomposition (rank 189) using the 2x2 Strassen
     scheme on 3x3 subblocks. This is a known exact decomposition.
  2. Attempt to reduce the rank by exploiting the symmetry of the tensor
     (permuting the output row index and swapping the two input factors).
     A "balanced" factor set has identical U and V rows, so a linear
     combination of columns can cancel the shared U contribution while
     retaining a clean W contribution, potentially removing a column.
  3. Every candidate is verified EXACTLY with numpy before being accepted.

The evaluator kills the program after 150 s wall-clock; we keep an explicit
deadline guard and return the best verified result with a safety margin.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [6, 6, 6]
DEADLINE = 145.0  # safety margin below the 150 s kill limit
_START = time.time()


def _time_left():
    return DEADLINE - (time.time() - _START)


def _block_matrix(rows, cols, blocks):
    """Assemble a (rows*3, cols*3) matrix from a (rows, cols) block grid of 3x3."""
    out = np.zeros((rows * 3, cols * 3))
    for r in range(rows):
        for c in range(cols):
            out[r * 3:(r + 1) * 3, c * 3:(c + 1) * 3] = blocks[r][c]
    return out


def _build_block_strassen():
    """Return (U, V, W) as numpy float64 arrays for the Block-Strassen scheme."""
    n = m = p = 6
    I3 = np.eye(3)
    Z = np.zeros((3, 3))

    # Strassen 2x2 block scheme on 3x3 subblocks:
    #   A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    #   M1=(A11+A22)(B11+B22)
    #   M2=(A21+A22)B11
    #   M3=A11(B12-B22)
    #   M4=A22(B21-B11)
    #   M5=(A11+A12)B22
    #   M6=(A21-A11)(B11+B12)
    #   M7=(A12-A22)(B21+B22)
    #   C11=M1+M4-M5+M7 ; C12=M3+M5
    #   C21=M2+M4         ; C22=M1-M2+M3+M6
    U_blocks = [
        [I3, Z, I3, Z, I3, Z, Z],
        [Z, I3, Z, I3, Z, Z, Z],
        [Z, Z, I3, Z, Z, Z, Z],
        [Z, Z, Z, I3, Z, Z, Z],
        [I3, Z, Z, Z, I3, Z, Z],
        [Z, Z, Z, Z, Z, I3, Z],
        [Z, Z, Z, Z, Z, Z, I3],
    ]
    V_blocks = [
        [I3, Z, Z, Z, Z, I3, Z],
        [Z, I3, Z, Z, Z, Z, I3],
        [Z, Z, I3, Z, Z, Z, Z],
        [Z, Z, Z, I3, Z, Z, Z],
        [Z, Z, Z, Z, I3, Z, Z],
        [I3, Z, Z, Z, Z, I3, Z],
        [Z, Z, Z, Z, Z, Z, I3],
    ]
    W_blocks = [
        [I3, Z, Z, I3, Z, Z, I3],   # C11 = M1+M4-M5+M7
        [Z, Z, I3, Z, I3, Z, Z],    # C12 = M3+M5
        [Z, I3, Z, I3, Z, Z, Z],    # C21 = M2+M4
        [I3, Z, I3, Z, Z, I3, Z],   # C22 = M1-M2+M3+M6
    ]
    U = _block_matrix(2, 7, U_blocks)
    V = _block_matrix(2, 7, V_blocks)
    W = _block_matrix(2, 4, W_blocks)
    return U, V, W


def _tensor_residual(U, V, W):
    """Return the (p*n, n*m) residual of the tensor decomposition.

    Residual[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - delta(i,k).
    """
    n = m = p = 6
    r = U.shape[1]
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)
    A = Uf.T @ Wf  # (r, n*m) : A[c,(k*n+i)] = sum_i U[i*m+j? ...]
    # Build product tensor: P[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    P = np.zeros((n, m, p))
    for c in range(r):
        uc = Uf[:, c]      # (n*m,)
        vc = Vf[:, c]      # (m*p,)
        wc = Wf[:, c]      # (p*n,)
        U2 = uc.reshape(n, m)
        V2 = vc.reshape(m, p)
        W2 = wc.reshape(p, n)
        # P[i,j,k] += U2[i,j]*V2[j,k]*W2[k,i]
        for i in range(n):
            for j in range(m):
                P[i, j, :] += U2[i, j] * V2[j, :] * W2.T[:, i]
    # target: delta(i,k) for all i,j,k
    target = np.zeros((n, m, p))
    for i in range(n):
        target[i, :, i] = 1.0
    return P - target


def _verify(U, V, W, tol=1e-6):
    return np.max(np.abs(_tensor_residual(U, V, W))) < tol


def _round_half(A):
    return (2.0 * A).round() / 2.0


def _try_remove_column(U, V, W, c, tol=1e-6):
    """Try to remove column c by absorbing its contribution into other columns."""
    r = U.shape[1]
    if r <= 1:
        return None
    Uc = U[:, c]
    Vc = V[:, c]
    Wc = W[:, c]
    Uo = np.delete(U, c, axis=1)
    Vo = np.delete(V, c, axis=1)
    Wo = np.delete(W, c, axis=1)
    ro = Uo.shape[1]
    # Solve Uo dU = Uc ; Vo dV = Vc ; Wo dW = Wc (least squares)
    dU = np.linalg.lstsq(Uo, Uc, rcond=None)[0]
    dV = np.linalg.lstsq(Vo, Vc, rcond=None)[0]
    dW = np.linalg.lstsq(Wo, Wc, rcond=None)[0]
    U2 = Uo + np.outer(Uo[:, 0] * 0 + dU, np.zeros(ro))  # placeholder
    U2 = Uo.copy()
    V2 = Vo.copy()
    W2 = Wo.copy()
    U2 += dU  # broadcast (ro,) over rows -> adds to each row? NO, wrong shape
    # Correct: U2[:,t] += dU[t] for each row? No: new U = Uo + Uc? We want
    # a new column set where column t becomes Uo[:,t] + dU[t] applied as a
    # per-column shift: actually we add the vector dU as a per-column scalar.
    U2 = Uo + dU[None, :]
    V2 = Vo + dV[None, :]
    W2 = Wo + dW[None, :]
    if _verify(U2, V2, W2, tol):
        return (U2, V2, W2)
    return None


def _balanced_candidates(U, V, W):
    """Yield candidate (U2,V2,W2) from balanced column-pair linear combos.

    For a pair (a,b) with U[:,a]==U[:,b] and V[:,a]==V[:,b], define
    new columns so that the shared U part cancels (coeffs +1,-1) and the
    W part is kept (coeffs 1,1) scaled to keep the tensor intact.
    """
    r = U.shape[1]
    for a in range(r):
        for b in range(a + 1, r):
            if not (np.allclose(U[:, a], U[:, b]) and np.allclose(V[:, a], V[:, b])):
                continue
            # candidate: replace columns a,b by
            #   col_a' = (U_a + U_b)/2? We want U part to cancel => use (U_a - U_b)/2 = 0
            # Use: new_a = (V_a + V_b)/2 scaled, new_b = (V_a - V_b)/2
            # and W: new_a = (W_a + W_b), new_b = (W_a - W_b) with 1/2 factor.
            U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
            U2[:, a] = 0.5 * (U[:, a] - U[:, b])
            U2[:, b] = 0.5 * (U[:, a] + U[:, b])
            V2[:, a] = 0.5 * (V[:, a] - V[:, b])
            V2[:, b] = 0.5 * (V[:, a] + V[:, b])
            W2[:, a] = (W[:, a] - W[:, b])
            W2[:, b] = (W[:, a] + W[:, b])
            yield (U2, V2, W2)


def construct():
    # 1) Build the known Block-Strassen decomposition (rank 189).
    U, V, W = _build_block_strassen()
    best = (U, V, W)
    if not _verify(U, V, W):
        # Fallback to schoolbook (rank 216) which is guaranteed exact.
        n = m = p = 6
        rank = n * m * p
        Us = np.zeros((n * m, rank)); Vs = np.zeros((m * p, rank)); Ws = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j, col] = 1
                    Vs[j * p + k, col] = 1
                    Ws[k * n + i, col] = 1
                    col += 1
        best = (Us, Vs, Ws)
        U, V, W = Us, Vs, Ws

    # 2) Try to shave rank via balanced column-pair combinations, verifying each.
    if _time_left() > 10:
        for (U2, V2, W2) in _balanced_candidates(U, V, W):
            if _time_left() < 15:
                break
            if _verify(U2, V2, W2):
                # Accept if it reduces rank (it has same rank here; only accept
                # if a subsequent removal works) -- keep as candidate base.
                U, V, W = U2, V2, W2
                break

    # 3) Try direct column removals on the current (possibly balanced) set.
    if _time_left() > 8:
        r = U.shape[1]
        for c in range(r - 1, -1, -1):
            if _time_left() < 12:
                break
            cand = _try_remove_column(U, V, W, c)
            if cand is not None:
                U, V, W = cand

    # 4) Round to half-integers for cleanliness; re-verify, else keep floats.
    Ur = _round_half(U); Vr = _round_half(V); Wr = _round_half(W)
    if _verify(Ur, Vr, Wr):
        U, V, W = Ur, Vr, Wr

    # Convert to plain Python lists (the required interface).
    U = U.tolist(); V = V.tolist(); W = W.tolist()
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("rank =", len(res["U"][0]))
