"""Rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.

Strategy:
  - Build a verified rank-88 construction (schoolbook for row 0 + Strassen on 2x4x8).
  - Attempt exact rank reduction via:
      (a) direct linear-dependency check on the rank-1 terms,
      (b) random perturbation + re-reduction to escape local structure.
  - Return the best (lowest-rank) verified decomposition found.
"""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 8]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check the cyclic bilinear identity:
       sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1  for all i,j,k.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_rank88():
    """Verified rank-88 construction: schoolbook(1x4x8) + Strassen(2x4x8).

    For i=0: 32 schoolbook terms.
    For i=1,2: partition 4 into 2 blocks of 2, 8 into 4 blocks of 2.
    Each 2x2x2 subblock uses 7 Strassen terms. Total: 2*2*4*7 = 112? No.
    Actually: 2 (i-blocks) * 2 (j-blocks) * 4 (k-blocks) * 7 (Strassen) = 112.
    But we only need 56, so we use a smarter partition.

    Let me just build it directly with the known-good Strassen 2x2x2 pattern
    applied to each (i,j,k) block, but sharing structure.

    Actually, the simplest correct approach: for i in {1,2}, treat the 4x8
    multiplication as 2x(2x4) + 2x(2x4) where each 2x2x4 uses 7 terms per
    2x2x2 sub-block... this gets complicated. Let me just use the verified
    construction from the current program.
    """
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    # Part 1: i=0 schoolbook 1x4x8 -> 32 terms
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
    # Part 2: i=1,2 Strassen on 2x2x2 blocks
    # Strassen 2x2x2: 7 multiplications
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # For 2x4x8: partition j into 2 blocks of 2, k into 4 blocks of 2.
    # For each (i-block, j-block, k-block) we do a 2x2x2 Strassen.
    # i-blocks: 1 (since i is 2 elements: rows 1 and 2)
    # j-blocks: 2 (cols 0-1, cols 2-3)
    # k-blocks: 4 (cols 0-1, 2-3, 4-5, 6-7)
    # Total: 1 * 2 * 4 * 7 = 56 terms.

    sU = [
        [1, 0, 0, 1],   # M1: A11+A22
        [0, 0, 1, 1],   # M2: A21+A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11+A12
        [-1, 0, 1, 0],  # M6: A21-A11
        [0, 1, 0, -1],  # M7: A12-A22
    ]
    sV = [
        [1, 0, 0, 1],   # M1: B11+B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B21-B22
        [-1, 0, 1, 0],  # M4: B12-B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11+B12
        [0, 0, 1, 1],   # M7: B21+B22
    ]
    sW = [
        [1, 0, 0, 1],   # M1: C11
        [0, 0, 1, -1],  # M2: C11
        [0, 1, 0, 1],   # M3: C12
        [1, 0, 1, 0],   # M4: C11
        [-1, 1, 0, 0],  # M5: C11
        [0, 0, 0, 1],   # M6: C22
        [1, 0, 0, 0],   # M7: C11
    ]
    # Wait, the W matrix for Strassen should be:
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    # So W for each (i,k) pair:
    # W[0,0] (C11 from M1): 1
    # W[0,0] (C11 from M4): 1
    # W[0,0] (C11 from M5): -1
    # W[0,0] (C11 from M7): 1
    # W[0,1] (C12 from M3): 1
    # W[0,1] (C12 from M5): 1
    # W[1,0] (C21 from M2): 1
    # W[1,0] (C21 from M4): 1
    # W[1,1] (C22 from M1): 1
    # W[1,1] (C22 from M2): -1
    # W[1,1] (C22 from M3): 1
    # W[1,1] (C22 from M6): 1

    # Let me redefine sW properly.
    # sW[t][l] where t is the Strassen term index (0-6) and l is the (i,k) index (0-3)
    # l=0: (i=0,k=0), l=1: (i=0,k=1), l=2: (i=1,k=0), l=3: (i=1,k=1)
    sW = [
        [1, 0, 0, 1],   # M1: C11=1, C22=1
        [0, 0, 1, -1],  # M2: C21=1, C22=-1
        [0, 1, 0, 1],   # M3: C12=1, C22=1
        [1, 0, 1, 0],   # M4: C11=1, C21=1
        [-1, 1, 0, 0],  # M5: C11=-1, C12=1
        [0, 0, 0, 1],   # M6: C22=1
        [1, 0, 0, 0],   # M7: C11=1
    ]

    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (js + lj), col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(js + lj) * p + (ks + lk), col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(ks + lk) * n + (1 + li), col] += sW[t][l]
                col += 1
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, deadline, tol=1e-9):
    """Attempt to reduce rank by finding genuine linear dependencies among
    the rank-1 terms. A column r is redundant if there exist CONSTANT
    coefficients c_s (s != r) such that for all (i,j,k):
      U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = sum_{s!=r} c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
    """
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    W = np.asarray(W, dtype=float).copy()
    R = U.shape[1]
    nmp = n * m * p

    if time.time() > deadline - 10:
        return U, V, W

    # Precompute f_s(i,j,k) = U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] for all s
    # f is a 3D array of shape (R, n, m, p)
    f = np.zeros((R, n, m, p))
    for r in range(R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    f[r, i, j, k] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]

    # For each column r, check if f_r can be expressed as a constant linear
    # combination of the other f_s. This is a linear system:
    #   sum_{s != r} c_s * f_s(i,j,k) = f_r(i,j,k)  for all (i,j,k)
    # This is nmp equations in (R-1) unknowns.

    best_U, best_V, best_W = U, V, W
    best_rank = R

    for r in range(R):
        if time.time() > deadline - 5:
            break

        # Build A: nmp x (R-1), b: nmp
        A = np.zeros((nmp, R - 1))
        b = np.zeros(nmp)
        idx = 0
        s_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b[idx] = f[r, i, j, k]
                    for s in range(R):
                        if s == r:
                            continue
                        A[idx, s_idx] = f[s, i, j, k]
                        s_idx += 1
                    idx += 1

        # Solve A c = b
        try:
            c, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            residual = np.linalg.norm(A @ c - b)
            if residual < tol:
                # Column r is redundant! Remove it.
                mask = [s for s in range(R) if s != r]
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                best_U, best_V, best_W = U, V, W
                best_rank = len(mask)
                # Reset f for the new R
                R = len(mask)
                f = np.zeros((R, n, m, p))
                for r2 in range(R):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                f[r2, i, j, k] = U[i * m + j, r2] * V[j * p + k, r2] * W[k * n + i, r2]
                # Restart the loop from the beginning
                r = -1
                continue
        except np.linalg.LinAlgError:
            continue
        r += 1

    return best_U, best_V, best_W


def _random_perturb_and_reduce(U, V, W, n, m, p, deadline, n_trials=50, tol=1e-9):
    """Randomly perturb the decomposition and try to reduce rank.

    The idea: if the current decomposition is not rank-minimal, a random
    perturbation followed by rank reduction might find a lower-rank form.
    """
    U0 = np.asarray(U, dtype=float).copy()
    V0 = np.asarray(V, dtype=float).copy()
    W0 = np.asarray(W, dtype=float).copy()
    R0 = U0.shape[1]
    nmp = n * m * p

    best_U, best_V, best_W = U0, V0, W0
    best_rank = R0

    rng = np.random.RandomState(42)

    for trial in range(n_trials):
        if time.time() > deadline - 10:
            break

        # Perturb
        scale = 0.1
        U_p = U0 + scale * rng.randn(*U0.shape)
        V_p = V0 + scale * rng.randn(*V0.shape)
        W_p = W0 + scale * rng.randn(*W0.shape)

        # Try to reduce rank on the perturbed decomposition
        U_r, V_r, W_r = _try_reduce_rank(U_p, V_p, W_p, n, m, p, deadline, tol)

        # Verify
        if _verify(U_r, V_r, W_r, n, m, p, tol):
            R_new = U_r.shape[1]
            if R_new < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = R_new

    return best_U, best_V, best_W


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Build the rank-88 decomposition
    U, V, W = _build_rank88()

    # Verify it's correct
    if not _verify(U, V, W, n, m, p):
        raise RuntimeError("Rank-88 construction failed verification")

    # Try to reduce rank (best effort)
    U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)

    # Try random perturbation + reduction
    if time.time() < deadline - 20:
        U, V, W = _random_perturb_and_reduce(U, V, W, n, m, p, deadline)

    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fall back to the original rank-88
        U, V, W = _build_rank88()

    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")

    # Quick verification
    U_arr = np.array(result['U'])
    V_arr = np.array(result['V'])
    W_arr = np.array(result['W'])
    n, m, p = 3, 4, 8
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0
                for rr in range(rank):
                    t_val += result['U'][i*m+j][rr] * result['V'][j*p+k][rr] * result['W'][k*n+i][rr]
                if abs(t_val - 1) > 1e-9:
                    max_err = max(max_err, abs(t_val - 1))
    print(f"Max tensor error: {max_err}")
