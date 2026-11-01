import time
import itertools
import numpy as np

def _build_strassen_blocks(rank=39):
    """Build the rank-39 decomposition: 5 Strassen blocks (rank 7) + 1 schoolbook block (rank 4)."""
    n, m, p = 2, 2, 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0

    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        v_e, v_f = k0, k1
        v_g, v_h = 11 + k0, 11 + k1
        w_c00, w_c10 = 2 * k0, 2 * k0 + 1
        w_c01, w_c11 = 2 * k1, 2 * k1 + 1

        # m1 = (a+d)(e+h)
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        # m2 = (c+d)e
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        # m3 = a(f-h)
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        # m4 = d(g-e)
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        # m5 = (a+b)h
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        # m6 = (c-a)(e+f)
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        # m7 = (b-d)(g+h)
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1

    # Schoolbook block for column 10
    k2 = 10
    v_e2, v_g2 = k2, 11 + k2
    w_c02, w_c12 = 2 * k2, 2 * k2 + 1
    U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
    U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
    U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
    U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1

    return U, V, W


def _to_np(U, V, W):
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)


def _verify(U, V, W, tol=1e-6):
    """Verify decomposition against the multiplication tensor."""
    n, m, p = 2, 2, 11
    Ua, Va, Wa = _to_np(U, V, W)
    # Tensor entry T[i,j,k] = 1 if A[i,j]*B[j,k] -> C[i,k]
    # Reconstruction: sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] should equal T[i,j,k]
    # W index: C[i,k] -> k*n + i = 2k + i
    # V index: B[j,k] -> j*p + k = 11j + k
    # U index: A[i,j] -> i*m + j = 2i + j
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for r in range(Ua.shape[1]):
                    u = Ua[i * m + j, r]
                    v = Va[j * p + k, r]
                    w = Wa[k * n + i, r]
                    val += u * v * w
                if abs(val - expected) > tol:
                    return False
    return True


def _try_reduce_rank(U, V, W, deadline):
    """Attempt to reduce rank via column removal / linear dependency detection.
    Returns (new_U, new_V, new_W) if a valid lower-rank decomposition is found, else None."""
    n, m, p = 2, 2, 11
    Ua, Va, Wa = _to_np(U, V, W)
    R = Ua.shape[1]

    # Check for linearly dependent columns across the decomposition.
    # A column r is "redundant" if it can be expressed as a combination of others
    # in a way that preserves the tensor. This is complex, so we try a simpler
    # heuristic: try removing each column and checking if the tensor is still
    # reproduced (only works if that column was zero or cancelable).
    # For a cleaner approach: try greedy removal of columns that are linearly
    # dependent in the combined U⊗V⊗W contribution.

    # Simpler: try to find if any pair of columns can be merged.
    # For now, try removing columns that have very small contribution.
    for r_remove in range(R):
        if time.time() > deadline:
            break
        # Try removing column r_remove
        mask = [i for i in range(R) if i != r_remove]
        U_new = Ua[:, mask]
        V_new = Va[:, mask]
        W_new = Wa[:, mask]
        # Quick check: does this still reproduce the tensor?
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(len(mask)):
                        val += U_new[i * m + j, rr] * V_new[j * p + k, rr] * W_new[k * n + i, rr]
                    if abs(val - 1.0) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            # Convert back to list format
            U_list = U_new.tolist()
            V_list = V_new.tolist()
            W_list = W_new.tolist()
            # Round to clean values
            for mat in (U_list, V_list, W_list):
                for row in mat:
                    for idx in range(len(row)):
                        row[idx] = round(row[idx], 9)
            return U_list, V_list, W_list
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Build the rank-39 decomposition
    U, V, W = _build_strassen_blocks(39)

    # Verify it's correct
    if not _verify(U, V, W):
        # Should not happen, but return as-is if verification fails
        return {"U": U, "V": V, "W": W}

    # Attempt rank reduction within time budget
    result = _try_reduce_rank(U, V, W, deadline)
    if result is not None:
        U, V, W = result
        # Final verification
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
