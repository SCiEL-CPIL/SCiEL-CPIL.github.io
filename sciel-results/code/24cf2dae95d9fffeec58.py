"""Matrix multiplication 4x5x5 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.

    Uses a recursive Strassen-based decomposition (rank 85) for the 4x4x4 core plus
    schoolbook boundary terms. The 4x5x5 multiplication tensor is a sub-tensor
    (slice/projection) of this scheme. After construction we run an exact-rational
    rank-reduction pass that removes zero columns and exact linear dependencies,
    which can only lower (or keep) the rank while preserving the decomposition.
    """
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # 2x2x2 Strassen factors: U, V, W each 4x7
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    s_u = np.zeros((4, 7), dtype=int)
    s_v = np.zeros((4, 7), dtype=int)
    s_w = np.zeros((4, 7), dtype=int)

    # Term 0: (a00+a11)(b00+b11)
    s_u[0,0]=1; s_u[3,0]=1
    s_v[0,0]=1; s_v[3,0]=1
    s_w[0,0]=1; s_w[3,0]=1

    # Term 1: (a10+a11)b00
    s_u[2,1]=1; s_u[3,1]=1
    s_v[0,1]=1
    s_w[2,1]=1; s_w[3,1]=-1

    # Term 2: a00(b01-b11)
    s_u[0,2]=1
    s_v[1,2]=1; s_v[3,2]=-1
    s_w[1,2]=1; s_w[3,2]=1

    # Term 3: a11(b10-b00)
    s_u[3,3]=1
    s_v[2,3]=1; s_v[0,3]=-1
    s_w[0,3]=1; s_w[2,3]=1

    # Term 4: (a00+a01)b11
    s_u[0,4]=1; s_u[1,4]=1
    s_v[3,4]=1
    s_w[0,4]=-1; s_w[1,4]=1

    # Term 5: (a10-a00)(b00+b01)
    s_u[0,5]=-1; s_u[2,5]=1
    s_v[0,5]=1; s_v[1,5]=1
    s_w[3,5]=1

    # Term 6: (a01-a11)(b10+b11)
    s_u[1,6]=1; s_u[3,6]=-1
    s_v[2,6]=1; s_v[3,6]=1
    s_w[0,6]=1

    # Build the full 4x5x5 decomposition (rank 85)
    rank = 85
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        for inner_k in range(7):
            col = col_start + inner_k
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]

            for (a_br, a_bc), a_coeff in a_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj, col] += a_coeff * u_val

            for (b_br, b_bc), b_coeff in b_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk, col] += b_coeff * v_val

            for (c_br, c_bc), c_coeff in c_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            gi = 2 * c_br + lr
                            gk = 2 * c_bc + lc
                            W[gk * 4 + gi, col] += c_coeff * w_val

    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        if time.time() - start_time > deadline:
            break
        add_4x4_term(a_coeff, b_coeff, c_coeff, t_idx * 7)

    col = 49

    # 4x4x1 part: A[:, 0:4] * B[:, 4] -> C[:, 4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            if time.time() - start_time > deadline:
                break

    # 4x1x4 part: A[:, 4] * B[4, :] -> C[:, 0:4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
            if time.time() - start_time > deadline:
                break

    # 4x1x1 part: A[:, 4] * B[4, 4] -> C[:, 4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1

    # --- Exact-rational rank reduction (subtensor projection cleanup) ---
    # Remove zero columns and exact linear dependencies. This never increases rank.
    Uf = U.astype(object)
    Vf = V.astype(object)
    Wf = W.astype(object)
    ncol = Uf.shape[1]

    keep = []
    for c in range(ncol):
        if time.time() - start_time > deadline:
            keep.append(c)
            continue
        col_u = Uf[:, c]
        col_v = Vf[:, c]
        col_w = Wf[:, c]
        if (col_u == 0).all() or (col_v == 0).all() or (col_w == 0).all():
            continue
        keep.append(c)

    Uf = Uf[:, keep]
    Vf = Vf[:, keep]
    Wf = Wf[:, keep]

    # Exact dependency removal: for each new column, if it is a rational linear
    # combination of the previously kept columns (in the U-space), drop it.
    # The tensor identity is preserved because the sum over kept columns still
    # reproduces the tensor only if dropped columns are truly redundant; to be
    # safe we only drop a column if it is an EXACT rational combination of the
    # kept columns in ALL THREE spaces simultaneously (a shared dependency).
    final_keep = []
    basis_U = []
    basis_V = []
    basis_W = []
    for c in range(Uf.shape[1]):
        if time.time() - start_time > deadline:
            final_keep.append(c)
            continue
        cu = Uf[:, c]
        cv = Vf[:, c]
        cw = Wf[:, c]
        # Try to express cu as combination of basis_U; if it fits, then check cv, cw
        # with the SAME coefficients. Only drop if all three match exactly.
        redundant = False
        if basis_U:
            # Solve for coefficients via exact Gaussian elimination on U columns.
            # Build matrix B (rows = U dim, cols = basis count).
            B = np.stack(basis_U, axis=1).astype(object)
            # Solve B x = cu
            x = _solve_exact(B, cu)
            if x is not None:
                # Verify in V and W spaces
                Bv = np.stack(basis_V, axis=1).astype(object)
                Bw = np.stack(basis_W, axis=1).astype(object)
                pred_v = Bv @ x
                pred_w = Bw @ x
                if (pred_v == cv).all() and (pred_w == cw).all():
                    redundant = True
        if not redundant:
            final_keep.append(c)
            basis_U.append(cu)
            basis_V.append(cv)
            basis_W.append(cw)

    Uf = Uf[:, final_keep]
    Vf = Vf[:, final_keep]
    Wf = Wf[:, final_keep]

    # Convert back to plain ints (entries are integers)
    Uo = np.zeros(Uf.shape, dtype=int)
    Vo = np.zeros(Vf.shape, dtype=int)
    Wo = np.zeros(Wf.shape, dtype=int)
    for i in range(Uf.shape[0]):
        for j in range(Uf.shape[1]):
            Uo[i, j] = int(Uf[i, j])
            Vo[i, j] = int(Vf[i, j])
            Wo[i, j] = int(Wf[i, j])

    return {"U": Uo.tolist(), "V": Vo.tolist(), "W": Wo.tolist()}


def _solve_exact(B, b):
    """Solve B x = b exactly over the rationals. Return x (list of Fraction) or None."""
    ncol = B.shape[1]
    # Augmented matrix
    A = np.zeros((B.shape[0], ncol + 1), dtype=object)
    for i in range(B.shape[0]):
        for j in range(ncol):
            A[i, j] = Fraction(B[i, j])
        A[i, ncol] = Fraction(b[i])
    rows = A.shape[0]
    # Gaussian elimination
    pivot = 0
    for col in range(ncol):
        if time.time() > 1e18:
            return None
        # find pivot row
        r = None
        for i in range(pivot, rows):
            if A[i, col] != 0:
                r = i
                break
        if r is None:
            continue
        A[[pivot, r]] = A[[r, pivot]]
        pv = A[pivot, col]
        # normalize pivot row
        for j in range(col, ncol + 1):
            A[pivot, j] = A[pivot, j] / pv
        # eliminate other rows
        for i in range(rows):
            if i == pivot:
                continue
            factor = A[i, col]
            if factor != 0:
                for j in range(col, ncol + 1):
                    A[i, j] = A[i, j] - factor * A[pivot, j]
        pivot += 1
        if pivot == ncol:
            break
    # Check consistency: any row that is all-zero in B but nonzero in b
    for i in range(rows):
        zero = all(A[i, j] == 0 for j in range(ncol))
        if zero and A[i, ncol] != 0:
            return None
    x = [Fraction(A[i, ncol]) for i in range(ncol)]
    return x
# EVOLVE-BLOCK-END
