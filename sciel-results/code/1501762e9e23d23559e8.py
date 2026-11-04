import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = 4, 4, 6

    # Strassen 2x2x2 coefficients
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]

                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    # Check for linear dependencies
    if time.time() < DEADLINE - 20:
        try:
            # Stack U, V, W
            # We need c such that U c = 0, V c = 0, W c = 0
            # This is the null space of [U; V; W]
            M = np.vstack([U, V, W])
            # Compute SVD to find null space
            # Use np.linalg.svd for better numerical stability
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            # The null space is spanned by the right singular vectors corresponding to zero singular values
            # Tolerance for zero
            tol = 1e-9
            null_mask = s < tol
            null_space = Vt[null_mask].T
            
            if null_space.shape[1] > 0:
                # We have dependencies
                # We want to remove as many columns as possible
                # This is a set cover problem, but for small null space dimension, we can iterate
                # For each null vector, we can zero out a column by adding a multiple of the null vector to the coefficients?
                # Actually, if c is in null space, we can subtract c from the decomposition?
                # No, the decomposition is fixed.
                # If c is in null space, it means the sum of terms weighted by c is zero.
                # This implies that the terms are linearly dependent.
                # We can remove one term from the support of c if we adjust the others?
                # No, we can just remove a column if its coefficient in c is non-zero and we can express it as a combination of others?
                # Simpler: if the null space has dimension d, we can reduce the rank by d.
                # We need to find a basis for the column space that excludes d columns.
                
                # Greedy removal
                removed = set()
                # Sort null vectors by sparsity?
                # For each null vector, find the column with largest magnitude and try to remove it
                for nv in null_space.T:
                    if time.time() > DEADLINE - 10: break
                    # Find index with max abs value
                    idx = np.argmax(np.abs(nv))
                    if nv[idx] != 0:
                        # Check if we can remove idx
                        # This requires checking if the remaining columns still span the tensor
                        # This is expensive.
                        pass
                
                # Alternative: Just check if the rank of [U, V, W] is less than 77
                # If rank < 77, we can reduce.
                rank_M = np.sum(s > tol)
                if rank_M < R:
                    # We can potentially reduce rank to rank_M
                    # But we need to find the specific columns to keep.
                    # We can use a greedy algorithm to find a subset of columns of size rank_M that spans the same space.
                    # This is a basis selection problem.
                    
                    # For now, just report the potential rank reduction
                    pass
        except Exception as e:
            pass

    def to_list(mat):
        result = []
        for row in mat:
            result.append([])
            for x in row:
                x_rounded = round(x)
                if abs(x - x_rounded) < 1e-9:
                    result[-1].append(int(x_rounded))
                else:
                    result[-1].append(float(x))
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
