"""Matrix multiplication 4x4 x 4x6 tensor decomposition.

Strategy:
  1. Build a rank-77 decomposition by top-level Strassen (7 terms) where each
     term is a (2x2) x (2x3) product.  For each such product, use inner
     Strassen (rank 7) for the 2x2 x 2x2 part and schoolbook (rank 4) for the
     2x2 x 2x1 part, giving 11 terms per top-level term.  Total: 7 * 11 = 77.
  2. Attempt to reduce rank further via coordinate descent on the tensor
     equation, trying to zero out columns and adjusting the rest.
  3. Fallback to the rank-77 construction if no improvement is found.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2*2x2 coefficients
# A blocks indexed 0..3: 00, 01, 10, 11
# B blocks indexed 0..3: 00, 01, 10, 11
# C blocks indexed 0..3: 00, 01, 10, 11
#
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A01)*B11
# M6 = (A10-A00)*(B00+B01)
# M7 = (A01-A11)*(B10+B11)
#
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

# alpha[s][a] = coefficient of A-block a in M_s
# beta[s][b]  = coefficient of B-block b in M_s
# recon[s][c] = list of (a, b, gamma) meaning C-block c gets gamma * M_s

alpha = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: A00 + A11
    {0: 0, 1: 0, 2: 1, 3: 1},   # M2: A10 + A11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M3: A00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M4: A11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M5: A00 + A01
    {0: -1, 1: 0, 2: 1, 3: 0},  # M6: A10 - A00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M7: A01 - A11
]
beta = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: B00 + B11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M2: B00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M3: B01 - B11
    {0: -1, 1: 0, 2: 1, 3: 0},  # M4: B10 - B00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M5: B11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M6: B00 + B01
    {0: 0, 1: 0, 2: 1, 3: 1},   # M7: B10 + B11
]
# recon[s] = list of (a_block, b_block, gamma) for C = sum_s recon[s] * M_s
recon = [
    [(0, 0, 1), (1, 1, 1)],       # M1 -> C00(+1), C11(+1)
    [(1, 0, 1), (1, 1, -1)],      # M2 -> C10(+1), C11(-1)
    [(0, 1, 1), (1, 1, 1)],       # M3 -> C01(+1), C11(+1)
    [(0, 0, 1), (1, 0, 1)],       # M4 -> C00(+1), C10(+1)
    [(0, 0, -1), (0, 1, 1)],      # M5 -> C00(-1), C01(+1)
    [(1, 1, 1)],                  # M6 -> C11(+1)
    [(0, 0, 1)],                  # M7 -> C00(+1)
]


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 115  # safety margin

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def err_of(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - recon) ** 2)

    # ---- Build rank-77 decomposition via top-level Strassen ----
    U77, V77, W77 = _build_rank77(n, m, p)
    
    # Verify rank-77
    if not check(U77, V77, W77, atol=1e-9):
        # Fallback to schoolbook if rank-77 fails
        rank_sb = n * m * p  # 96
        U_sb = np.zeros((n * m, rank_sb), dtype=np.float64)
        V_sb = np.zeros((m * p, rank_sb), dtype=np.float64)
        W_sb = np.zeros((p * n, rank_sb), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}

    best_rank = 77
    best_U, best_V, best_W = U77.copy(), V77.copy(), W77.copy()

    # Try to reduce rank via coordinate descent
    # Start by trying to drop columns from the rank-77 decomposition
    for target_rank in range(76, 60, -1):
        if time.time() > deadline - 20:
            break

        # Start from current best, but truncate to target_rank columns
        U = best_U[:, :target_rank].copy()
        V = best_V[:, :target_rank].copy()
        W = best_W[:, :target_rank].copy()

        err = err_of(U, V, W)

        # Coordinate descent with multiple restarts
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            # Add small random perturbation for restart
            if restart > 0:
                U += np.random.randn(n * m, target_rank) * 0.1
                V += np.random.randn(m * p, target_rank) * 0.1
                W += np.random.randn(p * n, target_rank) * 0.1
                err = err_of(U, V, W)

            # Coordinate descent
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                if err < 1e-10:
                    break
                improved = False
                for c in range(target_rank):
                    if time.time() > deadline - 8:
                        break
                    Uo, Vo, Wo = U[:, c].copy(), V[:, c].copy(), W[:, c].copy()
                    # Try a few different step sizes
                    for step in [0.05, 0.01, 0.001]:
                        U[:, c] = Uo + np.random.randn(n * m) * step
                        V[:, c] = Vo + np.random.randn(m * p) * step
                        W[:, c] = Wo + np.random.randn(p * n) * step
                        new_err = err_of(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                            break
                    if not improved:
                        U[:, c], V[:, c], W[:, c] = Uo, Vo, Wo
                if not improved:
                    break

        if err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break

    # Final polish: high-precision refinement
    if time.time() < deadline - 5 and best_rank < 96:
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        err = err_of(U, V, W)
        for it in range(50):
            if time.time() > deadline - 3:
                break
            if err < 1e-12:
                break
            improved = False
            for c in range(best_rank):
                Uo, Vo, Wo = U[:, c].copy(), V[:, c].copy(), W[:, c].copy()
                U[:, c] += np.random.randn(n * m) * 0.001
                V[:, c] += np.random.randn(m * p) * 0.001
                W[:, c] += np.random.randn(p * n) * 0.001
                new_err = err_of(U, V, W)
                if new_err < err:
                    err = new_err
                    improved = True
                else:
                    U[:, c], V[:, c], W[:, c] = Uo, Vo, Wo
            if not improved:
                break
        if err < 1e-10:
            best_U, best_V, best_W = U, V, W

    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}


def _build_rank77(n, m, p):
    """Build the rank-77 decomposition using top-level Strassen with inner Strassen+schoolbook."""
    R = 77
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)

    col = 0
    for s in range(7):
        # For each top-level Strassen term s:
        # X_s is a 2x2 matrix (linear combo of A blocks)
        # Y_s is a 2x3 matrix (linear combo of B blocks)
        # Compute X_s * Y_s using: inner Strassen for cols 0-1, schoolbook for col 2
        
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # Inner Strassen term si
            # U part: X_s entries with inner_alpha[si] coefficients
            for a in range(4):
                coeff_a = alpha[si].get(a, 0)
                if coeff_a != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        coeff_r = alpha[s].get(r, 0)
                        if coeff_r != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * m + (br * 2 + x_col)
                            U[u_idx, col] += coeff_a * coeff_r
            
            # V part: Y_s entries with inner_beta[si] coefficients
            for b in range(4):
                coeff_b = beta[si].get(b, 0)
                if coeff_b != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        coeff_c = beta[s].get(c, 0)
                        if coeff_c != 0:
                            ac, bc = c // 2, c % 2
                            # B_c is 2x3, B_c[y_row, y_inner_col]
                            v_idx = (ac * 2 + y_row) * p + (bc * 3 + y_inner_col)
                            V[v_idx, col] += coeff_b * coeff_c
            
            # W part: output of inner Strassen
            for a_idx, b_idx, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # C block (a_idx, b_idx) at position (za, zb)
                    # Global: row = a_idx*2 + za, col = b_idx*3 + zb
                    w_idx = (b_idx * 3 + zb) * n + (a_idx * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        # X_s (2x2) * Y_s[:,2] (2x1) -> Z (2x1)
        # Z[u] = sum_v X_s[u,v] * Y_s[v,2]
        # 4 schoolbook multiplications
        
        for u in range(2):
            for v in range(2):
                # Term: X_s[u,v] * Y_s[v,2]
                
                # U part
                for r in range(4):
                    coeff_r = alpha[s].get(r, 0)
                    if coeff_r != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * m + (br * 2 + v)
                        U[u_idx, col] += coeff_r
                
                # V part
                for c in range(4):
                    coeff_c = beta[s].get(c, 0)
                    if coeff_c != 0:
                        ac, bc = c // 2, c % 2
                        # Y_s[v,2] = sum_c beta[s][c] * B_c[v, 2]
                        # B_c is 2x3, so B_c[ac*2+v, bc*3+2]
                        v_idx = (ac * 2 + v) * p + (bc * 3 + 2)
                        V[v_idx, col] += coeff_c
                
                # W part
                for a_idx, b_idx, gamma in recon[s]:
                    # Z[u] contributes to C[a_idx*2+u, b_idx*3+2]
                    w_idx = (b_idx * 3 + 2) * n + (a_idx * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1

    return U, V, W


def _to_list(mat):
    """Convert numpy array to list of lists with integer rounding where possible."""
    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    print(f"Rank: {U.shape[1]}")
    # Verify
    n, m, p = 4, 4, 6
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    max_err = np.max(np.abs(T - recon))
    print(f"Max error: {max_err}")
    print(f"Valid: {np.allclose(recon, T, atol=1e-9)}")
