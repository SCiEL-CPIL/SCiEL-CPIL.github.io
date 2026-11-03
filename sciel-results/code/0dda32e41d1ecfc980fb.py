"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    t0 = time.time()
    DEADLINE = t0 + 140.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    def verify_decomp(U, V, W, rank):
        """Exact verification using Fraction arithmetic."""
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def schoolbook():
        """Return schoolbook decomposition of rank n*m*p = 48."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def strassen_rank44():
        """
        Construct rank-44 decomposition using block Strassen.
        Split m=3 into m1=1 and m2=2.
        Part 1: 2x1x8 (Rank 16)
        Part 2: 2x2x8, split into 4 blocks of 2x2x2 (Rank 7 each) -> Total 28.
        Total Rank: 44.
        """
        rank = 44
        U = [[0] * rank for _ in range(n * m)]  # 6 x 44
        V = [[0] * rank for _ in range(m * p)]  # 24 x 44
        W = [[0] * rank for _ in range(p * n)]  # 16 x 44

        col = 0

        # --- Part 1: 2x1x8 (j=0) ---
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        if elapsed() > DEADLINE:
            return None

        # --- Part 2: 2x2x8 (j=1,2) ---
        for block_idx in range(4):
            k_start = block_idx * 2

            steps = []

            # M1 = (A00 + A11) * (B00 + B11)
            steps.append({
                'U': {1: 1, 5: 1},
                'V': {1*p + k_start: 1, 2*p + (k_start+1): 1},
                'W': {k_start*n + 0: 1, (k_start+1)*n + 1: 1}
            })

            # M2 = (A10 + A11) * B00
            steps.append({
                'U': {4: 1, 5: 1},
                'V': {1*p + k_start: 1},
                'W': {k_start*n + 1: 1, (k_start+1)*n + 1: -1}
            })

            # M3 = A00 * (B01 - B11)
            steps.append({
                'U': {1: 1},
                'V': {1*p + (k_start+1): 1, 2*p + (k_start+1): -1},
                'W': {(k_start+1)*n + 0: 1, (k_start+1)*n + 1: 1}
            })

            # M4 = A11 * (B10 - B00)
            steps.append({
                'U': {5: 1},
                'V': {2*p + k_start: 1, 1*p + k_start: -1},
                'W': {k_start*n + 0: 1, k_start*n + 1: 1}
            })

            # M5 = (A00 + A01) * B11
            steps.append({
                'U': {1: 1, 2: 1},
                'V': {2*p + (k_start+1): 1},
                'W': {k_start*n + 0: -1, (k_start+1)*n + 0: 1}
            })

            # M6 = (A10 - A00) * (B00 + B01)
            steps.append({
                'U': {4: 1, 1: -1},
                'V': {1*p + k_start: 1, 1*p + (k_start+1): 1},
                'W': {(k_start+1)*n + 1: 1}
            })

            # M7 = (A01 - A11) * (B10 + B11)
            steps.append({
                'U': {2: 1, 5: -1},
                'V': {2*p + k_start: 1, 2*p + (k_start+1): 1},
                'W': {k_start*n + 0: 1}
            })

            for step in steps:
                for u_idx, val in step['U'].items():
                    U[u_idx][col] += val
                for v_idx, val in step['V'].items():
                    V[v_idx][col] += val
                for w_idx, val in step['W'].items():
                    W[w_idx][col] += val
                col += 1

        if col != rank:
            return None
        return U, V, W

    def try_rank16():
        """
        Try to find a rank-4 bilinear decomposition for the 2x3x2 sub-problem,
        then apply it to 4 disjoint column pairs to get rank 16.
        
        The 2x3x2 tensor: A is 2x3, B is 3x2, C is 2x2.
        For each (i, j, k_local) where i in {0,1}, j in {0,1,2}, k_local in {0,1}:
            T[i,j,k_local] = 1
        Decomposition: sum_r U[i,j,r] * V[j,k_local,r] * W[k_local,i,r] = 1
        
        We solve this bilinear system using iterative linearization.
        """
        if elapsed() > DEADLINE:
            return None

        r = 4  # rank of 2x3x2 sub-problem
        # Dimensions for sub-problem: n_sub=2, m_sub=3, p_sub=2
        n_sub, m_sub, p_sub = 2, 3, 2

        # Number of tensor entries: n_sub * m_sub * p_sub = 12
        # Variables per rank term: n_sub*m_sub + m_sub*p_sub + p_sub*n_sub = 6+6+4 = 16
        # Total variables: r * 16 = 64
        # Constraints: 12 (each = 1)
        # Very underdetermined!

        # We'll use an iterative approach: fix two of the three matrices, solve for the third.
        # Start with a random initialization and iterate.

        np.random.seed(42)
        U_sub = np.random.randn(n_sub * m_sub, r) * 0.5
        V_sub = np.random.randn(m_sub * p_sub, r) * 0.5
        W_sub = np.random.randn(p_sub * n_sub, r) * 0.5

        best_residual = float('inf')
        best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()

        for iteration in range(5000):
            if elapsed() > DEADLINE:
                return None

            # Compute residual for current decomposition
            residual = 0.0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        s = 0.0
                        for c in range(r):
                            s += U_sub[i * m_sub + j, c] * V_sub[j * p_sub + k, c] * W_sub[k * n_sub + i, c]
                        residual += (s - 1.0) ** 2

            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()

            if residual < 1e-12:
                break

            # Iteratively update each matrix
            for which in range(3):
                if elapsed() > DEADLINE:
                    return None

                if which == 0:
                    # Solve for U_sub: for each (i,j), sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
                    # This is: for each (i,j), U[i,j,:] dot (V[j,k,:] * W[k,i,:] elementwise) = 1 for each k
                    # Actually it's bilinear, so we linearize: fix V and W, solve for U
                    # For each (i,j): the equation is sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c]) = 1 for each k
                    # This gives p_sub equations per (i,j) with r unknowns
                    # Stack all: (n_sub*m_sub*p_sub) equations, (n_sub*m_sub*r) unknowns
                    A_mat = np.zeros((n_sub * m_sub * p_sub, n_sub * m_sub * r))
                    b_vec = np.ones(n_sub * m_sub * p_sub)
                    eq_idx = 0
                    for i in range(n_sub):
                        for j in range(m_sub):
                            for k in range(p_sub):
                                for c in range(r):
                                    A_mat[eq_idx, (i * m_sub + j) * r + c] = V_sub[j * p_sub + k, c] * W_sub[k * n_sub + i, c]
                                eq_idx += 1
                    # Least squares
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_sub = sol.reshape(n_sub * m_sub, r)

                elif which == 1:
                    # Solve for V_sub
                    A_mat = np.zeros((n_sub * m_sub * p_sub, m_sub * p_sub * r))
                    b_vec = np.ones(n_sub * m_sub * p_sub)
                    eq_idx = 0
                    for i in range(n_sub):
                        for j in range(m_sub):
                            for k in range(p_sub):
                                for c in range(r):
                                    A_mat[eq_idx, (j * p_sub + k) * r + c] = U_sub[i * m_sub + j, c] * W_sub[k * n_sub + i, c]
                                eq_idx += 1
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_sub = sol.reshape(m_sub * p_sub, r)

                else:
                    # Solve for W_sub
                    A_mat = np.zeros((n_sub * m_sub * p_sub, p_sub * n_sub * r))
                    b_vec = np.ones(n_sub * m_sub * p_sub)
                    eq_idx = 0
                    for i in range(n_sub):
                        for j in range(m_sub):
                            for k in range(p_sub):
                                for c in range(r):
                                    A_mat[eq_idx, (k * n_sub + i) * r + c] = U_sub[i * m_sub + j, c] * V_sub[j * p_sub + k, c]
                                eq_idx += 1
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_sub = sol.reshape(p_sub * n_sub, r)

        # Check if we found a good solution
        if best_residual > 1e-6:
            return None

        # Now build the full rank-16 decomposition
        # Apply the rank-4 sub-decomposition to 4 disjoint column pairs
        total_rank = 4 * r  # 16
        U_full = [[0] * total_rank for _ in range(n * m)]   # 6 x 16
        V_full = [[0] * total_rank for _ in range(m * p)]   # 24 x 16
        W_full = [[0] * total_rank for _ in range(p * n)]   # 16 x 16

        for block in range(4):
            k_start = block * 2
            col_base = block * r

            for c in range(r):
                # U: A[i, j] for i in {0,1}, j in {0,1,2}
                for i in range(n_sub):
                    for j in range(m_sub):
                        U_full[i * m + j][col_base + c] = best_U[i * m_sub + j, c]

                # V: B[j, k_local] for j in {0,1,2}, k_local in {0,1}
                # Global: B[j, k_start + k_local]
                for j in range(m_sub):
                    for k_loc in range(p_sub):
                        V_full[j * p + k_start + k_loc][col_base + c] = best_V[j * p_sub + k_loc, c]

                # W: C[k_local, i] for k_local in {0,1}, i in {0,1}
                # Global: W[k_start + k_local, i] -> index (k_start+k_loc)*n + i
                for k_loc in range(p_sub):
                    for i in range(n_sub):
                        W_full[(k_start + k_loc) * n + i][col_base + c] = best_W[k_loc * n_sub + i, c]

        return U_full, V_full, W_full

    # 1. Try rank-16 construction
    if elapsed() < DEADLINE - 10:
        result16 = try_rank16()
        if result16 is not None:
            U16, V16, W16 = result16
            if verify_decomp(U16, V16, W16, 16):
                return {"U": U16, "V": V16, "W": W16}

    # 2. Try the deterministic rank-44 construction
    result44 = strassen_rank44()
    if result44 is not None:
        U44, V44, W44 = result44
        if verify_decomp(U44, V44, W44, 44):
            return {"U": U44, "V": V44, "W": W44}

    # 3. Fallback to schoolbook rank 48
    U_sch, V_sch, W_sch = schoolbook()
    return {"U": U_sch, "V": V_sch, "W": W_sch}
# EVOLVE-BLOCK-END
