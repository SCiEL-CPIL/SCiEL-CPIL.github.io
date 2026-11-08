import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    # For 2x4 * 4x8: n=2, m=4, p=8
    # X is n x m (2x4), Y is m x p (4x8), Z is n x p (2x8)
    # Tensor T[i,j,k] = delta_{i,k} (since Z[i,k] = sum_j X[i,j] Y[j,k])
    # Wait, the tensor for matrix multiplication C = A*B where A is (n,m) and B is (m,p)
    # has entries T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 if the indices match, which is just the identity in j
    # Actually the tensor is: T_{i,j,k} = δ_{jk} (no, that's not right either)
    # The matrix multiplication tensor T_{i,j,k} satisfies:
    # (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # So T_{i,j,k} = 1 for all valid (i,j,k) - it's a 3D identity-like tensor
    # Actually no: T_{i,j,k} is the coefficient such that (A*B)_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}
    # So T_{i,j,k} = 1 for all i,j,k where i in [0,n), j in [0,m), k in [0,p)
    # The tensor has dimensions n x m x p = 2 x 4 x 8 = 64 elements, all equal to 1

    # Rank lower bound: max(n*p, m*p, n*m) for standard lower bound, but that's not tight
    # For 2x4x8: naive rank = 2*4*8 = 64
    # Strassen-like: 8 blocks of 2x2x2, each rank 7 = 56
    # Can we do better?

    # Let's try a direct approach: start with rank 64 (naive) and try to reduce
    # Or start with a known good decomposition and try to find linear dependencies

    # Approach: Use the fact that the tensor T[i,j,k] = 1 (all ones) for 2x4x8
    # We need to find U (2*4 x R), V (4*8 x R), W (8*2 x R) such that:
    # T[i,j,k] = sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

    # Let me try a structured approach:
    # For 2x4x8, we can think of it as 4 blocks of 2x1x8 or 2 blocks of 2x2x8
    # Or use the known result that rank(2x4x8) might be less than 56

    # Let's try rank 48 first (6 blocks of 2x2x2 with 8 rank each? No...)
    # Actually, let's just try to find the minimal rank by solving the system

    # The tensor equation is bilinear/multilinear. For a given rank R, we have:
    # 64 equations, 3*R*R unknowns (U has 8*R entries, V has 32*R, W has 16*R)
    # Total unknowns: 56*R
    # This is a system of 64 polynomial equations of degree 3 in 56*R unknowns

    # A better approach: fix U and V, then W is determined linearly
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For fixed U,V: sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r]) = 1
    # This is a linear system in W entries

    # Even better: use the fact that we can parameterize
    # Let's try a rank-reduction from the known rank-56 solution

    # Start with the rank-56 Strassen solution
    # Then try to see if we can reduce the rank by finding linear dependencies

    # For rank R, the condition is that the 64x(3R) matrix (or rather the tensor)
    # can be expressed as a sum of R rank-1 tensors

    # Let me try a different structural approach:
    # Use the decomposition of 2x4x8 as a Kronecker product structure
    # 2x4x8 = (2x2x2) x (1x2x4) or similar

    # Actually, let me just try to find a rank < 56 solution by direct construction
    # or by solving a smaller system

    # Known results: rank(2x3x3) = 7 (Strassen-like), rank(2x4x4) might be 16 or less
    # For 2x4x8: let's try rank 48, 44, 40, etc.

    # Let me try a systematic approach:
    # 1. Start with rank 56 (Strassen)
    # 2. Try to eliminate terms by solving for dependencies

    # Alternative: use the fact that for 2xnxn, the rank is at most 2*rank(nxn)
    # For 4x4, rank is at most 16 (naive) or 12 (if Strassen-like applies)
    # So 2x4x8 <= 2 * rank(4x8) ... this isn't quite right

    # Let me just try to construct a solution with rank < 56
    # by exploiting the rectangular structure

    # For 2x4x8, we can write:
    # Z = X * Y where X is 2x4, Y is 4x8
    # Z[i,k] = sum_j X[i,j] Y[j,k]

    # We can split X into two 2x2 blocks: X = [X_left | X_right]
    # And Y into two 4x4 blocks: Y = [Y_top; Y_bottom]
    # Then Z = X_left * Y_top + X_right * Y_bottom

    # Each 2x2x4 product can be done with rank... 
    # 2x2x4: naive is 16, Strassen-like might be less

    # Actually, let me try a direct numerical approach:
    # Assume rank R, and try to solve for the factors

    # For a given R, the problem is:
    # Find U (8 x R), V (32 x R), W (16 x R) such that
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i in [0,2), j in [0,4), k in [0,8)

    # This is a system of 64 equations.
    # If we fix U and V, W is determined by a linear system.
    # Specifically, for each (k,i), we have:
    # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j

    # Let's fix U and V randomly (or with structure) and solve for W,
    # then check if the reconstruction is exact.

    # Better: use an iterative approach
    # 1. Initialize U, V, W randomly with rank R
    # 2. Alternate: fix two, solve for the third (least squares)
    # 3. Repeat until convergence

    # Let me try this for various ranks

    def try_rank(R, max_iters=200, tol=1e-10):
        """Try to find a rank-R decomposition using ALS (Alternating Least Squares)"""
        if time.time() > deadline - 10:
            return None

        np.random.seed(R * 1000 + 42)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)

        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None

            # Update W: for each (k, i), solve for W[k*n+i, :]
            # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
            # Let A[k,i] be a matrix where A[k,i][j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Then A[k,i] @ W[k*n+i, :] = 1 (vector of ones of length m)
            # This is an underdetermined system (m equations, R unknowns) if R > m
            # We want the minimum norm solution

            for k in range(p):
                for i in range(n):
                    # Build the system: for each j, sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                    A = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    # Solve A @ W_vec = 1 (m x R system, R unknowns)
                    # Use least squares
                    W_vec, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k * n + i, :] = W_vec

            # Update U: for each (i, j), solve for U[i*m+j, :]
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    U_vec, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i * m + j, :] = U_vec

            # Update V: for each (j, k), solve for V[j*p+k, :]
            # sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    V_vec, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j * p + k, :] = V_vec

            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                                  for r in range(R))
                        err = max(err, abs(val - 1.0))
            if err < tol:
                return U, V, W

        return None

    # Try ranks from 56 downward
    best_result = None
    for R in range(56, 32, -1):
        if time.time() > deadline - 20:
            break
        result = try_rank(R)
        if result is not None:
            best_result = result
            break

    if best_result is None:
        # Fallback to rank 56 Strassen
        R = 56
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))

        # Strassen 2x2x2 patterns
        strassen_U = [
            [(0, 1), (3, 1)],
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (0, -1)],
            [(1, -1), (3, 1)],
        ]
        strassen_V = [
            [(0, 1), (3, 1)],
            [(0, 1)],
            [(1, 1), (3, -1)],
            [(2, 1), (0, -1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (3, 1)],
        ]
        strassen_W = [
            [(0, 1), (3, 1)],
            [(2, 1)],
            [(1, 1)],
            [(0, 1), (2, 1)],
            [(0, -1), (1, 1)],
            [(3, 1)],
            [(0, -1)],
        ]

        def add_block(base_rank, i_off, j_off, k_off):
            for r in range(7):
                cr = base_rank + r
                for idx, val in strassen_U[r]:
                    li, lj = divmod(idx, 2)
                    gi, gj = li + i_off, lj + j_off
                    U[gi * 4 + gj, cr] += val
                for idx, val in strassen_V[r]:
                    lj, lk = divmod(idx, 2)
                    gj, gk = lj + j_off, lk + k_off
                    V[gj * 8 + gk, cr] += val
                for idx, val in strassen_W[r]:
                    lk, li = divmod(idx, 2)
                    gk, gi = lk + k_off, li + i_off
                    W[gk * 2 + gi, cr] += val

        for k_off in range(0, 8, 2):
            for j_off in range(0, 4, 2):
                base = k_off * 7 + j_off * 7
                add_block(base, 0, j_off, k_off)

        U_list = [[U[i, r] for r in range(R)] for i in range(n * m)]
        V_list = [[V[i, r] for r in range(R)] for i in range(m * p)]
        W_list = [[W[i, r] for r in range(R)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    U_arr, V_arr, W_arr = best_result
    R = U_arr.shape[1]
    U_list = [[U_arr[i, r] for r in range(R)] for i in range(n * m)]
    V_list = [[V_arr[i, r] for r in range(R)] for i in range(m * p)]
    W_list = [[W_arr[i, r] for r in range(R)] for i in range(p * n)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

