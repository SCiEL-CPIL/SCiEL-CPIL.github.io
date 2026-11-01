"""
Strategy: Global algebraic search for a lower-rank decomposition.
The current baseline uses 6 independent Strassen blocks (rank 42).
We hypothesize that a global decomposition of rank R (where 24 <= R < 42) exists
by sharing structure between blocks.

We model the solution as a linear combination of "Strassen-like" basis tensors.
Specifically, we allow the 12 columns of B to be partitioned into 6 pairs,
and for each pair, we use a rank-7 decomposition (Strassen).
This gives a candidate space of 42 terms.
We then check if these 42 terms can be reduced to R terms by finding a linear
dependence or a better basis.

However, a more direct approach is to assume the rank is R and solve for the
factors using a structured initialization based on Strassen, but allowing
cross-block mixing.

Given the 150s limit, we will:
1. Construct the standard 42-rank solution.
2. Attempt to find a rank-36 solution using a 4x[2x2x3] partition with a
   hypothetical rank-9 scheme for [2x2x3].
3. Attempt to find a rank-30 solution using a 5x[2x2x2.4] partition? No, dimensions must be integers.
4. Use a random linear combination of the 42 Strassen blocks to see if we can
   collapse the rank.

Actually, the most promising "structural" change is to use the **Bini-Capovani
decomposition** or similar for 2x2x2, but applied globally.
For 2x2x12, the rank is at least 24.
Let's try to construct a rank-24 solution directly.
If A is 2x2 and B is 2x12, C is 2x12.
C = A B.
This is equivalent to 12 independent 2x2 * 2x1 multiplications? No.
B has 2 rows and 12 columns.
C[i, k] = sum_j A[i, j] B[j, k].
For each k (0..11), we compute a 2x1 vector c_k = A * b_k.
The tensor for 2x2 * 2x1 has rank 4.
So 12 * 4 = 48 is the schoolbook rank.
Strassen on 2x2 * 2x2 gives rank 7.
If we group the 12 columns into 6 pairs of 2, we get 6 * 7 = 42.

Can we do better?
The rank of the tensor T_{2,2,12} is the rank of the matrix multiplication tensor.
It is known that rank([2,2,2]) = 7.
rank([2,2,4]) <= 14.
rank([2,2,12]) <= 42.

Is there a known lower bound?
rank >= n*m*p / (n+m+p-2) is not the right bound.
rank >= max(n*m, m*p, p*n) = max(4, 24, 24) = 24.

Let's try to find a rank-24 decomposition.
If rank is 24, it must be "tight" with respect to the m*p and p*n bounds.
This implies that the U, V, W matrices must be "generic" enough to saturate these bounds.

We will attempt a gradient descent search for rank 24, 26, 28, 30, 32, 34, 36, 38, 40.
We use a structured initialization:
- Initialize U, V, W with small random noise.
- Use Adam optimizer if possible (scipy.optimize.minimize with L-BFGS-B or similar).
- Since we need exact integer/half-integer solutions, we will search for real solutions
  and then round them, verifying the exactness.

To make the search more likely to succeed, we will use a "warm start" from the
Strassen-42 solution, but randomly project it down to rank R.
"""

import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    n, m, p = DIMENSIONS

    def time_left():
        return deadline - time.time()

    # Build target tensor T[a,b,c]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T)
        return err < tol

    def make_strassen_42():
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        prods = [
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            ([(2,1), (3,1)], [(0,1)], [(2,1)]),
            ([(0,1)], [(1,1), (3,-1)], [(1,1)]),
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,-1), (3,1)]),
        ]

        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                u_entries, v_entries, w_entries = prods[s]

                for idx, coeff in u_entries:
                    U[idx, col] = coeff

                for idx, coeff in v_entries:
                    j = idx // 2
                    k_loc = idx % 2
                    k = 2 * t + k_loc
                    V[j * p + k, col] = coeff

                for idx, coeff in w_entries:
                    k_loc = idx // 2
                    i = idx % 2
                    k = 2 * t + k_loc
                    W[k * n + i, col] = coeff

        return U, V, W

    # Baseline
    U42, V42, W42 = make_strassen_42()
    if verify(U42, V42, W42):
        best_U, best_V, best_W = U42, V42, W42
        best_rank = 42
    else:
        # Fallback to schoolbook 48
        best_rank = 48
        U_fb = np.zeros((n * m, best_rank))
        V_fb = np.zeros((m * p, best_rank))
        W_fb = np.zeros((p * n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        best_U, best_V, best_W = U_fb, V_fb, W_fb

    # Try to find lower rank using optimization
    # We try ranks from 24 to 40 (even steps)
    # We use a warm start from Strassen-42 by taking random linear combinations of the 42 columns
    # and projecting to rank R.

    np.random.seed(42)

    def flatten_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_params(x):
        size_U = n * m * best_rank
        size_V = m * p * best_rank
        U = x[:size_U].reshape(n * m, best_rank)
        V = x[size_U:size_U + size_V].reshape(m * p, best_rank)
        W = x[size_U + size_V:].reshape(p * n, best_rank)
        return U, V, W

    def objective(x):
        U, V, W = unflatten_params(x)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_rec - T
        return np.sum(diff * diff)

    # We will try to optimize for ranks: 40, 38, 36, 34, 32, 30, 28, 26, 24
    # We stop if we find a valid solution.
    for target_rank in [40, 38, 36, 34, 32, 30, 28, 26, 24]:
        if time_left() < 20:
            break

        if target_rank >= best_rank:
            continue

        # Initialize
        # Strategy: Take the 42 Strassen columns, and form target_rank random linear combinations
        # This preserves the "structure" of the solution.
        if target_rank < 42:
            # Random projection
            proj_matrix = np.random.randn(target_rank, 42)
            U_init = U42 @ proj_matrix.T
            V_init = V42 @ proj_matrix.T
            W_init = W42 @ proj_matrix.T
        else:
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1

        x0 = flatten_params(U_init, V_init, W_init)

        # Optimize
        # Use L-BFGS-B as it handles large vectors well
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success and res.fun < 1e-8:
                U_opt, V_opt, W_opt = unflatten_params(res.x)
                
                # Try to round to half-integers
                # The Strassen solution uses integers. The optimum might be close to integers.
                # We try rounding to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify(U_round, V_round, W_round):
                    best_U, best_V, best_W = U_round, V_round, W_round
                    best_rank = target_rank
                    break
                else:
                    # Try integers
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if verify(U_int, V_int, W_int):
                        best_U, best_V, best_W = U_int, V_int, W_int
                        best_rank = target_rank
                        break
        except Exception:
            pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
