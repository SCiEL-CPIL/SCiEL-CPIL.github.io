"""Combined: structured rank-reduction via exhaustive search for small candidate ranks, with Strassen baseline."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    N = n * m  # 8
    M = m * p  # 16
    P = p * n  # 8

    def verify_decomp(U, V, W, n, m, p):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(
                        U[i * m + j, :] * V[j * p + k, :],
                        W[k * n + i, :]
                    ))
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def strassen_2x2_decomp(P_rows, P_cols, Q_rows, Q_cols, R_rows, R_cols,
                            base_col, U, V, W):
        u11 = P_rows[0] * m + P_cols[0]
        u12 = P_rows[0] * m + P_cols[1]
        u21 = P_rows[1] * m + P_cols[0]
        u22 = P_rows[1] * m + P_cols[1]
        v11 = Q_rows[0] * p + Q_cols[0]
        v12 = Q_rows[0] * p + Q_cols[1]
        v21 = Q_rows[1] * p + Q_cols[0]
        v22 = Q_rows[1] * p + Q_cols[1]
        w11 = R_cols[0] * n + R_rows[0]
        w12 = R_cols[1] * n + R_rows[0]
        w21 = R_cols[0] * n + R_rows[1]
        w22 = R_cols[1] * n + R_rows[1]

        c = base_col + 0
        U[u11, c] += 1; U[u22, c] += 1; V[v11, c] += 1; V[v22, c] += 1; W[w11, c] += 1; W[w22, c] += 1
        c = base_col + 1
        U[u21, c] += 1; U[u22, c] += 1; V[v11, c] += 1; W[w21, c] += 1; W[w22, c] += -1
        c = base_col + 2
        U[u11, c] += 1; V[v12, c] += 1; V[v22, c] += -1; W[w12, c] += 1; W[w22, c] += 1
        c = base_col + 3
        U[u22, c] += 1; V[v21, c] += 1; V[v11, c] += -1; W[w11, c] += 1; W[w21, c] += 1
        c = base_col + 4
        U[u11, c] += 1; U[u12, c] += 1; V[v22, c] += 1; W[w11, c] += -1; W[w12, c] += 1
        c = base_col + 5
        U[u21, c] += 1; U[u11, c] += -1; V[v11, c] += 1; V[v12, c] += 1; W[w22, c] += 1
        c = base_col + 6
        U[u12, c] += 1; U[u22, c] += -1; V[v21, c] += 1; V[v22, c] += 1; W[w11, c] += 1; W[w22, c] += -1
        return base_col + 7

    # Build Strassen rank-28 baseline
    rank_28 = 28
    U28 = np.zeros((N, rank_28), dtype=float)
    V28 = np.zeros((M, rank_28), dtype=float)
    W28 = np.zeros((P, rank_28), dtype=float)

    col = strassen_2x2_decomp([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 0, U28, V28, W28)
    col = strassen_2x2_decomp([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1], col, U28, V28, W28)
    col = strassen_2x2_decomp([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], col, U28, V28, W28)
    col = strassen_2x2_decomp([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3], col, U28, V28, W28)

    best = None
    if verify_decomp(U28, V28, W28, n, m, p):
        best = (U28.copy(), V28.copy(), W28.copy())

    # ---- Exhaustive search for lower ranks using integer/half-integer entries ----
    # For small ranks, we can try to find valid decompositions by searching over
    # the space of possible entries.

    def search_rank(R, max_val=2, deadline_check=None):
        """Search for a valid decomposition of rank R with entries in {-max_val,...,max_val}/2.
        Uses a greedy approach: build columns one at a time, checking partial validity."""
        if deadline_check and time.time() > deadline_check:
            return None

        # For rank R, we need U (N x R), V (M x R), W (P x R)
        # We'll use a randomized local search approach

        # Start from the Strassen decomposition, zero out extra columns, and try to repair
        if R >= 28:
            # Just use Strassen (already found)
            return None

        # Try random initialization with local search
        np.random.seed(42)
        for attempt in range(50):
            if deadline_check and time.time() > deadline_check:
                return None

            # Initialize with small random values
            U = np.zeros((N, R), dtype=float)
            V = np.zeros((M, R), dtype=float)
            W = np.zeros((P, R), dtype=float)

            # Try to build a valid decomposition column by column
            # Each column r contributes a rank-1 tensor
            # We need sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T (the multiplication tensor)

            # For the multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
            # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

            # This is a system of 32 equations (2*4*4) in 3*R unknowns (with structure)
            # For small R, this might be solvable

            # Let's try a structured approach: use the Strassen-like block structure
            # but with fewer columns by combining

            # Actually, let's try a different approach: use the known lower bounds
            # and try to find a decomposition via linear algebra

            # For rank R, the multiplication tensor T can be written as:
            # T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
            # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}

            # The multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k
            # (since C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j]*B[j,k] is 1)

            # Wait, that's not right. Let me re-examine.
            # The tensor T is defined such that:
            # C[i,k] = sum_{j,r} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
            # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
            # (since A[i,j] and B[j,k] are arbitrary)

            # So the condition is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

            # This is a system of 32 equations. For rank R, we have 3*R*max(1, R) variables
            # (actually 3*R variables per column, but they're coupled)

            # Let's try a simple approach: for small R, try all possible assignments
            # of {-1, 0, 1} to the entries and check if the system is satisfied

            # For R=20, that's 3*20=60 variables, each in {-1,0,1}, so 3^60 possibilities - too many
            # But we can use the structure: the equations are linear in the products

            # Actually, let's try a greedy approach: fix U and V, solve for W
            # Or use the fact that for each (i,j,k), the equation is linear in W

            # For fixed U and V, the system for W is:
            # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
            # This is a system of 32 equations in 8*R unknowns (W has 8 rows)

            # If R >= 4, we have 8*R >= 32 unknowns, so the system might be solvable

            # Let's try: for each (k*n+i), we have R unknowns W[k*n+i, 0..R-1]
            # The equations involving W[k*n+i, r] are: for all j,
            #   sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1

            # So for fixed (i,k), we have m equations in R unknowns:
            #   sum_r W[i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for j=0..m-1
            # where W[i, r] = W[k*n+i, r]

            # This is a linear system: A_{i,k} * W_{i,k} = 1
            # where A_{i,k}[j, r] = U[i*m+j, r] * V[j*p+k, r]

            # So we can:
            # 1. Choose U and V
            # 2. For each (i,k), solve the linear system for W[i,k,:]
            # 3. Check if the solution is valid (and has nice entries)

            # Let's try this approach with random U and V

            for _ in range(100):
                if deadline_check and time.time() > deadline_check:
                    return None

                # Random U and V with entries in {-1, 0, 1}
                U = np.random.choice([-1, 0, 1], size=(N, R))
                V = np.random.choice([-1, 0, 1], size=(M, R))

                # For each (i,k), solve for W[i,k,:]
                W_sol = np.zeros((P, R), dtype=float)
                valid = True
                for i in range(n):
                    for k in range(p):
                        # Build the system: A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        A = np.zeros((m, R), dtype=float)
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        # Solve A * W_col = 1
                        try:
                            W_col, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            # Check if the solution is exact
                            if np.max(np.abs(A @ W_col - np.ones(m))) > 1e-8:
                                valid = False
                                break
                            W_sol[k*n+i, :] = W_col
                        except:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    # Check the full decomposition
                    if verify_decomp(U, V, W_sol, n, m, p):
                        return U, V, W_sol

        return None

    # Try ranks from 20 to 27
    for R in range(20, 28):
        if time.time() > deadline - 10:
            break
        result = search_rank(R, deadline_check=deadline - 5)
        if result is not None:
            Uc, Vc, Wc = result
            if best is None or Uc.shape[1] < best[0].shape[1]:
                best = (Uc, Vc, Wc)
            break  # Found a lower rank, no need to continue

    # ---- Finalize ----
    if best is None:
        # Schoolbook fallback
        rank = n * m * p
        Us = np.zeros((N, rank))
        Vs = np.zeros((M, rank))
        Ws = np.zeros((P, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j, c] = 1
                    Vs[j * p + k, c] = 1
                    Ws[k * n + i, c] = 1
                    c += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    Us, Vs, Ws = best
    def clean(A):
        A = np.asarray(A, dtype=float)
        A = np.round(A * 2) / 2.0
        return A.tolist()

    return {"U": clean(Us), "V": clean(Vs), "W": clean(Ws)}
# EVOLVE-BLOCK-END
