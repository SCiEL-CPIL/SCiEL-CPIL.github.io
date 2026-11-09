<<<<<<< SEARCH
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 6, 24, 16

    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    best_rank = n * m * p  # schoolbook = 48
    best_U = None
    best_V = None
    best_W = None

    # Schoolbook decomposition
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Winograd-style approach: treat the p=8 dimension as a convolution
    # and try to find a low-rank decomposition using structured transforms.
    # For a 2x3x8 multiplication, we can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a set of 2*8 = 16 output values, each a sum over 3 terms.
    #
    # The key insight: if we can find a transform on the k-dimension that
    # diagonalizes or simplifies the multiplication, we can reduce the rank.
    #
    # Let's try a direct approach: search for rank R decompositions using
    # a structured initialization based on Winograd transforms.
    #
    # For a 1x1x8 convolution (simplest case), Winograd uses a 4-point transform
    # to reduce from 8 multiplications to 4. We can try similar ideas here.
    #
    # Actually, let's try a more systematic search. The matrix multiplication
    # tensor for 2x3x8 has a specific structure. Let's try to find the
    # minimum rank by searching over possible ranks.

    # Try ranks from a reasonable lower bound up to schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # But for matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # Actually, for M[n,m] x M[m,p], the rank is at least max(n*m, m*p, p*n)
    # which is 24 here. Let's try from 24 upward.

    # Use a smarter search: try to find exact half-integer solutions
    for R in range(24, best_rank + 1):
        if time.time() > deadline:
            break

        # Try multiple random initializations with different strategies
        found = False
        for trial in range(30):
            if time.time() > deadline:
                break

            # Different initialization strategies
            if trial < 10:
                # Random initialization
                np.random.seed(42 + R * 100 + trial)
                U = np.random.randn(nm, R)
                V = np.random.randn(mp, R)
                W = np.random.randn(pn, R)
            elif trial < 20:
                # Structured initialization based on Winograd-like patterns
                # Try to initialize with block structure
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                # For each (i,j,k) triple, try to share rank components
                # This is heuristic - try to create overlapping patterns
                np.random.seed(42 + R * 100 + trial)
                for r in range(R):
                    # Randomly select which (i,j,k) triples share this rank
                    for i in range(n):
                        for j in range(m):
                            if np.random.random() < 0.5:
                                U[i * m + j, r] = np.random.choice([-1, 0, 1])
                        for k in range(p):
                            if np.random.random() < 0.5:
                                V[j * p + k, r] = np.random.choice([-1, 0, 1])
                            if np.random.random() < 0.5:
                                W[k * n + i, r] = np.random.choice([-1, 0, 1])
            else:
                # Sparse initialization
                np.random.seed(42 + R * 100 + trial)
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    # Each rank component touches a few entries
                    n_u = np.random.randint(1, 4)
                    n_v = np.random.randint(1, 4)
                    n_w = np.random.randint(1, 4)
                    u_idx = np.random.choice(nm, n_u, replace=False)
                    v_idx = np.random.choice(mp, n_v, replace=False)
                    w_idx = np.random.choice(pn, n_w, replace=False)
                    U[u_idx, r] = np.random.choice([-1, 1], n_u)
                    V[v_idx, r] = np.random.choice([-1, 1], n_v)
                    W[w_idx, r] = np.random.choice([-1, 1], n_w)

            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() > deadline:
                    break

                # Update U
                for i in range(n):
                    for j in range(m):
                        M_mat = np.zeros((p, R))
                        for k in range(p):
                            M_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        x, residuals, rank_m, sv = np.linalg.lstsq(M_mat, b, rcond=None)
                        U[i * m + j, :] = x

                # Update V
                for j in range(m):
                    for k in range(p):
                        M_mat = np.zeros((n, R))
                        for i in range(n):
                            M_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        x, residuals, rank_m, sv = np.linalg.lstsq(M_mat, b, rcond=None)
                        V[j * p + k, :] = x

                # Update W
                for i in range(n):
                    for k in range(p):
                        M_mat = np.zeros((m, R))
                        for j in range(m):
                            M_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        x, residuals, rank_m, sv = np.linalg.lstsq(M_mat, b, rcond=None)
                        W[k * n + i, :] = x

                # Check convergence
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            residual += abs(val - 1.0)
                if residual < 1e-6:
                    converged = True
                    break

            if converged:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2

                # Verify
                valid = True
                for ii in range(nm):
                    for jj in range(mp):
                        for kk in range(pn):
                            val = np.sum(U_r[ii, :] * V_r[jj, :] * W_r[kk, :])
                            expected = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        if ii == i * m + j and jj == j * p + k and kk == k * n + i:
                                            expected = 1.0
                            if abs(val - expected) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break

                if valid:
                    best_rank = R
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break

        if found:
            break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
=======
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 6, 24, 16

    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    best_rank = n * m * p  # schoolbook = 48
    best_U = None
    best_V = None
    best_W = None

    # Schoolbook decomposition
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def verify_decomposition(U, V, W, tol=1e-9):
        """Fast verification using tensor contraction."""
        # Contract: result[ii,jj,kk] = sum_r U[ii,r]*V[jj,r]*W[kk,r]
        # We only need to check the 48 non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check a few zero entries to ensure no spurious contributions
        # (sample-based check for speed)
        np.random.seed(12345)
        for _ in range(20):
            ii = np.random.randint(nm)
            jj = np.random.randint(mp)
            kk = np.random.randint(pn)
            # Check if this is a valid (i,j,k) triple
            valid = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if ii == i*m+j and jj == j*p+k and kk == k*n+i:
                            valid = True
            if not valid:
                val = np.sum(U[ii, :] * V[jj, :] * W[kk, :])
                if abs(val) > tol:
                    return False
        return True

    def check_half_integer(U, V, W):
        """Check if rounding to half-integers gives exact decomposition."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify_decomposition(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    # Try ranks from a reasonable lower bound up to schoolbook
    for R in range(24, best_rank + 1):
        if time.time() > deadline:
            break

        found = False
        for trial in range(40):
            if time.time() > deadline:
                break

            np.random.seed(42 + R * 1000 + trial * 7)

            if trial < 12:
                # Random Gaussian initialization
                U = np.random.randn(nm, R)
                V = np.random.randn(mp, R)
                W = np.random.randn(pn, R)
            elif trial < 24:
                # Structured: exploit the block structure of matrix mult
                # U has shape (2*3, R), V has shape (3*8, R), W has shape (8*2, R)
                # For each rank component, try to have structure in the j and k dimensions
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    # Choose a pattern for this rank component
                    # U: for each (i,j), assign value
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, r] = np.random.choice([-1, 0, 1])
                    # V: for each (j,k), assign value - try to be sparse in k
                    for j in range(m):
                        for k in range(p):
                            if np.random.random() < 0.6:
                                V[j*p+k, r] = np.random.choice([-1, 1])
                    # W: for each (k,i), assign value
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, r] = np.random.choice([-1, 0, 1])
            elif trial < 32:
                # Winograd-inspired: use small integer patterns
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    # Pattern 1: U depends only on j, V on (j,k), W on (k,i)
                    u_vals = np.random.choice([-1, 0, 1], size=n*m)
                    for idx in range(n*m):
                        U[idx, r] = u_vals[idx]
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k, r] = np.random.choice([-1, 0, 1])
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, r] = np.random.choice([-1, 0, 1])
            else:
                # Sparse with varied density
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    density = np.random.uniform(0.3, 0.7)
                    for idx in range(nm):
                        if np.random.random() < density:
                            U[idx, r] = np.random.choice([-1, 1])
                    for idx in range(mp):
                        if np.random.random() < density:
                            V[idx, r] = np.random.choice([-1, 1])
                    for idx in range(pn):
                        if np.random.random() < density:
                            W[idx, r] = np.random.choice([-1, 1])

            # ALS iterations with early stopping
            converged = False
            prev_residual = float('inf')
            for it in range(150):
                if time.time() > deadline:
                    break

                # Update U: for each (i,j), solve min ||V[j,:] * W[i,:]^T * x - 1||
                for i in range(n):
                    for j in range(m):
                        # M_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        M_mat = (V[j*p:(j+1)*p, :] * W.reshape(p, n, 2)[:, i, :].T).T  # (p, R)
                        # Actually let's be more careful
                        M_mat = np.zeros((p, R))
                        for k in range(p):
                            M_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        x, _, _, _ = np.linalg.lstsq(M_mat, b, rcond=None)
                        U[i*m+j, :] = x

                # Update V
                for j in range(m):
                    for k in range(p):
                        M_mat = np.zeros((n, R))
                        for i in range(n):
                            M_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        x, _, _, _ = np.linalg.lstsq(M_mat, b, rcond=None)
                        V[j*p+k, :] = x

                # Update W
                for i in range(n):
                    for k in range(p):
                        M_mat = np.zeros((m, R))
                        for j in range(m):
                            M_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        x, _, _, _ = np.linalg.lstsq(M_mat, b, rcond=None)
                        W[k*n+i, :] = x

                # Check convergence via tensor residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            residual += abs(val - 1.0)
                
                if residual < 1e-8:
                    converged = True
                    break
                if abs(prev_residual - residual) < 1e-10:
                    converged = True
                    break
                prev_residual = residual

            if converged:
                result = check_half_integer(U, V, W)
                if result is not None:
                    U_r, V_r, W_r = result
                    best_rank = R
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
                # Also try without rounding - maybe it's already close enough
                # and we can try a polish step
                if verify_decomposition(U, V, W, tol=1e-6):
                    # Try rounding with different offsets
                    for offset in [0.0, 0.25, 0.5, 0.75]:
                        U_r = np.round((U + offset) * 2) / 2 - offset
                        V_r = np.round((V + offset) * 2) / 2 - offset
                        W_r = np.round((W + offset) * 2) / 2 - offset
                        if verify_decomposition(U_r, V_r, W_r):
                            best_rank = R
                            best_U = U_r
                            best_V = V_r
                            best_W = W_r
                            found = True
                            break
                    if found:
                        break

        if found:
            break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
>>>>>>> REPLACE
