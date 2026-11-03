import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 4, 5
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15

    def get_strassen_block(i0, j0, k0):
        """
        Returns 7 term columns for a 2x2x2 Strassen block.
        Each term is a tuple (u_sparse, v_sparse, w_sparse)
        where each sparse is a list of (index, value).
        """
        terms = []

        # Term 1: (A11+A22)(B11+B22) -> contributes to C11, C22
        u1 = [((i0+0)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v1 = [((j0+0)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w1 = [((k0+0)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u1, v1, w1))

        # Term 2: (A21+A22)B11 -> contributes to C21, C22
        u2 = [((i0+1)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v2 = [((j0+0)*p + (k0+0), 1)]
        w2 = [((k0+0)*n + (i0+1), 1), ((k0+1)*n + (i0+1), -1)]
        terms.append((u2, v2, w2))

        # Term 3: A11(B21-B11) -> contributes to C11, C12
        u3 = [((i0+0)*m + (j0+0), 1)]
        v3 = [((j0+0)*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)]
        w3 = [((k0+1)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u3, v3, w3))

        # Term 4: A22(B11-B22) -> contributes to C11, C12
        u4 = [((i0+1)*m + (j0+1), 1)]
        v4 = [((j0+1)*p + (k0+0), 1), ((j0+0)*p + (k0+0), -1)]
        w4 = [((k0+0)*n + (i0+0), 1), ((k0+0)*n + (i0+1), 1)]
        terms.append((u4, v4, w4))

        # Term 5: (A11+A12)B22 -> contributes to C11, C12
        u5 = [((i0+0)*m + (j0+0), 1), ((i0+0)*m + (j0+1), 1)]
        v5 = [((j0+1)*p + (k0+1), 1)]
        w5 = [((k0+0)*n + (i0+0), -1), ((k0+1)*n + (i0+0), 1)]
        terms.append((u5, v5, w5))

        # Term 6: (A21-A11)(B11+B12) -> contributes to C21
        u6 = [((i0+1)*m + (j0+0), 1), ((i0+0)*m + (j0+0), -1)]
        v6 = [((j0+0)*p + (k0+0), 1), ((j0+0)*p + (k0+1), 1)]
        w6 = [((k0+1)*n + (i0+1), 1)]
        terms.append((u6, v6, w6))

        # Term 7: (A12-A22)(B21+B22) -> contributes to C21
        u7 = [((i0+0)*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)]
        v7 = [((j0+1)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w7 = [((k0+0)*n + (i0+0), 1)]
        terms.append((u7, v7, w7))

        return terms

    def sparse_to_dense(sparse, dim):
        arr = np.zeros(dim)
        for idx, val in sparse:
            arr[idx] = val
        return arr

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_sparse, v_sparse, w_sparse):
        U_cols.append(sparse_to_dense(u_sparse, dim_u))
        V_cols.append(sparse_to_dense(v_sparse, dim_v))
        W_cols.append(sparse_to_dense(w_sparse, dim_w))

    def add_schoolbook(i, j, k):
        u = np.zeros(dim_u); u[i*m + j] = 1
        v = np.zeros(dim_v); v[j*p + k] = 1
        w = np.zeros(dim_w); w[k*n + i] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

    covered = set()

    # Four Strassen blocks covering i in {0,1}, all j in {0,1,2,3}, all k in {0,1,2,3}
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                terms = get_strassen_block(i0, j0, k0)
                for u, v, w in terms:
                    add_term(u, v, w)
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i0+di, j0+dj, k0+dk))

    # Schoolbook for the rest (i=2, all j, all k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                add_schoolbook(i, j, k)

    # Stack columns
    if U_cols:
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
    else:
        U = np.zeros((dim_u, 1))
        V = np.zeros((dim_v, 1))
        W = np.zeros((dim_w, 1))

    rank = U.shape[1]

    def verify(U, V, W, tol=1e-9):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    s = float(np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :]))
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    max_err = verify(U, V, W)

    if max_err > 1e-9:
        # Fallback to pure schoolbook
        U_sb = np.zeros((dim_u, n*m*p))
        V_sb = np.zeros((dim_v, n*m*p))
        W_sb = np.zeros((dim_w, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        U, V, W = U_sb, V_sb, W_sb
        rank = n*m*p

    # Greedy column removal: try removing each column and re-verify
    if time.time() < deadline - 15:
        kept = list(range(rank))
        improved = True
        while improved:
            improved = False
            for r in range(len(kept)):
                if time.time() > deadline - 10:
                    break
                new_kept = kept[:r] + kept[r+1:]
                U_trial = U[:, new_kept]
                V_trial = V[:, new_kept]
                W_trial = W[:, new_kept]
                if verify(U_trial, V_trial, W_trial, tol=1e-9) <= 1e-9:
                    kept = new_kept
                    U = U_trial
                    V = V_trial
                    W = W_trial
                    rank = len(kept)
                    improved = True
                    break

    # Now try ALS-based rank reduction from the current rank downward
    # This uses the inspiration's alternating least squares approach
    best_U, best_V, best_W = U, V, W
    best_rank = rank

    if time.time() < deadline - 30:
        # Try to find a lower rank solution using ALS
        for target_rank in range(best_rank - 1, max(30, best_rank - 15), -1):
            if time.time() > deadline - 25:
                break
            
            # Try a few random initializations
            max_trials = 5 if time.time() < deadline - 40 else 2
            for trial in range(max_trials):
                if time.time() > deadline - 20:
                    break
                
                # Initialize from current solution with some perturbation
                if trial == 0:
                    # Use a random subset of current columns as initialization
                    if best_rank >= target_rank:
                        cols = np.random.choice(best_rank, size=target_rank, replace=False)
                        U_init = best_U[:, cols].copy()
                        V_init = best_V[:, cols].copy()
                        W_init = best_W[:, cols].copy()
                    else:
                        U_init = np.random.randn(dim_u, target_rank) * 0.1
                        V_init = np.random.randn(dim_v, target_rank) * 0.1
                        W_init = np.random.randn(dim_w, target_rank) * 0.1
                else:
                    U_init = np.random.randn(dim_u, target_rank) * 0.5
                    V_init = np.random.randn(dim_v, target_rank) * 0.5
                    W_init = np.random.randn(dim_w, target_rank) * 0.5
                
                U_als = U_init.copy()
                V_als = V_init.copy()
                W_als = W_init.copy()
                
                converged = False
                for iteration in range(100):
                    if time.time() > deadline - 15:
                        break
                    
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            A = np.zeros((p, target_rank))
                            for k in range(p):
                                for c in range(target_rank):
                                    A[k, c] = V_als[j*p+k, c] * W_als[k*n+i, c]
                            try:
                                ATA = A.T @ A
                                ATb = A.T @ np.ones(p)
                                U_als[i*m+j, :] = np.linalg.solve(ATA + 1e-12*np.eye(target_rank), ATb)
                            except:
                                pass
                    
                    # Update V
                    for j in range(m):
                        for k in range(p):
                            A = np.zeros((n, target_rank))
                            for i in range(n):
                                for c in range(target_rank):
                                    A[i, c] = U_als[i*m+j, c] * W_als[k*n+i, c]
                            try:
                                ATA = A.T @ A
                                ATb = A.T @ np.ones(n)
                                V_als[j*p+k, :] = np.linalg.solve(ATA + 1e-12*np.eye(target_rank), ATb)
                            except:
                                pass
                    
                    # Update W
                    for k in range(p):
                        for i in range(n):
                            A = np.zeros((m, target_rank))
                            for j in range(m):
                                for c in range(target_rank):
                                    A[j, c] = U_als[i*m+j, c] * V_als[j*p+k, c]
                            try:
                                ATA = A.T @ A
                                ATb = A.T @ np.ones(m)
                                W_als[k*n+i, :] = np.linalg.solve(ATA + 1e-12*np.eye(target_rank), ATb)
                            except:
                                pass
                    
                    # Check convergence
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = float(np.dot(U_als[i*m+j, :], V_als[j*p+k, :] * W_als[k*n+i, :]))
                                error += (1.0 - val) ** 2
                    
                    if error < 1e-14:
                        converged = True
                        break
                    elif error < 1e-10:
                        # Try a few more iterations
                        for _ in range(20):
                            if time.time() > deadline - 12:
                                break
                            for i in range(n):
                                for j in range(m):
                                    A = np.zeros((p, target_rank))
                                    for k in range(p):
                                        for c in range(target_rank):
                                            A[k, c] = V_als[j*p+k, c] * W_als[k*n+i, c]
                                    try:
                                        ATA = A.T @ A
                                        ATb = A.T @ np.ones(p)
                                        U_als[i*m+j, :] = np.linalg.solve(ATA + 1e-14*np.eye(target_rank), ATb)
                                    except:
                                        pass
                            for j in range(m):
                                for k in range(p):
                                    A = np.zeros((n, target_rank))
                                    for i in range(n):
                                        for c in range(target_rank):
                                            A[i, c] = U_als[i*m+j, c] * W_als[k*n+i, c]
                                    try:
                                        ATA = A.T @ A
                                        ATb = A.T @ np.ones(n)
                                        V_als[j*p+k, :] = np.linalg.solve(ATA + 1e-14*np.eye(target_rank), ATb)
                                    except:
                                        pass
                            for k in range(p):
                                for i in range(n):
                                    A = np.zeros((m, target_rank))
                                    for j in range(m):
                                        for c in range(target_rank):
                                            A[j, c] = U_als[i*m+j, c] * V_als[j*p+k, c]
                                    try:
                                        ATA = A.T @ A
                                        ATb = A.T @ np.ones(m)
                                        W_als[k*n+i, :] = np.linalg.solve(ATA + 1e-14*np.eye(target_rank), ATb)
                                    except:
                                        pass
                            error = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        val = float(np.dot(U_als[i*m+j, :], V_als[j*p+k, :] * W_als[k*n+i, :]))
                                        error += (1.0 - val) ** 2
                            if error < 1e-14:
                                converged = True
                                break
                        break
                    elif error > 1.0:
                        break  # Diverged
                
                if converged:
                    # Verify with strict tolerance
                    if verify(U_als, V_als, W_als, tol=1e-9) <= 1e-9:
                        best_U, best_V, best_W = U_als, V_als, W_als
                        best_rank = target_rank
                        break  # Break out of trial loop
                else:
                    # Even if not fully converged, check if close enough
                    if verify(U_als, V_als, W_als, tol=1e-9) <= 1e-9:
                        best_U, best_V, best_W = U_als, V_als, W_als
                        best_rank = target_rank
                        break

    U, V, W = best_U, best_V, best_W
    rank = best_rank

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
