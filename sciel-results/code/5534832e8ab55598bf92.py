"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 138  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 3, 7
    nm, mp, pn = n * m, m * p, p * n  # 9, 21, 21

    # Precompute index maps (row index -> (i,j,k) triple)
    # U rows: a = i*m + j   -> (i, j, -)
    # V rows: b = j*p + k   -> (i, j, k)
    # W rows: c = k*n + i   -> (i, j, k)
    U_idx = [(a // m, a % m, -1) for a in range(nm)]
    V_idx = [(b // p, b % p, b % p) for b in range(mp)]
    W_idx = [(c % n, -1, c // n) for c in range(pn)]

    def tensor_eval(U, V, W):
        """Compute T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] for the 63 nonzero positions.
        Returns (T_vec, T_flat) where T_vec is length 63 in (i,j,k) order,
        T_flat is the full (nm,mp,pn) tensor (sparse)."""
        T = np.zeros((nm, mp, pn), dtype=float)
        # Vectorize over the 63 (i,j,k) triples
        # For term r: T[i*m+j, j*p+k, k*n+i] += U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
        T_vec = np.zeros((n, m, p), dtype=float)
        for r in range(U.shape[1]):
            # u_r is (nm,), v_r is (mp,), w_r is (pn,)
            # Build (i,j,k) contributions:
            # U part: for each (i,j), u_val = U[i*m+j, r]
            # V part: for each (j,k), v_val = V[j*p+k, r]
            # W part: for each (k,i), w_val = W[k*n+i, r]
            # T[i,j,k] += U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Reshape: U_r -> (n,m), V_r -> (m,p), W_r -> (p,n)
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
            # w_r[k,i] = w_r.T[i,k], so T += u_r[i,j] * v_r[j,k] * w_r.T[i,k]
            # = einsum('ij,jk,ik->ijk', u_r, v_r, w_r.T)
            T_vec += np.einsum('ij,jk,ik->ijk', u_r, v_r, w_r.T)
        # Scatter into flat tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = T_vec[i, j, k]
        return T_vec, T

    def verify(U, V, W, tol=1e-7):
        T_vec, _ = tensor_eval(np.array(U), np.array(V), np.array(W))
        return float(np.max(np.abs(T_vec - 1.0))) < tol

    def polish(U, V, W, iters=80, tol=1e-9, deadline=None):
        """Polish a near-solution using alternating least squares (exact per-row solve)."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]

        # Precompute: for each U row a=(i,j), the (mp*pn) matrix B_a where
        # B_a[b,c] = V[b,:] * W[c,:]  (flattened outer of V[b] and W[c])
        # Actually B_a is the same for all a! It's just V ⊗ W reshaped:
        # B[b, c, col] = V[b, col] * W[c, col]  -> shape (mp, pn, r)
        # Then T[a, b, c] = sum_col U[a, col] * B[b, c, col]
        # For fixed a: T[a, :, :] = sum_col U[a, col] * B[:, :, col]
        # Flatten: T[a].flat = (B.reshape(mp*pn, r)) @ U[a, :]
        # So U[a, :] = lstsq(B_mat, T[a].flat)
        B_mat = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, r)  # (mp*pn, r)

        for it in range(iters):
            if deadline is not None and time.time() > deadline:
                break

            # Solve for U
            for a in range(nm):
                T_a = np.zeros(mp * pn)
                # T[a, b, c] is nonzero only when (a,b,c) corresponds to (i,j,k)
                # a = i*m+j, so i=a//m, j=a%m
                i, j = a // m, a % m
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    T_a[b * pn + c] = 1.0
                # U[a, :] = lstsq(B_mat, T_a)
                U[a, :], _, _, _ = np.linalg.lstsq(B_mat, T_a, rcond=None)

            # Solve for V
            # For fixed b=(j,k): T[a, b, c] = sum_col V[b, col] * (U[a, col] * W[c, col])
            # T[:, b, :].flat = (A_mat) @ V[b, :]  where A_mat[a*pn+c, col] = U[a,col]*W[c,col]
            A_mat = (U[:, :, None] * W[None, :, :]).reshape(nm * pn, r)
            for b in range(mp):
                T_b = np.zeros(nm * pn)
                j, k = b // p, b % p
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    T_b[a * pn + c] = 1.0
                V[b, :], _, _, _ = np.linalg.lstsq(A_mat, T_b, rcond=None)

            # Solve for W
            # For fixed c=(k,i): T[a, b, c] = sum_col W[c, col] * (U[a, col] * V[b, col])
            # T[:, :, c].flat = (C_mat) @ W[c, :]  where C_mat[a*mp+b, col] = U[a,col]*V[b,col]
            C_mat = (U[:, :, None] * V[None, :, :]).reshape(nm * mp, r)
            for c in range(pn):
                T_c = np.zeros(nm * mp)
                k, i = c // n, c % n
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    T_c[a * mp + b] = 1.0
                W[c, :], _, _, _ = np.linalg.lstsq(C_mat, T_c, rcond=None)

        T_vec, _ = tensor_eval(U, V, W)
        err = float(np.max(np.abs(T_vec - 1.0)))
        if err < tol:
            return U, V, W, err
        return None

    def gradient_search(r, seed, max_iters=300, deadline=None):
        """Gradient descent with normalization to find rank-r decomposition."""
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)

        # Normalize each column
        for c in range(r):
            nu, nv, nw = np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c])
            if nu > 1e-12 and nv > 1e-12 and nw > 1e-12:
                scale = (nu * nv * nw) ** (1.0 / 3.0)
                U[:, c] /= (nu * scale)
                V[:, c] /= (nv * scale)
                W[:, c] /= (nw * scale)

        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        # Precompute index arrays for gradient computation
        # For each (i,j,k) triple: a=i*m+j, b=j*p+k, c=k*n+i
        triples = [(i, j, k, i * m + j, j * p + k, k * n + i)
                   for i in range(n) for j in range(m) for k in range(p)]

        lr = 0.02
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break

            T_vec, _ = tensor_eval(U, V, W)
            residual = T_vec - 1.0  # (n, m, p)
            err = float(np.max(np.abs(residual)))

            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if err < 1e-9:
                break

            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)

            for c in range(r):
                u_r = U[:, c].reshape(n, m)
                v_r = V[:, c].reshape(m, p)
                w_r = W[:, c].reshape(p, n)
                # grad_U[i,j] = sum_k residual[i,j,k] * v_r[j,k] * w_r[k,i]
                # grad_V[j,k] = sum_i residual[i,j,k] * u_r[i,j] * w_r[k,i]
                # grad_W[k,i] = sum_j residual[i,j,k] * u_r[i,j] * v_r[j,k]
                gw = w_r  # (p, n) = w_r[k, i]
                # grad_U[i,j] = sum_k residual[i,j,k] * v_r[j,k] * w_r[k,i]
                # = for each i,j: dot(residual[i,j,:], v_r[j,:] * w_r[:,i])
                grad_U_c = np.zeros((n, m))
                grad_V_c = np.zeros((m, p))
                grad_W_c = np.zeros((p, n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res = residual[i, j, k]
                            grad_U_c[i, j] += res * v_r[j, k] * w_r[k, i]
                            grad_V_c[j, k] += res * u_r[i, j] * w_r[k, i]
                            grad_W_c[k, i] += res * u_r[i, j] * v_r[j, k]
                grad_U[:, c] = grad_U_c.flatten()
                grad_V[:, c] = grad_V_c.flatten()
                grad_W[:, c] = grad_W_c.flatten()

            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

            # Normalize columns
            for c in range(r):
                nu, nv, nw = np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c])
                if nu > 1e-12 and nv > 1e-12 and nw > 1e-12:
                    scale = (nu * nv * nw) ** (1.0 / 3.0)
                    U[:, c] /= (nu * scale)
                    V[:, c] /= (nv * scale)
                    W[:, c] /= (nw * scale)

        # Check best
        T_vec, _ = tensor_eval(best_U, best_V, best_W)
        best_err = float(np.max(np.abs(T_vec - 1.0)))
        if best_err < 1e-7:
            # Try to polish
            polished = polish(best_U, best_V, best_W, iters=40, tol=1e-9,
                              deadline=deadline)
            if polished is not None:
                return polished[0], polished[1], polished[2]
            return best_U, best_V, best_W
        return None

    # Schoolbook fallback: rank = 63
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_school, V_school, W_school = schoolbook()
    best_result = (U_school, V_school, W_school)
    best_rank = n * m * p

    # Search ranks from 25 to 40 (theoretically interesting range)
    # Lower bound: max(n*p, m*p, n*m) = max(21, 21, 9) = 21
    # Try a few ranks with multiple seeds, time-boxed
    ranks_to_try = list(range(25, 41))
    seeds_per_rank = 4
    time_per_attempt = 1.5  # seconds per gradient attempt

    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        for seed in range(seeds_per_rank):
            if time.time() > deadline - 5:
                break
            result = gradient_search(r, seed=1000 + r * 100 + seed,
                                     max_iters=200,
                                     deadline=deadline - 5)
            if result is not None:
                U_try, V_try, W_try = result
                if verify(U_try, V_try, W_try, tol=1e-7):
                    best_result = (U_try, V_try, W_try)
                    best_rank = r
                    break  # found valid at this rank, move to next rank
        if best_rank < n * m * p:
            # Try to go lower
            continue

    # If we found something below schoolbook, try to polish it further
    if best_rank < n * m * p:
        U_b, V_b, W_b = best_result
        polished = polish(U_b, V_b, W_b, iters=60, tol=1e-9, deadline=deadline - 3)
        if polished is not None:
            best_result = (polished[0], polished[1], polished[2])

    U, V, W = best_result
    # Convert to lists of lists with float entries
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
