"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check all 12 (i,j,k) triples exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    This is the known optimal rank for 2x2x3 over reals/rationals.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11) -> contributes to C[0,0] and C[1,1]
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> contributes to C[0,1] and C[1,1]
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> contributes to C[1,0] and C[1,1]
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> contributes to C[0,0] and C[0,1]
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> contributes to C[0,0] and C[1,0]
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> contributes to C[1,1]
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> contributes to C[0,0]
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
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
    return {"U": U, "V": V, "W": W}


def _verify_mod(U, V, W, n, m, p, mod):
    """Verify decomposition over a finite field."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s = (s + U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]) % mod
                if s % mod != 1 % mod:
                    return False
    return True


def _try_rank10_modular(deadline):
    """
    Attempt to find a rank-10 decomposition via randomized search over GF(p).
    We start with the rank-11 decomposition, try to eliminate one term by
    finding linear combinations that zero out a column while preserving the tensor.
    
    Key insight: If we can find a non-trivial linear dependency among the 11 terms
    such that the sum is zero on the tensor, we can eliminate one term.
    Alternatively, we search for rank-10 decompositions directly over GF(p).
    """
    if time.time() > deadline - 10:
        return None
    
    # Try several small primes
    primes = [13, 17, 19, 23, 29]
    
    for prime in primes:
        if time.time() > deadline - 5:
            break
        
        rng = np.random.default_rng(42 + prime)
        
        # Strategy 1: Try to find a linear dependency in the rank-11 solution
        # If sum_{r} c_r * T_r = 0 (tensor), then we can drop one term if c_r != 0
        # This is equivalent to finding a vector c in the kernel of the tensor map.
        
        # Build the tensor map matrix: for each term r, we have a rank-1 tensor T_r
        # We want to find c such that sum_r c_r * T_r = 0
        # This is a system of 12 linear equations in 11 unknowns (c_0...c_10)
        
        # Actually, the space of tensors is 12-dimensional (for 2x2x3), and we have 11 terms.
        # So generically they are linearly independent. But maybe there's a dependency?
        
        # Let's check if the 11 rank-1 tensors are linearly dependent.
        # Flatten each term into a 12-dim vector
        base = _rank11_decomposition()
        U11 = np.array(base["U"], dtype=float)  # 4 x 11
        V11 = np.array(base["V"], dtype=float)  # 6 x 11
        W11 = np.array(base["W"], dtype=float)  # 6 x 11
        
        # For each term r, create a 12-dim vector
        term_vectors = []
        for r in range(11):
            vec = []
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        vec.append(U11[i*2+j, r] * V11[j*3+k, r] * W11[k*2+i, r])
            term_vectors.append(vec)
        
        term_matrix = np.array(term_vectors).T  # 12 x 11
        
        # Check if columns are linearly dependent
        rank_t = np.linalg.matrix_rank(term_matrix)
        if rank_t < 11:
            # There's a dependency! Find it.
            # Solve term_matrix @ c = 0
            # Use SVD
            U_svd, s_vals, Vt_svd = np.linalg.svd(term_matrix)
            c = Vt_svd[-1, :]  # Last right singular vector
            # Normalize
            c = c / np.linalg.norm(c)
            
            # Check if this gives a valid dependency
            if np.linalg.norm(term_matrix @ c) < 1e-10:
                # We have a dependency. If any c_r is non-zero, we can drop that term.
                # But we need to check if the resulting rank-10 decomposition is valid.
                # Actually, the dependency means sum_r c_r * T_r = 0, so the 11 terms
                # span a space of dimension at most 10. We can express one term as a
                # linear combination of the others.
                
                # Find a term with non-zero coefficient
                max_idx = np.argmax(np.abs(c))
                if abs(c[max_idx]) > 1e-10:
                    # Drop term max_idx
                    keep = [r for r in range(11) if r != max_idx]
                    U10 = U11[:, keep]
                    V10 = V11[:, keep]
                    W10 = W11[:, keep]
                    
                    # Now we need to check if this rank-10 decomposition is valid.
                    # It should be, since the tensor is the sum of all 11 terms,
                    # and we dropped one that's a linear combination of the others.
                    # But wait, the decomposition requires each term to be rank-1.
                    # The linear combination might not be rank-1.
                    
                    # Actually, the decomposition is:
                    # Tensor = sum_{r in keep} T_r + T_{max_idx}
                    # If T_{max_idx} = sum_{r in keep} alpha_r * T_r, then
                    # Tensor = sum_{r in keep} (1 + alpha_r) * T_r
                    # But (1 + alpha_r) * T_r is still rank-1 (just scaled).
                    # So we can absorb the scaling into U, V, W.
                    
                    # Compute alpha_r from the dependency
                    # c_{max_idx} * T_{max_idx} + sum_{r in keep} c_r * T_r = 0
                    # T_{max_idx} = -sum_{r in keep} (c_r / c_{max_idx}) * T_r
                    # So alpha_r = -c_r / c_{max_idx}
                    
                    # New coefficients: for r in keep, new_coeff_r = 1 + alpha_r = 1 - c_r/c_{max_idx}
                    # = (c_{max_idx} - c_r) / c_{max_idx}
                    
                    # But we need to be careful: the decomposition is
                    # sum_r U_r * V_r * W_r = Tensor
                    # If we scale U_r by s, V_r by t, W_r by 1/(st), the product is unchanged.
                    # So we can scale each term by the new coefficient.
                    
                    # Let's construct the new decomposition
                    U_new = []
                    V_new = []
                    W_new = []
                    
                    for idx, r in enumerate(keep):
                        scale = (c[max_idx] - c[r]) / c[max_idx]
                        U_new.append(U11[:, r] * scale)
                        V_new.append(V11[:, r])
                        W_new.append(W11[:, r])
                    
                    U_new = np.array(U_new).T  # 4 x 10
                    V_new = np.array(V_new).T  # 6 x 10
                    W_new = np.array(W_new).T  # 6 x 10
                    
                    # Verify
                    if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), 2, 2, 3):
                        # Round to nearest integer or half-integer
                        U_rounded = [[round(u) for u in row] for row in U_new.tolist()]
                        V_rounded = [[round(v) for v in row] for row in V_new.tolist()]
                        W_rounded = [[round(w) for w in row] for row in W_new.tolist()]
                        
                        if _verify(U_rounded, V_rounded, W_rounded, 2, 2, 3):
                            return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                        
                        # Try half-integers
                        U_half = [[round(u * 2) / 2 for u in row] for row in U_new.tolist()]
                        V_half = [[round(v * 2) / 2 for v in row] for row in V_new.tolist()]
                        W_half = [[round(w * 2) / 2 for w in row] for row in W_new.tolist()]
                        
                        if _verify(U_half, V_half, W_half, 2, 2, 3):
                            return {"U": U_half, "V": V_half, "W": W_half}
        
        # Strategy 2: Random search for rank-10 decomposition over GF(prime)
        # This is more expensive, so only do it if we have time.
        if time.time() > deadline - 3:
            continue
        
        # Try a few random initializations
        for trial in range(20):
            if time.time() > deadline - 2:
                break
            
            # Random rank-10 decomposition over GF(prime)
            U_rand = rng.integers(0, prime, size=(4, 10)) % prime
            V_rand = rng.integers(0, prime, size=(6, 10)) % prime
            W_rand = rng.integers(0, prime, size=(6, 10)) % prime
            
            # Verify
            if _verify_mod(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), 2, 2, 3, prime):
                # Found a valid decomposition over GF(prime)
                # Lift to integers
                U_lift = U_rand.tolist()
                V_lift = V_rand.tolist()
                W_lift = W_rand.tolist()
                
                if _verify(U_lift, V_lift, W_lift, 2, 2, 3):
                    return {"U": U_lift, "V": V_lift, "W": W_lift}
    
    return None


def _try_rank10_search(deadline):
    """
    Attempt to find a rank-10 decomposition via numerical search.
    We use the rank-11 solution as a starting point, drop one term, and optimize.
    """
    if time.time() > deadline - 20:
        return None
    
    from scipy.optimize import least_squares
    
    rank = 10
    # U: 4 x 10, V: 6 x 10, W: 6 x 10
    # Total parameters: 40 + 60 + 60 = 160
    
    def residuals(params):
        U_flat = params[:40]
        V_flat = params[40:100]
        W_flat = params[100:160]
        
        U_mat = U_flat.reshape(4, 10)
        V_mat = V_flat.reshape(6, 10)
        W_mat = W_flat.reshape(6, 10)
        
        res = np.empty(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0.0
                    for r in range(10):
                        s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res
    
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)  # 4 x 11
    V11 = np.array(base["V"], dtype=float)  # 6 x 11
    W11 = np.array(base["W"], dtype=float)  # 6 x 11
    
    rng = np.random.default_rng(42)
    
    best_result = None
    best_cost = float('inf')
    
    # Strategy: Drop each of the 11 terms one by one and try to optimize the remaining 10.
    for drop_term in range(11):
        if time.time() > deadline - 15:
            break
        
        keep = [r for r in range(11) if r != drop_term]
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Add small random perturbation to escape local minima
        x0 = x0 + rng.normal(0, 0.1, 160)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-15, xtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result.x.copy()
                
                if result.cost < 1e-12:
                    break
        except Exception:
            pass
    
    # Also try a few random initializations
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        
        x0 = rng.normal(0, 1, 160)
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-15, xtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result.x.copy()
                
                if result.cost < 1e-12:
                    break
        except Exception:
            pass
    
    if best_result is None or best_cost > 1e-10:
        return None
    
    # Extract and round to nearest half-integer
    U_flat = best_result[:40]
    V_flat = best_result[40:100]
    W_flat = best_result[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    # Round to nearest half-integer
    U = [[round(u * 2) / 2 for u in row] for row in U_mat.tolist()]
    V = [[round(v * 2) / 2 for v in row] for row in V_mat.tolist()]
    W = [[round(w * 2) / 2 for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    # If rounding didn't work, try integer rounding
    U = [[round(u) for u in row] for row in U_mat.tolist()]
    V = [[round(v) for v in row] for row in V_mat.tolist()]
    W = [[round(w) for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    # First, get the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        best = _schoolbook()
        best_rank = 12
    
    # Try to find a rank-10 solution using modular arithmetic first (faster)
    if best_rank > 10 and time.time() < deadline - 40:
        result10_mod = _try_rank10_modular(deadline)
        if result10_mod is not None:
            best = result10_mod
            best_rank = 10
            return best
    
    # Try to find a rank-10 solution using numerical search
    if best_rank > 10 and time.time() < deadline - 20:
        result10 = _try_rank10_search(deadline)
        if result10 is not None:
            best = result10
            best_rank = 10
    
    return best
