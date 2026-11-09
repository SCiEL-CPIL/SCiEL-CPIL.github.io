import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized exact verification. U:(n*m,r) V:(m*p,r) W:(p*n,r)."""
    rank = U.shape[1]
    T = np.einsum('uc,vc,wc->uvw', U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > tol:
                    return False
    return True

def _local_search_pm1(rank, n, m, p, n_restarts, deadline):
    """Local search for +/-1 solution."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Precompute index mappings
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # Residual index r_idx = i * (m*p) + j * p + k
    
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                r_idx = i * (m * p) + j * p + k
                triples.append((u, v, w, r_idx))
    
    num_triples = len(triples)
    
    # For each variable, we need to know which residuals it affects and how.
    # Var U[u, c] affects residuals where u is fixed.
    # Var V[v, c] affects residuals where v is fixed.
    # Var W[w, c] affects residuals where w is fixed.
    
    # To speed up delta calculation, we can maintain the current residuals R.
    # Delta for flipping U[u, c] from val to -val:
    # R_new = R - 2 * val * (V[v, c] * W[w, c]) for all (u,v,w) with this u.
    # Sum of squares change:
    # sum (R_i - 2*val*P_i)^2 - sum R_i^2
    # = sum (R_i^2 - 4*val*R_i*P_i + 4*val^2*P_i^2) - sum R_i^2
    # = -4*val * sum(R_i*P_i) + 4*val^2 * sum(P_i^2)
    
    # We will precompute for each variable the list of (r_idx, P) pairs.
    # But P depends on the current values of V and W.
    # So we cannot fully precompute P.
    # However, we can compute P on the fly for the affected residuals.
    
    best_rank_found = None
    best_U, best_V, best_W = None, None, None
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        U = np.random.choice([-1, 1], size=(size_u, rank))
        V = np.random.choice([-1, 1], size=(size_v, rank))
        W = np.random.choice([-1, 1], size=(size_w, rank))
        
        # Compute initial residuals
        R = np.zeros(num_triples)
        for idx, (u, v, w, r_idx) in enumerate(triples):
            # R[r_idx] = sum_c U[u,c] V[v,c] W[w,c] - 1
            # We can compute this efficiently
            pass
        
        # Vectorized residual calculation
        def calc_res(U, V, W):
            T = np.einsum('uc,vc,wc->uvw', U, V, W)
            res = np.zeros(num_triples)
            for idx, (u, v, w, r_idx) in enumerate(triples):
                res[r_idx] = T[u, v, w] - 1.0
            return res
        
        R = calc_res(U, V, W)
        cost = np.sum(R**2)
        
        if cost < 1e-9:
            return U, V, W, rank
            
        for step in range(1000):
            if time.time() > deadline:
                break
                
            best_delta = 0
            best_move = None
            
            # Check U flips
            for u in range(size_u):
                for c in range(rank):
                    val = U[u, c]
                    # Find affected residuals
                    # u = i*m + j => i = u//m, j = u%m
                    i = u // m
                    j = u % m
                    delta_sum_r_p = 0
                    delta_sum_p2 = 0
                    for k in range(p):
                        v = j * p + k
                        w = k * n + i
                        r_idx = i * (m * p) + j * p + k
                        P = V[v, c] * W[w, c]
                        delta_sum_r_p += R[r_idx] * P
                        delta_sum_p2 += P * P
                    # Delta cost = -4 * val * sum(R*P) + 4 * val^2 * sum(P^2)
                    # val is +/-1, so val^2 = 1
                    delta_cost = -4 * val * delta_sum_r_p + 4 * delta_sum_p2
                    if delta_cost < best_delta:
                        best_delta = delta_cost
                        best_move = ('U', u, c)
                        
            # Check V flips
            for v in range(size_v):
                for c in range(rank):
                    val = V[v, c]
                    j = v // p
                    k = v % p
                    delta_sum_r_p = 0
                    delta_sum_p2 = 0
                    for i in range(n):
                        u = i * m + j
                        w = k * n + i
                        r_idx = i * (m * p) + j * p + k
                        P = U[u, c] * W[w, c]
                        delta_sum_r_p += R[r_idx] * P
                        delta_sum_p2 += P * P
                    delta_cost = -4 * val * delta_sum_r_p + 4 * delta_sum_p2
                    if delta_cost < best_delta:
                        best_delta = delta_cost
                        best_move = ('V', v, c)
                        
            # Check W flips
            for w in range(size_w):
                for c in range(rank):
                    val = W[w, c]
                    k = w // n
                    i = w % n
                    delta_sum_r_p = 0
                    delta_sum_p2 = 0
                    for j in range(m):
                        u = i * m + j
                        v = j * p + k
                        r_idx = i * (m * p) + j * p + k
                        P = U[u, c] * V[v, c]
                        delta_sum_r_p += R[r_idx] * P
                        delta_sum_p2 += P * P
                    delta_cost = -4 * val * delta_sum_r_p + 4 * delta_sum_p2
                    if delta_cost < best_delta:
                        best_delta = delta_cost
                        best_move = ('W', w, c)
                        
            if best_move is None or best_delta >= 0:
                break
                
            # Apply move
            typ, idx1, idx2 = best_move
            if typ == 'U':
                U[idx1, idx2] *= -1
            elif typ == 'V':
                V[idx1, idx2] *= -1
            else:
                W[idx1, idx2] *= -1
                
            # Update residuals
            R = calc_res(U, V, W)
            cost = np.sum(R**2)
            
            if cost < 1e-9:
                return U, V, W, rank
                
    return None

def _try_least_squares(rank, n, m, p, n_attempts, deadline):
    """Structured optimization search for half-integer solutions."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_size = (size_u + size_v + size_w) * rank

    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    num_triples = len(triples)

    def unpack(x):
        U = x[:size_u * rank].reshape(size_u, rank)
        V = x[size_u * rank:(size_u + size_v) * rank].reshape(size_v, rank)
        W = x[(size_u + size_v) * rank:].reshape(size_w, rank)
        return U, V, W

    def residuals(x):
        U, V, W = unpack(x)
        res = np.empty(num_triples)
        for idx, (u, v, w) in enumerate(triples):
            res[idx] = np.dot(U[u, :], V[v, :] * W[w, :]) - 1.0
        return res

    def jacobian(x):
        U, V, W = unpack(x)
        J = np.zeros((num_triples, total_size))
        for idx, (u, v, w) in enumerate(triples):
            VW = V[v, :] * W[w, :]
            UW = U[u, :] * W[w, :]
            UV = U[u, :] * V[v, :]
            J[idx, u * rank:(u + 1) * rank] = VW
            J[idx, size_u * rank + v * rank:size_u * rank + (v + 1) * rank] = UW
            J[idx, (size_u + size_v) * rank + w * rank:(size_u + size_v) * rank + (w + 1) * rank] = UV
        return J

    best = None
    best_norm = np.inf
    for _ in range(n_attempts):
        if time.time() > deadline:
            break
        x0 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=total_size)
        try:
            from scipy.optimize import least_squares
            sol = least_squares(residuals, x0, jac=jacobian, method='lm',
                                max_nfev=2000)
            norm = np.linalg.norm(sol.fun)
            if norm < best_norm:
                best_norm = norm
                best = sol.x
            if norm < 1e-7:
                U, V, W = unpack(sol.x)
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
        except Exception:
            continue
    if best is not None and best_norm < 1e-4:
        try:
            from scipy.optimize import least_squares
            sol = least_squares(residuals, best, jac=jacobian, method='lm',
                                max_nfev=10000)
            U, V, W = unpack(sol.x)
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
        except Exception:
            pass
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)

    # Search ranks from low to high
    for rank in range(17, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Phase 1: Local Search on +/-1
        result = _local_search_pm1(rank, n, m, p, n_restarts=100, deadline=deadline - 20)
        if result is not None:
            U, V, W, r = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
            
        # Phase 2: Least Squares
        if time.time() > deadline - 20:
            break
        result = _try_least_squares(rank, n, m, p, n_attempts=10, deadline=deadline - 20)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break

    U_out = [[(int(x * 2) / 2) for x in row] for row in best_U]
    V_out = [[(int(x * 2) / 2) for x in row] for row in best_V]
    W_out = [[(int(x * 2) / 2) for x in row] for row in best_W]

    return {"U": U_out, "V": V_out, "W": W_out}
