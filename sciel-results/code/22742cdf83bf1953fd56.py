import numpy as np
import time
from itertools import product
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    def verify(U, V, W, R):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for k in range(p):
                wv = W[k * n + i]
                M = np.zeros((m, m))
                for j in range(m):
                    uv = U[i * m + j]
                    for jp in range(m):
                        vv = V[jp * p + k]
                        M[j, jp] = np.dot(wv, uv * vv)
                if not np.allclose(M, np.eye(m), atol=1e-7):
                    return False
        return True

    def to_half_int(M):
        return np.round(np.array(M, dtype=float) * 2) / 2

    # New strategy: exploit the structural symmetry of matrix multiplication.
    # For C = A*B where A is n x m and B is m x p, the tensor has a special form.
    # We use a parameterized ansatz with shared structure across the three factors.
    # 
    # Key insight: For 2x3x3, we can use a "block" structure where the rank-r
    # decomposition exploits the fact that n=2 is small. We parameterize:
    #   U[i,j,r] = alpha[i,r] * beta[j,r]
    #   V[j,k,r] = gamma[j,r] * delta[k,r]  
    #   W[k,i,r] = epsilon[k,r] * zeta[i,r]
    # 
    # This reduces the search space significantly while maintaining expressiveness.
    # The constraints become:
    #   sum_r zeta[i,r]*epsilon[k,r]*alpha[i,r]*beta[j,r]*gamma[jp,r]*delta[k,r] = delta_{j,jp}
    #   for all i,k,j,jp
    #
    # This is 2*3*3*3 = 54 scalar equations in 6*R unknowns.
    
    def build_from_structural(alpha, beta, gamma, delta, epsilon, zeta, R):
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = alpha[i, r] * beta[j, r]
            for j in range(m):
                for k in range(p):
                    V[j * p + k, r] = gamma[j, r] * delta[k, r]
            for k in range(p):
                for i in range(n):
                    W[k * n + i, r] = epsilon[k, r] * zeta[i, r]
        return U, V, W

    def structural_residual(x, R):
        # x layout: alpha(n,R), beta(m,R), gamma(m,R), delta(p,R), epsilon(p,R), zeta(n,R)
        idx = 0
        alpha = x[idx:idx+n*R].reshape(n, R); idx += n*R
        beta = x[idx:idx+m*R].reshape(m, R); idx += m*R
        gamma = x[idx:idx+m*R].reshape(m, R); idx += m*R
        delta = x[idx:idx+p*R].reshape(p, R); idx += p*R
        epsilon = x[idx:idx+p*R].reshape(p, R); idx += p*R
        zeta = x[idx:idx+n*R].reshape(n, R); idx += n*R
        
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = 0.0
                        for r in range(R):
                            val += zeta[i, r] * epsilon[k, r] * alpha[i, r] * beta[j, r] * gamma[jp, r] * delta[k, r]
                        res.append(val - (1.0 if j == jp else 0.0))
        return np.array(res)

    def try_rank_structural(R, max_restarts=20):
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            rng = np.random.RandomState(restart * 31 + R * 100)
            
            # Initialize with structured random values
            x0 = rng.randn(6 * R) * 0.5
            # Add some structure: make first few entries larger
            x0[:R] += 0.5  # alpha
            x0[R:2*R] += 0.3  # beta
            
            try:
                sol = least_squares(lambda x: structural_residual(x, R), x0,
                                    method='lm', max_nfev=800, xtol=1e-14, ftol=1e-14)
            except Exception:
                continue
            
            if sol.cost > 1e-12:
                # Try a few more restarts with different initializations
                continue
            
            idx = 0
            alpha = sol.x[idx:idx+n*R].reshape(n, R); idx += n*R
            beta = sol.x[idx:idx+m*R].reshape(m, R); idx += m*R
            gamma = sol.x[idx:idx+m*R].reshape(m, R); idx += m*R
            delta = sol.x[idx:idx+p*R].reshape(p, R); idx += p*R
            epsilon = sol.x[idx:idx+p*R].reshape(p, R); idx += p*R
            zeta = sol.x[idx:idx+n*R].reshape(n, R); idx += n*R
            
            U, V, W = build_from_structural(alpha, beta, gamma, delta, epsilon, zeta, R)
            
            # Round to half-integers
            U_r, V_r, W_r = to_half_int(U), to_half_int(V), to_half_int(W)
            if verify(U_r, V_r, W_r, R):
                return (U_r, V_r, W_r, R)
            
            # Try scaling
            for _ in range(8):
                sc = rng.uniform(0.5, 2.0)
                Us, Vs = U * sc, V / sc
                Ur, Vr, Wr = to_half_int(Us), to_half_int(Vs), to_half_int(W)
                if verify(Ur, Vr, Wr, R):
                    return (Ur, Vr, Wr, R)
        
        return None

    # Try a general (non-structured) approach with more restarts
    def general_residual(x, R):
        U = x[:nm*R].reshape(nm, R)
        V = x[nm*R: nm*R + mp*R].reshape(mp, R)
        W = x[nm*R + mp*R: nm*R + mp*R + pn*R].reshape(pn, R)
        res = []
        for i in range(n):
            for k in range(p):
                wv = W[k*n+i]
                for j in range(m):
                    uv = U[i*m+j]
                    for jp in range(m):
                        vv = V[jp*p+k]
                        res.append(np.dot(wv, uv*vv) - (1.0 if j == jp else 0.0))
        return np.array(res)

    def try_rank_general(R, max_restarts=10):
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            rng = np.random.RandomState(restart * 97 + R * 200)
            x0 = rng.randn(R * (nm + mp + pn)) * 0.3
            try:
                sol = least_squares(lambda x: general_residual(x, R), x0,
                                    method='lm', max_nfev=600, xtol=1e-14, ftol=1e-14)
            except Exception:
                continue
            if sol.cost > 1e-12:
                continue
            U = sol.x[:nm*R].reshape(nm, R)
            V = sol.x[nm*R: nm*R + mp*R].reshape(mp, R)
            W = sol.x[nm*R + mp*R: nm*R + mp*R + pn*R].reshape(pn, R)
            U_r, V_r, W_r = to_half_int(U), to_half_int(V), to_half_int(W)
            if verify(U_r, V_r, W_r, R):
                return (U_r, V_r, W_r, R)
            for _ in range(6):
                sc = rng.uniform(0.5, 2.0)
                Us, Vs = U * sc, V / sc
                Ur, Vr, Wr = to_half_int(Us), to_half_int(Vs), to_half_int(W)
                if verify(Ur, Vr, Wr, R):
                    return (Ur, Vr, Wr, R)
        return None

    # Search from low rank upward
    for R in range(13, 18):
        if time.time() > deadline - 10:
            break
        
        # Try structured approach first (fewer variables)
        result = try_rank_structural(R, max_restarts=25)
        if result is not None:
            U, V, W, r = result
            if r < len(best['U'][0]):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
        
        # Then try general approach
        result = try_rank_general(R, max_restarts=15)
        if result is not None:
            U, V, W, r = result
            if r < len(best['U'][0]):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break

    return best
