import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 115.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    R = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if s != 1:
                    return False
    return True


def _try_rank17_search(n, m, p, seed, max_iters=100):
    """
    Robust search for rank-17 decomposition using random initialization 
    and alternating minimization.
    """
    rng = random.Random(seed)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    R = 17
    
    # Random initialization
    u = [[rng.uniform(-1, 1) for _ in range(n * m)] for _ in range(R)]
    v = [[rng.uniform(-1, 1) for _ in range(m * p)] for _ in range(R)]
    w = [[rng.uniform(-1, 1) for _ in range(p * n)] for _ in range(R)]
    
    def compute_residual(u, v, w):
        res = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += u[r][u_idx(i, j)] * v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    res[(i, j, k)] = s - 1.0
        return res
    
    def solve_u_given_vw(v, w):
        A = np.zeros((n * m * p, R * n * m))
        rhs = np.ones(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        A[eq, r * n * m + i * m + j] = v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    eq += 1
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        u_new = [row[:] for row in u]
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    u_new[r][i * m + j] = sol[r * n * m + i * m + j]
        return u_new
    
    def solve_v_given_uw(u, w):
        A = np.zeros((n * m * p, R * m * p))
        rhs = np.ones(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        A[eq, r * m * p + j * p + k] = u[r][u_idx(i, j)] * w[r][w_idx(k, i)]
                    eq += 1
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        v_new = [row[:] for row in v]
        for r in range(R):
            for j in range(m):
                for k in range(p):
                    v_new[r][j * p + k] = sol[r * m * p + j * p + k]
        return v_new
    
    def solve_w_given_uv(u, v):
        A = np.zeros((n * m * p, R * p * n))
        rhs = np.ones(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        A[eq, r * p * n + k * n + i] = u[r][u_idx(i, j)] * v[r][v_idx(j, k)]
                    eq += 1
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        w_new = [row[:] for row in w]
        for r in range(R):
            for k in range(p):
                for i in range(n):
                    w_new[r][k * n + i] = sol[r * p * n + k * n + i]
        return w_new
    
    best_res = float('inf')
    best_state = None
    
    for it in range(max_iters):
        if deadline_exceeded():
            break
        
        u_new = solve_u_given_vw(v, w)
        if u_new is None: break
        u = u_new
        
        v_new = solve_v_given_uw(u, w)
        if v_new is None: break
        v = v_new
        
        w_new = solve_w_given_uv(u, v)
        if w_new is None: break
        w = w_new
        
        res = compute_residual(u, v, w)
        norm = sum(x * x for x in res.values()) ** 0.5
        
        if norm < best_res:
            best_res = norm
            best_state = (u[:], v[:], [row[:] for row in w])
            if norm < 1e-12:
                break
    
    if best_state is None:
        return None
    
    u, v, w = best_state
    
    # Snap to half-integers
    def round_half(x):
        h = x * 2
        r = round(h)
        if abs(h - r) < 1e-5:
            return Fraction(r, 2)
        return Fraction(x)
    
    U_frac = [[round_half(x) for x in row] for row in u]
    V_frac = [[round_half(x) for x in row] for row in v]
    W_frac = [[round_half(x) for x in row] for row in w]
    
    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
        return {"U": [[float(x) for x in row] for row in U_frac],
                "V": [[float(x) for x in row] for row in V_frac],
                "W": [[float(x) for x in row] for row in W_frac]}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    fallback = _schoolbook(n, m, p)
    
    # Try robust rank-17 search with multiple seeds
    # We increase the number of seeds and reduce iterations per seed to allow more restarts
    for seed in range(100):
        if deadline_exceeded():
            break
        result = _try_rank17_search(n, m, p, seed, max_iters=50)
        if result is not None:
            return result
    
    return fallback


def main():
    return construct()


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
