import time
import random
import math
import numpy as np

DIMENSIONS = [2, 3, 6]
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _residual(U, V, W, n, m, p):
    """Return max absolute tensor residual over all (i,j,k)."""
    worst = 0.0
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(r):
                    s += uc[c] * vc[c] * wc[c]
                diff = abs(s - 1.0)
                if diff > worst:
                    worst = diff
    return worst

def _try_round_verify(U, V, W, tol=1e-9):
    """Round to half-integers and check exactness."""
    Ur = [[round(v * 2) / 2 for v in row] for row in U]
    Vr = [[round(v * 2) / 2 for v in row] for row in V]
    Wr = [[round(v * 2) / 2 for v in row] for row in W]
    n, m, p = DIMENSIONS
    r = len(Ur[0]) if Ur else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ur[i * m + j][c] * Vr[j * p + k][c] * Wr[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return None
    return {"U": Ur, "V": Vr, "W": Wr}

def _refine(U, V, W, n, m, p, iters=100):
    """Gauss-Seidel style refinement to drive residual toward zero."""
    r = len(U[0]) if U else 0
    for _ in range(iters):
        # Update W: for each (i,k), solve m linear equations
        for i in range(n):
            for k in range(p):
                A = []
                for j in range(m):
                    row = [U[i * m + j][c] * V[j * p + k][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        W[k * n + i][c] = float(sol[c])
                except Exception:
                    pass
        # Update U: for each (i,j), solve p linear equations
        for i in range(n):
            for j in range(m):
                A = []
                for k in range(p):
                    row = [V[j * p + k][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        U[i * m + j][c] = float(sol[c])
                except Exception:
                    pass
        # Update V: for each (j,k), solve n linear equations
        for j in range(m):
            for k in range(p):
                A = []
                for i in range(n):
                    row = [U[i * m + j][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        V[j * p + k][c] = float(sol[c])
                except Exception:
                    pass
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _local_search_halfint(U, V, W, n, m, p, max_steps=500):
    """Local search on the half-integer grid to improve the decomposition."""
    r = len(U[0]) if U else 0
    # Start from rounded values
    Ur = [[round(v * 2) / 2 for v in row] for row in U]
    Vr = [[round(v * 2) / 2 for v in row] for row in V]
    Wr = [[round(v * 2) / 2 for v in row] for row in W]
    
    best_res = _residual(Ur, Vr, Wr, n, m, p)
    best_U, best_V, best_W = [row[:] for row in Ur], [row[:] for row in Vr], [row[:] for row in Wr]
    
    half_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for step in range(max_steps):
        if _remaining() < 5.0:
            break
        if best_res < 1e-9:
            break
        
        # Try flipping one random entry
        which = random.randint(0, 2)
        if which == 0:
            row_idx = random.randint(0, n * m - 1)
            col_idx = random.randint(0, r - 1)
            old = Ur[row_idx][col_idx]
            new = random.choice([v for v in half_vals if v != old])
            Ur[row_idx][col_idx] = new
            res = _residual(Ur, Vr, Wr, n, m, p)
            if res < best_res - 1e-12:
                best_res = res
                best_U = [row[:] for row in Ur]
                best_V = [row[:] for row in Vr]
                best_W = [row[:] for row in Wr]
            else:
                Ur[row_idx][col_idx] = old
        elif which == 1:
            row_idx = random.randint(0, m * p - 1)
            col_idx = random.randint(0, r - 1)
            old = Vr[row_idx][col_idx]
            new = random.choice([v for v in half_vals if v != old])
            Vr[row_idx][col_idx] = new
            res = _residual(Ur, Vr, Wr, n, m, p)
            if res < best_res - 1e-12:
                best_res = res
                best_U = [row[:] for row in Ur]
                best_V = [row[:] for row in Vr]
                best_W = [row[:] for row in Wr]
            else:
                Vr[row_idx][col_idx] = old
        else:
            row_idx = random.randint(0, p * n - 1)
            col_idx = random.randint(0, r - 1)
            old = Wr[row_idx][col_idx]
            new = random.choice([v for v in half_vals if v != old])
            Wr[row_idx][col_idx] = new
            res = _residual(Ur, Vr, Wr, n, m, p)
            if res < best_res - 1e-12:
                best_res = res
                best_U = [row[:] for row in Ur]
                best_V = [row[:] for row in Vr]
                best_W = [row[:] for row in Wr]
            else:
                Wr[row_idx][col_idx] = old
    
    if best_res < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None

def _try_rank(rank, n, m, p, rng, max_attempts=10):
    """Try to find a valid 2x3x6 decomposition of given rank using random init + refinement."""
    for attempt in range(max_attempts):
        if _remaining() < 5.0:
            break
        
        # Different initialization strategies
        if attempt % 3 == 0:
            # Sparse half-integer init
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            for row in U:
                for c in range(rank):
                    if rng.random() < 0.6:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
            for row in V:
                for c in range(rank):
                    if rng.random() < 0.6:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
            for row in W:
                for c in range(rank):
                    if rng.random() < 0.6:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
        elif attempt % 3 == 1:
            # Dense half-integer init
            U = [[rng.choice([-1, -0.5, 0.5, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, -0.5, 0.5, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, -0.5, 0.5, 1]) for _ in range(rank)] for _ in range(p * n)]
        else:
            # Float init
            U = [[rng.uniform(-2, 2) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.uniform(-2, 2) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.uniform(-2, 2) for _ in range(rank)] for _ in range(p * n)]
        
        U, V, W = _refine(U, V, W, n, m, p, iters=50)
        res = _residual(U, V, W, n, m, p)
        
        if res < 1e-5:
            rounded = _try_round_verify(U, V, W)
            if rounded is not None:
                return rounded
            
            # Try local search on half-integers
            ls_result = _local_search_halfint(U, V, W, n, m, p, max_steps=300)
            if ls_result is not None:
                return ls_result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    rng = random.Random(42)
    
    # Try ranks from just below schoolbook down to a reasonable minimum
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    for rank in range(best_rank - 1, 17, -1):
        if _remaining() < 10.0:
            break
        
        result = _try_rank(rank, n, m, p, rng, max_attempts=10)
        if result is not None:
            best = result
            best_rank = rank
            continue
    
    return best

if __name__ == "__main__":
    print(construct())
