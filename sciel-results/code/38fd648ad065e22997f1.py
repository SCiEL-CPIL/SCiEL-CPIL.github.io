"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i*m+j, j*p+k, k*n+i)] = 1
    
    start = time.time()
    deadline = start + 100
    
    # Try ranks from low to high
    for rank in range(13, 22):
        if time.time() > deadline:
            break
        for attempt in range(50):
            if time.time() > deadline:
                break
            res = _solve(n, m, p, rank, T, attempt)
            if res is not None:
                return res
    
    # Schoolbook fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _solve(n, m, p, rank, T, seed):
    nm, mp, pn = n*m, m*p, p*n
    rng = random.Random(42 + seed)
    
    U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(nm)]
    V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(mp)]
    W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(pn)]
    
    for it in range(150):
        # Update W
        for c in range(pn):
            for r in range(rank):
                num = 0.0
                den = 0.0
                for a in range(nm):
                    for b in range(mp):
                        t = T.get((a, b, c), 0)
                        num += U[a][r] * V[b][r] * t
                        den += (U[a][r] * V[b][r]) ** 2
                W[c][r] = num / den if den > 1e-14 else 0.0
        
        # Update U
        for a in range(nm):
            for r in range(rank):
                num = 0.0
                den = 0.0
                for b in range(mp):
                    for c in range(pn):
                        t = T.get((a, b, c), 0)
                        num += V[b][r] * W[c][r] * t
                        den += (V[b][r] * W[c][r]) ** 2
                U[a][r] = num / den if den > 1e-14 else 0.0
        
        # Update V
        for b in range(mp):
            for r in range(rank):
                num = 0.0
                den = 0.0
                for a in range(nm):
                    for c in range(pn):
                        t = T.get((a, b, c), 0)
                        num += U[a][r] * W[c][r] * t
                        den += (U[a][r] * W[c][r]) ** 2
                V[b][r] = num / den if den > 1e-14 else 0.0
        
        # Check error
        err = 0.0
        for (a, b, c), v in T.items():
            s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
            err += (s - v) ** 2
        if err < 1e-8:
            Ur = [[round(x * 2) / 2 for x in row] for row in U]
            Vr = [[round(x * 2) / 2 for x in row] for row in V]
            Wr = [[round(x * 2) / 2 for x in row] for row in W]
            ok = True
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        s = sum(Ur[a][r] * Vr[b][r] * Wr[c][r] for r in range(rank))
                        if abs(s - T.get((a, b, c), 0)) > 1e-9:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                return {"U": Ur, "V": Vr, "W": Wr}
    
    return None

# EVOLVE-BLOCK-END

