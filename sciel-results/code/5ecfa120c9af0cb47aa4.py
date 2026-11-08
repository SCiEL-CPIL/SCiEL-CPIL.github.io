import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    eqs_by_a = {}
    eqs_by_b = {}
    eqs_by_c = {}
    for idx, (a, b, c) in enumerate(triples):
        eqs_by_a.setdefault(a, []).append((idx, b, c))
        eqs_by_b.setdefault(b, []).append((idx, a, c))
        eqs_by_c.setdefault(c, []).append((idx, a, b))
    
    def check_solution(U, V, W, tol=1e-6):
        max_err = 0.0
        for (a, b, c) in triples:
            val = np.dot(U[a], V[b] * W[c])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        return max_err < tol
    
    def als_step(U, V, W, rank):
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        for a in range(nm):
            if a not in eqs_by_a:
                continue
            H = np.zeros((rank, rank))
            g = np.zeros(rank)
            for (idx, b, c) in eqs_by_a[a]:
                vw = V[b] * W[c]
                H += np.outer(vw, vw)
                g += vw
            H += 1e-10 * np.eye(rank)
            try:
                U_new[a] = np.linalg.solve(H, g)
            except:
                U_new[a] = np.linalg.lstsq(H, g, rcond=None)[0]
        
        for b in range(mp):
            if b not in eqs_by_b:
                continue
            H = np.zeros((rank, rank))
            g = np.zeros(rank)
            for (idx, a, c) in eqs_by_b[b]:
                uw = U[a] * W[c]
                H += np.outer(uw, uw)
                g += uw
            H += 1e-10 * np.eye(rank)
            try:
                V_new[b] = np.linalg.solve(H, g)
            except:
                V_new[b] = np.linalg.lstsq(H, g, rcond=None)[0]
        
        for c in range(pn):
            if c not in eqs_by_c:
                continue
            H = np.zeros((rank, rank))
            g = np.zeros(rank)
            for (idx, a, b) in eqs_by_c[c]:
                uv = U[a] * V[b]
                H += np.outer(uv, uv)
                g += uv
            H += 1e-10 * np.eye(rank)
            try:
                W_new[c] = np.linalg.solve(H, g)
            except:
                W_new[c] = np.linalg.lstsq(H, g, rcond=None)[0]
        
        return U_new, V_new, W_new
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    def try_round(U, V, W):
        """Try rounding to half-integers."""
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if check_solution(U_r, V_r, W_r, tol=1e-4):
                return U_r, V_r, W_r
        return None
    
    start_time = time.time()
    deadline = 135.0
    
    U_sb, V_sb, W_sb = schoolbook()
    best_solution = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try ranks from 30 to best_rank-1
    for rank in range(30, best_rank):
        if time.time() - start_time > deadline:
            break
        
        found = False
        for trial in range(5):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            for it in range(500):
                if time.time() - start_time > deadline:
                    break
                U, V, W = als_step(U, V, W, rank)
                
                if it % 25 == 0:
                    err = 0.0
                    for (a, b, c) in triples:
                        val = np.dot(U[a], V[b] * W[c])
                        err = max(err, abs(val - 1.0))
                    
                    if err < 1e-5:
                        for _ in range(30):
                            U, V, W = als_step(U, V, W, rank)
                        if check_solution(U, V, W, tol=1e-6):
                            rounded = try_round(U, V, W)
                            if rounded:
                                U, V, W = rounded
                            found = True
                            break
            
            if found:
                break
        
        if found:
            best_rank = rank
            best_solution = (U, V, W)
    
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
