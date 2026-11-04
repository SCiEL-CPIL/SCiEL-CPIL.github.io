"""Matrix multiplication 4,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Build target tensor entries
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries[(a, b, c)] = 1.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def verify(U, V, W, tol=1e-6):
        """Verify that the decomposition exactly reconstructs the target tensor."""
        rank = U.shape[1]
        # Reconstruct tensor using einsum
        R = np.einsum('ad,bd,cd->abc', U, V, W, optimize=True)
        # Create target tensor
        T = np.zeros((nm, mp, pn))
        for (a, b, c), val in target_entries.items():
            T[a, b, c] = val
        diff = R - T
        if np.max(np.abs(diff)) > tol:
            return False
        return True
    
    # Start with schoolbook
    U, V, W, rank = schoolbook()
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower-rank decompositions using structured approaches
    # Strategy: Try various random initializations and alternating minimization
    
    def try_find_decomposition(r, max_time=5.0, seed=42):
        """Try to find a rank-r decomposition using alternating minimization."""
        start = time.time()
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        for iteration in range(200):
            if time.time() - start > max_time or time.time() > deadline:
                break
            
            # Update W: fix U, V; for each c, solve linear system
            for c in range(pn):
                i = c % n
                k = c // n
                if k >= p:
                    continue
                
                M = np.zeros((m, r))
                t = np.ones(m)
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    for d in range(r):
                        M[j, d] = U[a, d] * V[b, d]
                
                try:
                    if r <= m:
                        W[c, :] = np.linalg.lstsq(M, t, rcond=None)[0]
                    else:
                        Mt = M.T
                        MMt = M @ Mt + 1e-10 * np.eye(r)
                        W[c, :] = Mt @ np.linalg.solve(MMt, t)
                except:
                    pass
            
            # Update U: fix V, W; for each a, solve linear system
            for a in range(nm):
                i = a // m
                j = a % m
                
                M = np.zeros((p, r))
                t = np.ones(p)
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    for d in range(r):
                        M[k, d] = V[b, d] * W[c, d]
                
                try:
                    if r <= p:
                        U[a, :] = np.linalg.lstsq(M, t, rcond=None)[0]
                    else:
                        Mt = M.T
                        MMt = M @ Mt + 1e-10 * np.eye(r)
                        U[a, :] = Mt @ np.linalg.solve(MMt, t)
                except:
                    pass
            
            # Update V: fix U, W; for each b, solve linear system
            for b in range(mp):
                j = b // p
                k = b % p
                
                M = np.zeros((n, r))
                t = np.ones(n)
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    for d in range(r):
                        M[i, d] = U[a, d] * W[c, d]
                
                try:
                    if r <= n:
                        V[b, :] = np.linalg.lstsq(M, t, rcond=None)[0]
                    else:
                        Mt = M.T
                        MMt = M @ Mt + 1e-10 * np.eye(r)
                        V[b, :] = Mt @ np.linalg.solve(MMt, t)
                except:
                    pass
        
        if verify(U, V, W, tol=1e-6):
            return U, V, W, r
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(nm, mp, pn)
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = try_find_decomposition(r, max_time=8.0, seed=seed + 100)
            if result is not None:
                U_sol, V_sol, W_sol, r_sol = result
                best_rank = r_sol
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
