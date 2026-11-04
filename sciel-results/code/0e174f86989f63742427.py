import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # Check all valid (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        # Check some invalid entries (sample)
        np.random.seed(42)
        for _ in range(1000):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            # Check if this is a valid triple
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k2 = b % p
            k1 = c // n
            i2 = c % n
            if i1 == i2 and j1 == j2 and k1 == k2:
                continue  # valid, skip
            s = 0.0
            for r in range(rank):
                s += U[a, r] * V[b, r] * W[c, r]
            if abs(s) > 1e-8:
                return False
        return True
    
    def try_rank(r, max_attempts=5):
        """Try to find a decomposition of rank r."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Use structured initialization based on Strassen-like patterns
            if attempt == 0:
                # Random but with some structure
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.random.randn(p*n, r)
            elif attempt == 1:
                # Block diagonal structure
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                block_size = max(1, r // 8)
                for b in range(min(8, r // block_size)):
                    start = b * block_size
                    end = min(start + block_size, r)
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, start:end] = np.random.randn(end - start) * 0.5
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k, start:end] = np.random.randn(end - start) * 0.5
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, start:end] = np.random.randn(end - start) * 0.5
            elif attempt == 2:
                # Sparse random
                U = np.random.randn(n*m, r) * (np.random.rand(n*m, r) < 0.3)
                V = np.random.randn(m*p, r) * (np.random.rand(m*p, r) < 0.3)
                W = np.random.randn(p*n, r) * (np.random.rand(p*n, r) < 0.3)
            elif attempt == 3:
                # Binary-like
                U = np.random.randint(0, 2, (n*m, r)).astype(float) - 0.5
                V = np.random.randint(0, 2, (m*p, r)).astype(float) - 0.5
                W = np.random.randint(0, 2, (p*n, r)).astype(float) - 0.5
            else:
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.random.randn(p*n, r)
            
            # Quick check on valid entries
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r_idx in range(r):
                            s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        error += (s - 1.0) ** 2
                        if error > 1.0:
                            break
                    if error > 1.0:
                        break
                if error > 1.0:
                    break
            
            if error < 1e-6:
                # Full verification
                if verify(U, V, W):
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r, V_r, W_r):
                        return U_r, V_r, W_r
        return None
    
    # Try ranks from schoolbook down
    # Known lower bounds suggest we should try reasonable ranks
    for r in range(rank_sb - 1, max(60, rank_sb - 30), -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_attempts=3)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            break
    
    # Convert to lists
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
