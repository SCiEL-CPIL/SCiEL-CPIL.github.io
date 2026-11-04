"""Find rank-minimal exact tensor decomposition for 3x4 x 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    def check_exact(U, V, W, tol=1e-10):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, rr] * V[j*p+k, rr] * W[i*n+k, rr] for rr in range(r))
                    if abs(s - 1.0) > tol:
                        return False
        for i in range(n):
            for j in range(m):
                for jp in range(m*p):
                    j_prime, k = divmod(jp, p)
                    if j_prime == j:
                        continue
                    for ip in range(n*p):
                        i_prime, k_prime = divmod(ip, p)
                        s = sum(U[i*m+j, rr] * V[jp, rr] * W[ip, rr] for rr in range(r))
                        if abs(s) > tol:
                            return False
        return True
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((n*p, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[i*n+k, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try various ranks with random init + ALS
    def als_round(U, V, W):
        r = U.shape[1]
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.array([V[jp,:] * W[ip,:] for jp in range(m*p) for ip in range(n*p)])
                b = np.array([1.0 if (ip//p==i and jp//p==j and ip%p==jp%p) else 0.0 
                             for jp in range(m*p) for ip in range(n*p)])
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i*m+j,:] = sol
        # Update V
        for jp in range(m*p):
            j_prime, k = divmod(jp, p)
            A = np.array([U[im,:] * W[ip,:] for im in range(n*m) for ip in range(n*p)])
            b = np.array([1.0 if (im//m==ip//p and im%m==j_prime and ip%p==k) else 0.0 
                         for im in range(n*m) for ip in range(n*p)])
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V[jp,:] = sol
        # Update W
        for ip in range(n*p):
            i_prime, k_prime = divmod(ip, p)
            A = np.array([U[im,:] * V[jp,:] for im in range(n*m) for jp in range(m*p)])
            b = np.array([1.0 if (im//m==i_prime and im%m==jp//p and jp%p==k_prime) else 0.0 
                         for im in range(n*m) for jp in range(m*p)])
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[ip,:] = sol
        return U, V, W
    
    for r in range(25, min(rank_sb, 85)):
        if time.time() - start_time > DEADLINE:
            break
        for trial in range(5):
            if time.time() - start_time > DEADLINE:
                break
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(n*p, r) * 0.1
            for _ in range(20):
                if time.time() - start_time > DEADLINE:
                    break
                U, V, W = als_round(U, V, W)
            if check_exact(U, V, W, tol=1e-8):
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
