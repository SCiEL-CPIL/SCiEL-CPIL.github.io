"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    nm, mp, pn = n*m, m*p, p*n
    target = np.ones((n, m, p))
    
    def compute_tensor(U, V, W):
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += uc[i, j] * vc[j, :] * wc[:, i]
        return T
    
    def verify(U, V, W, tol=1e-9):
        T = compute_tensor(U, V, W)
        return np.allclose(T, 1.0, atol=tol)
    
    def round_mat(M, denom=20):
        return [[float(Fraction(x).limit_denominator(denom)) for x in row] for row in M]
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def als_solve(r, seed, max_iters=200):
        """Alternating Least Squares for CP decomposition with better convergence."""
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        
        # Normalize initial factors
        norms_U = np.linalg.norm(U, axis=0, keepdims=True)
        norms_V = np.linalg.norm(V, axis=0, keepdims=True)
        norms_W = np.linalg.norm(W, axis=0, keepdims=True)
        U = U / (norms_U + 1e-10)
        V = V / (norms_V + 1e-10)
        W = W / (norms_W + 1e-10)
        
        prev_loss = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for U given V, W
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        x = np.zeros(r)
                    U[i*m+j, :] = x
            
            # Solve for V given U, W
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        x = np.zeros(r)
                    V[j*p+k, :] = x
            
            # Solve for W given U, V
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        x = np.zeros(r)
                    W[k*n+i, :] = x
            
            # Check convergence
            T = compute_tensor(U, V, W)
            loss = np.sum((T - target)**2)
            
            if abs(prev_loss - loss) < 1e-10 and it > 10:
                break
            if loss < 1e-12:
                break
            prev_loss = loss
        
        T = compute_tensor(U, V, W)
        loss = np.sum((T - target)**2)
        
        if loss < 1e-6:
            # Try to polish with more iterations
            for polish_iter in range(50):
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            x = np.zeros(r)
                        U[i*m+j, :] = x
                
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            x = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            x = np.zeros(r)
                        V[j*p+k, :] = x
                
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            x = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            x = np.zeros(r)
                        W[k*n+i, :] = x
                
                T = compute_tensor(U, V, W)
                loss = np.sum((T - target)**2)
                if loss < 1e-12:
                    break
            
            if loss < 1e-8:
                for denom in [10, 20, 50, 100, 200]:
                    Ur = np.array(round_mat(U, denom))
                    Vr = np.array(round_mat(V, denom))
                    Wr = np.array(round_mat(W, denom))
                    if verify(Ur, Vr, Wr, tol=1e-10):
                        return Ur.tolist(), Vr.tolist(), Wr.tolist()
        return None
    
    # Try ranks from high to low with multiple seeds
    for r in range(rank_sb - 1, max(rank_sb - 30, 1), -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Use multiple seeds for better coverage
        for seed in range(r, r + 8):
            if time.time() > deadline - 5:
                break
            result = als_solve(r, seed, max_iters=150)
            if result is not None:
                U_l, V_l, W_l = result
                best = {"U": U_l, "V": V_l, "W": W_l}
                best_rank = r
                found = True
                break
        if found:
            # Try one more rank lower if time permits
            if time.time() < deadline - 20 and r > 1:
                for seed in range(r-1, r-1+4):
                    if time.time() > deadline - 5:
                        break
                    result = als_solve(r-1, seed, max_iters=100)
                    if result is not None:
                        U_l, V_l, W_l = result
                        best = {"U": U_l, "V": V_l, "W": W_l}
                        best_rank = r-1
                        break
    
    return best
# EVOLVE-BLOCK-END
