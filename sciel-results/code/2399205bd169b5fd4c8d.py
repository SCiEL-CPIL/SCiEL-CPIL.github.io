"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook fallback
    schoolbook_U = np.zeros((nm, n * m * p))
    schoolbook_V = np.zeros((mp, n * m * p))
    schoolbook_W = np.zeros((pn, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U[i * m + j, col] = 1
                schoolbook_V[j * p + k, col] = 1
                schoolbook_W[k * n + i, col] = 1
                col += 1
    best_result = {
        "U": schoolbook_U.tolist(),
        "V": schoolbook_V.tolist(),
        "W": schoolbook_W.tolist()
    }
    best_rank = n * m * p
    
    np.random.seed(42)
    
    def check_error(U, V, W, tol=1e-9):
        """Check max error of tensor decomposition using vectorized einsum."""
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        max_err = np.max(np.abs(T - 1.0))
        return float(max_err)
    
    def update_U(U, V, W, rank):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        A = np.einsum('jkr,kir->ijkr', V_3d, W_3d)
        for i in range(n):
            for j in range(m):
                try:
                    x = np.linalg.lstsq(A[i, j], np.ones(p), rcond=None)[0]
                    U_3d[i, j, :] = x
                except:
                    pass
        return U_3d.reshape(nm, rank)

    def update_V(U, V, W, rank):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        A = np.einsum('ijr,kir->jkir', U_3d, W_3d)
        for j in range(m):
            for k in range(p):
                try:
                    x = np.linalg.lstsq(A[j, k], np.ones(n), rcond=None)[0]
                    V_3d[j, k, :] = x
                except:
                    pass
        return V_3d.reshape(mp, rank)

    def update_W(U, V, W, rank):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        A = np.einsum('ijr,jkr->kijr', U_3d, V_3d)
        for k in range(p):
            for i in range(n):
                try:
                    x = np.linalg.lstsq(A[k, i], np.ones(m), rcond=None)[0]
                    W_3d[k, i, :] = x
                except:
                    pass
        return W_3d.reshape(pn, rank)
    
    def polish_solution(U, V, W, rank, max_iters=100):
        """High-precision polish pass with tighter tolerance."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            U = update_U(U, V, W, rank)
            V = update_V(U, V, W, rank)
            W = update_W(U, V, W, rank)
            max_err = check_error(U, V, W)
            if max_err < 1e-12:
                break
        return U, V, W, max_err
    
    def try_exact_rational(U, V, W, rank, max_trials=100):
        """Try to find exact rational/half-integer solution near given solution."""
        # First try simple rounding to half-integers
        for scale in [2, 4, 8]:
            U_round = np.round(U * scale) / scale
            V_round = np.round(V * scale) / scale
            W_round = np.round(W * scale) / scale
            err = check_error(U_round, V_round, W_round)
            if err < 1e-9:
                return U_round, V_round, W_round
        
        # Try with small perturbations
        for _ in range(max_trials):
            noise_u = np.random.randn(nm, rank) * 0.02
            noise_v = np.random.randn(mp, rank) * 0.02
            noise_w = np.random.randn(pn, rank) * 0.02
            U_t = np.round((U + noise_u) * 2) / 2
            V_t = np.round((V + noise_v) * 2) / 2
            W_t = np.round((W + noise_w) * 2) / 2
            err = check_error(U_t, V_t, W_t)
            if err < 1e-9:
                return U_t, V_t, W_t
        return None, None, None
    
    # Try ranks from 108 to 119
    for rank in range(108, 120):
        if time.time() > deadline - 25:
            break
        
        restarts = 30 if rank < 115 else 15
        
        for restart in range(restarts):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                U = np.random.randn(nm, rank) * 0.3
                V = np.random.randn(mp, rank) * 0.3
                W = np.random.randn(pn, rank) * 0.3
            elif restart % 3 == 1:
                U = np.random.randn(nm, rank) * 0.5
                V = np.random.randn(mp, rank) * 0.5
                W = np.random.randn(pn, rank) * 0.5
            else:
                # Structured initialization based on schoolbook pattern
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                # Initialize with some schoolbook-like structure
                for c in range(rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                # Add noise to break symmetry
                U += np.random.randn(nm, rank) * 0.1
                V += np.random.randn(mp, rank) * 0.1
                W += np.random.randn(pn, rank) * 0.1
            
            # ALS iterations
            converged = False
            for it in range(80):
                if time.time() > deadline - 5:
                    break
                
                U = update_U(U, V, W, rank)
                V = update_V(U, V, W, rank)
                W = update_W(U, V, W, rank)
                
                max_err = check_error(U, V, W)
                if max_err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Polish with higher precision
                U, V, W, final_err = polish_solution(U, V, W, rank, max_iters=50)
                
                if final_err < 1e-9:
                    # Try to find exact rational solution
                    U_h, V_h, W_h = try_exact_rational(U, V, W, rank, max_trials=100)
                    if U_h is not None:
                        best_rank = rank
                        best_result = {
                            "U": U_h.tolist(),
                            "V": V_h.tolist(),
                            "W": W_h.tolist()
                        }
                        break
                    else:
                        # Use floating point solution if it's accurate enough
                        best_rank = rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        break
        
        if best_rank < n * m * p:
            break
    
    return best_result
# EVOLVE-BLOCK-END

