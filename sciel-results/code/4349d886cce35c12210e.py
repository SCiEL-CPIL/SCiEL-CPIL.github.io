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
    
    deadline = time.time() + 130  # safety margin before 150s kill
    
    best_rank = n * m * p  # start with schoolbook
    best_result = None
    
    # First, provide the schoolbook as fallback
    schoolbook_U = np.zeros((nm, best_rank))
    schoolbook_V = np.zeros((mp, best_rank))
    schoolbook_W = np.zeros((pn, best_rank))
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
    
    np.random.seed(42)
    
    def check_solution(U, V, W, tol=1e-9):
        """Check if U, V, W form an exact decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err < tol
    
    def try_round_to_half_int(U, V, W, max_trials=50):
        """Try to find a half-integer solution near the given solution."""
        # First try direct rounding
        for scale in [2, 4, 8, 16]:
            U_round = np.round(U * scale) / scale
            V_round = np.round(V * scale) / scale
            W_round = np.round(W * scale) / scale
            if check_solution(U_round, V_round, W_round, 1e-8):
                return U_round, V_round, W_round
        
        # Try local search with perturbations
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            noise_scale = 0.01 * (trial + 1)
            U_t = np.round((U + np.random.randn(nm, U.shape[1]) * noise_scale) * 2) / 2
            V_t = np.round((V + np.random.randn(mp, V.shape[1]) * noise_scale) * 2) / 2
            W_t = np.round((W + np.random.randn(pn, W.shape[1]) * noise_scale) * 2) / 2
            if check_solution(U_t, V_t, W_t, 1e-8):
                return U_t, V_t, W_t
        return None
    
    def als_solve(U, V, W, rank, max_iters=100, tol=1e-10):
        """Run ALS iterations to converge."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Fix V and W, solve for U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i * m + j, :] = x
                    except:
                        pass
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j * p + k, :] = x
                    except:
                        pass
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k * n + i, :] = x
                    except:
                        pass
            
            # Check convergence
            if check_solution(U, V, W, tol):
                return True
        
        return False
    
    # Try ranks from lower values upward
    # For 4x5x6, theoretical lower bound is around 40-60
    # Try from 50 up to 119
    for rank in range(50, 120):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts with different initialization strategies
        for restart in range(8):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Small random
                U = np.random.randn(nm, rank) * 0.3
                V = np.random.randn(mp, rank) * 0.3
                W = np.random.randn(pn, rank) * 0.3
            elif restart == 1:
                # Larger random
                U = np.random.randn(nm, rank) * 0.8
                V = np.random.randn(mp, rank) * 0.8
                W = np.random.randn(pn, rank) * 0.8
            elif restart == 2:
                # Sparse-like initialization
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                for c in range(rank):
                    idx_u = np.random.randint(0, nm)
                    idx_v = np.random.randint(0, mp)
                    idx_w = np.random.randint(0, pn)
                    U[idx_u, c] = np.random.choice([-1, 1])
                    V[idx_v, c] = np.random.choice([-1, 1])
                    W[idx_w, c] = np.random.choice([-1, 1])
            elif restart == 3:
                # Structured initialization based on Kronecker product idea
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = (i * m + j) * p + k
                            if c < rank:
                                U[i * m + j, c] = 1.0
                                V[j * p + k, c] = 1.0
                                W[k * n + i, c] = 1.0
                # Add random perturbation
                U += np.random.randn(nm, rank) * 0.1
                V += np.random.randn(mp, rank) * 0.1
                W += np.random.randn(pn, rank) * 0.1
            else:
                # Random with varying scales
                scale = 0.2 + 0.1 * restart
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
            
            # Run ALS
            if als_solve(U, V, W, rank):
                # Found a solution! Try to get exact half-integer version
                result = try_round_to_half_int(U, V, W)
                if result is not None:
                    U_round, V_round, W_round = result
                    best_rank = rank
                    best_result = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    break  # found for this rank, go to lower ranks
        
        if best_rank < n * m * p:
            break  # found a better rank, stop
    
    if time.time() > deadline - 2:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
