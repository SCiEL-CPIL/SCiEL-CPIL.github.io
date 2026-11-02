import numpy as np
from scipy.optimize import least_squares
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m  # 6
    size_V = m * p  # 15
    size_W = p * n  # 10
    
    # Build target tensor indices
    targets_alpha = []
    targets_beta = []
    targets_gamma = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets_alpha.append(a)
                targets_beta.append(b)
                targets_gamma.append(c)
                
    targets_alpha = np.array(targets_alpha, dtype=int)
    targets_beta = np.array(targets_beta, dtype=int)
    targets_gamma = np.array(targets_gamma, dtype=int)
    
    num_targets = len(targets_alpha)
    
    # Precompute all zero entries for penalty
    all_indices_a = np.arange(size_U)
    all_indices_b = np.arange(size_V)
    all_indices_c = np.arange(size_W)
    
    # Create full index arrays for zero constraints
    # We'll use a subset of zero entries to keep it tractable
    # Total entries: 6*15*10 = 900, non-zero: 30, zero: 870
    
    def compute_full_residuals(U, V, W):
        """Compute residuals for all entries (non-zero targets and zero entries)"""
        # Compute full tensor
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Build target tensor
        T_target = np.zeros((size_U, size_V, size_W))
        T_target[targets_alpha, targets_beta, targets_gamma] = 1.0
        
        # Flatten difference
        diff = (T_full - T_target).flatten()
        return diff
    
    def residuals_from_flat(x, R):
        """Compute residuals from flattened parameter vector"""
        U = x[:size_U*R].reshape(size_U, R)
        V = x[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
        W = x[(size_U+size_V)*R:].reshape(size_W, R)
        return compute_full_residuals(U, V, W)
    
    def verify_solution(U, V, W):
        """Verify if the solution is exact (within tolerance)"""
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((size_U, size_V, size_W))
        T_target[targets_alpha, targets_beta, targets_gamma] = 1.0
        max_err = np.max(np.abs(T_full - T_target))
        return max_err < 1e-9, max_err
    
    def try_rank_R(R, num_restarts=10, max_nfev=200):
        """Try to find a decomposition of rank R"""
        num_vars = (size_U + size_V + size_W) * R
        
        for restart in range(num_restarts):
            # Different initialization strategies
            if restart == 0:
                # Random small integers
                x0 = np.random.randint(-2, 3, num_vars).astype(float)
            elif restart == 1:
                # Random with smaller magnitude
                x0 = np.random.uniform(-1, 1, num_vars)
            elif restart == 2:
                # Structured: try to build from schoolbook and perturb
                # Initialize with a partial schoolbook-like structure
                x0 = np.zeros(num_vars)
                # Add some random structure
                for col in range(min(R, 5)):
                    idx = np.random.choice(size_U, 2, replace=False)
                    x0[idx*0 + col] = 1.0  # This is wrong indexing, let me fix
                # Actually, let's just use random with different seeds
                np.random.seed(restart * 12345)
                x0 = np.random.randint(-3, 4, num_vars).astype(float)
            else:
                # More random restarts with varying scales
                scale = 0.5 + restart * 0.1
                x0 = np.random.uniform(-scale, scale, num_vars)
            
            try:
                res = least_squares(
                    lambda x: residuals_from_flat(x, R),
                    x0,
                    method='trf',
                    max_nfev=max_nfev,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10
                )
                
                if res.cost < 1e-8:
                    # Try to snap to half-integers
                    x_sol = res.x
                    x_snapped = np.round(x_sol * 2) / 2.0
                    
                    U = x_snapped[:size_U*R].reshape(size_U, R)
                    V = x_snapped[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
                    W = x_snapped[(size_U+size_V)*R:].reshape(size_W, R)
                    
                    is_valid, max_err = verify_solution(U, V, W)
                    if is_valid:
                        return U.tolist(), V.tolist(), W.tolist()
                        
                    # If snapping didn't work, try without snapping
                    is_valid, max_err = verify_solution(U, V, W)
                    if is_valid:
                        return U.tolist(), V.tolist(), W.tolist()
                        
            except Exception:
                pass
        
        return None
    
    start_time = time.time()
    deadline = 110  # Leave 10s safety margin
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = 15 for non-trivial
    # Upper bound: n*m*p = 30 (schoolbook)
    for R in range(10, 31):
        if time.time() > deadline:
            break
            
        # Try this rank
        result = try_rank_R(R, num_restarts=5, max_nfev=150)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
