import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0  # Leave margin for 150s limit
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_solution = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook
    
    # Precompute indices for equations
    # Equations: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta(j, j')
    # for all i in [0,n), j in [0,m), j' in [0,m), k in [0,p)
    
    def check_solution(U, V, W, rank):
        # Verify exactness
        # U, V, W are numpy arrays
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j, r] * V[jp * p + k, r] * W[k * n + i, r]
                        target = 1.0 if j == jp else 0.0
                        if not np.isclose(s, target, atol=1e-9):
                            return False
        return True

    def snap_and_verify(U, V, W, rank):
        # Snap to nearest integer or half-integer
        # Multiply by 2, round, divide by 2
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        
        if check_solution(U_s, V_s, W_s, rank):
            return U_s, V_s, W_s
        return None

    def optimize_rank(R):
        # Random search + local optimization
        # Variables: U (nm x R), V (mp x R), W (pn x R)
        # Total vars: R * (nm + mp + pn)
        
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Random initialization
            # Use small integers to encourage rational solutions
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            W = np.random.choice([-1, 0, 1], size=(p * n, R))
            
            # If random init works, return it
            snapped = snap_and_verify(U, V, W, R)
            if snapped:
                return snapped
            
            # Gradient Descent
            # We minimize sum of squared errors
            # This is expensive to compute gradient analytically for each step
            # Let's do a simple numerical gradient or just use scipy if available
            # For speed, we will do a few steps of coordinate descent or random perturbation
            
            # Actually, for small R, we can try to solve a linear system if we fix two matrices?
            # No, it's cubic.
            
            # Let's try a simple iterative improvement:
            # Pick a random variable, perturb it, if error decreases, keep it.
            # This is slow.
            
            # Alternative: Use the fact that we can scale.
            # Let's just try more random inits with different distributions.
            # Sometimes random sparse matrices work for small ranks.
            
            # Try Gaussian init
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            snapped = snap_and_verify(U, V, W, R)
            if snapped:
                return snapped
            
            # Try to use scipy.optimize if time permits
            # But 50 attempts * 1000 steps might be too slow.
            # Let's rely on the fact that for R close to 30, solutions are dense and close to schoolbook.
            # We can start from schoolbook (padded with 0s) and do gradient descent.
            
            if R < rank_schoolbook:
                # Pad schoolbook to rank R? No, schoolbook is rank 30.
                # If R < 30, we can't start from schoolbook directly.
                # We can try to project?
                
                # Let's try a random init with values in [-0.5, 0.5]
                U = np.random.rand(n * m, R) - 0.5
                V = np.random.rand(m * p, R) - 0.5
                W = np.random.rand(p * n, R) - 0.5
                snapped = snap_and_verify(U, V, W, R)
                if snapped:
                    return snapped

        return None

    # Search for lower ranks
    for R in range(rank_schoolbook - 1, 15, -1):
        if time.time() > deadline:
            break
        
        res = optimize_rank(R)
        if res:
            U_s, V_s, W_s = res
            best_solution = {
                "U": U_s.tolist(),
                "V": V_s.tolist(),
                "W": W_s.tolist()
            }
            best_rank = R
            # We can stop if we find a very low rank, but let's try to go lower
            # However, lower is better.
            # But finding rank 20 might take long.
            # Let's continue to R=15 or until time limit.
            
    return best_solution
# EVOLVE-BLOCK-END
