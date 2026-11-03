"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Ensure arrays are float64 for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        """Compute the QAP cost for permutation p."""
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        bp = b_f[np.ix_(p, p)]
        return int(np.sum(a_f * bp))

    def compute_swap_delta(i, j, p):
        """
        Compute the change in cost when swapping locations of facilities i and j.
        p[i] and p[j] are the current locations.
        """
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0

        delta = 0.0

        # Diagonal terms
        delta += a_f[i, i] * (b_f[lj, lj] - b_f[li, li])
        delta += a_f[j, j] * (b_f[li, li] - b_f[lj, lj])

        # Cross terms between i and j
        delta += a_f[i, j] * (b_f[lj, li] - b_f[li, lj])
        delta += a_f[j, i] * (b_f[li, lj] - b_f[lj, li])

        # Terms with other facilities k (vectorized)
        # Create mask excluding i and j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]

        if len(ks) > 0:
            pk = p[ks]

            # a[i,k] * (b[lj,pk] - b[li,pk])
            delta += np.sum(a_f[i, ks] * (b_f[lj, pk] - b_f[li, pk]))
            # a[k,i] * (b[pk,lj] - b[pk,li])
            delta += np.sum(a_f[ks, i] * (b_f[pk, lj] - b_f[pk, li]))
            # a[j,k] * (b[li,pk] - b[lj,pk])
            delta += np.sum(a_f[j, ks] * (b_f[li, pk] - b_f[lj, pk]))
            # a[k,j] * (b[pk,li] - b[pk,lj])
            delta += np.sum(a_f[ks, j] * (b_f[pk, li] - b_f[pk, lj]))

        return delta

    def local_search(p, max_time=None, max_iter=20):
        """
        First-improvement local search with 2-opt swaps.
        Returns the best permutation found.
        """
        if max_time is None:
            max_time = deadline

        cost = compute_cost(p)
        improved = True
        iter_count = 0

        while improved and time.time() < max_time and iter_count < max_iter:
            improved = False
            iter_count += 1
            for i in range(n):
                if time.time() >= max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() >= max_time:
                        break
                    delta = compute_swap_delta(i, j, p)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break  # First improvement found, restart scan
                if improved:
                    break

        return p, cost

    def admm_solve(a_mat, b_mat, rho=100.0, max_iter=200, tol=1e-6):
        """
        Solve QAP using ADMM with Birkhoff polytope projection.
        Variables: X (assignment matrix, n x n)
        Objective: min trace(A X B X^T)
        Constraint: X in Birkhoff polytope (row/col sums = 1, X >= 0)
        
        ADMM formulation:
        min f(X) + g(Z)
        s.t. X = Z
        where f(X) = 0.5 * trace(A X B X^T) - <Lambda, X> + <U, X>
        and g(Z) = indicator of Birkhoff polytope
        
        Updates:
        X^{k+1} = argmin 0.5 * trace(A X B X^T) - <Lambda^k, X> + <U^k, X> + (rho/2)||X - Z^k + U^k/rho||^2
        Z^{k+1} = P_B(X^{k+1} + U^k/rho)
        U^{k+1} = U^k + X^{k+1} - Z^{k+1}
        
        For X update, we use gradient descent on the quadratic term + quadratic regularization.
        The gradient of 0.5 * trace(A X B X^T) w.r.t X is A X B + A^T X B^T (symmetrized).
        Actually, gradient of trace(A X B X^T) is A X B + A^T X B^T.
        Let M = A X B + A^T X B^T.
        The update is:
        X^{k+1} = (I - (1/rho) * (something)) ... this is complex.
        
        Simpler approach: Use a fixed number of gradient steps for X, then project Z.
        Or, since the Hessian is not constant, we can just do:
        X^{k+1} = X^k - alpha * (Gradient + rho * (X^k - Z^k + U^k/rho))
        
        Let's use a simple gradient step for X.
        """
        # Initialize X with a random doubly stochastic matrix
        # Start with uniform
        X = np.ones((n, n)) / n
        
        # U (dual variable)
        U = np.zeros((n, n))
        
        # Z (primal variable for constraint)
        Z = X.copy()
        
        # Precompute A^T and B^T
        a_t = a_mat.T
        b_t = b_mat.T
        
        for k in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # X-update: Gradient descent step
            # Gradient of 0.5 * trace(A X B X^T)
            # d/dX trace(A X B X^T) = A X B + A^T X B^T
            grad_obj = a_mat @ X @ b_mat + a_t @ X @ b_t
            grad_reg = rho * (X - Z + U / rho)
            grad_total = grad_obj + grad_reg
            
            # Step size
            # Use a small step size
            alpha = 0.1 / rho  # Heuristic step size
            
            X = X - alpha * grad_total
            
            # Z-update: Projection onto Birkhoff polytope
            # Z^{k+1} = P_B(X + U/rho)
            V = X + U / rho
            Z = project_birkhoff(V)
            
            # U-update
            U = U + X - Z
            
            # Check convergence
            if k % 10 == 0:
                primal_res = np.linalg.norm(X - Z)
                dual_res = np.linalg.norm(rho * (Z - Z_prev)) if k > 0 else 0
                if primal_res < tol and dual_res < tol:
                    break
            Z_prev = Z.copy()
        
        # Return the best permutation from X or Z
        # Use X to get the assignment
        return X, Z

    def project_birkhoff(V):
        """
        Project a matrix V onto the Birkhoff polytope.
        This is done by iteratively adjusting row and column sums to 1.
        """
        # Start with V, clip to non-negative
        X = np.maximum(V, 0)
        
        # Normalize rows and columns
        # Use Sinkhorn-Knopp or simple iterative scaling
        for _ in range(50):  # Fixed number of iterations
            # Row sums
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X = X / row_sums
            
            # Column sums
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            X = X / col_sums
            
        # Final normalization
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X = X / row_sums
        
        return X

    def extract_permutation(X):
        """
        Extract a permutation from a doubly stochastic matrix X.
        Use the Hungarian algorithm to find the best assignment.
        """
        # We want to maximize the sum of X[i, p[i]]
        # So we minimize -X
        row_ind, col_ind = linear_sum_assignment(-X)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    # Generate initial solutions using different methods
    
    best_p = None
    best_cost = float('inf')
    
    # 1. Rank matching
    heavy = np.argsort(-a_f.sum(axis=1), kind="stable")
    central = np.argsort(b_f.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    p_ls, cost_ls = local_search(p_rank.copy(), deadline - 10)
    if cost_ls < best_cost:
        best_p = p_ls.copy()
        best_cost = cost_ls
    
    # 2. Spectral / Linear Assignment
    if time.time() < deadline - 20:
        # Compute A = a @ b.T
        A = a_f @ b_f.T
        row_ind, col_ind = linear_sum_assignment(-A)
        p_spectral = np.empty(n, dtype=np.int64)
        p_spectral[row_ind] = col_ind
        
        p_ls, cost_ls = local_search(p_spectral.copy(), deadline - 10)
        if cost_ls < best_cost:
            best_p = p_ls.copy()
            best_cost = cost_ls
    
    # 3. ADMM-based initialization
    if time.time() < deadline - 30:
        try:
            X_admm, Z_admm = admm_solve(a_f, b_f, rho=50.0, max_iter=100)
            # Use X to extract permutation
            p_admm = extract_permutation(X_admm)
            
            p_ls, cost_ls = local_search(p_admm.copy(), deadline - 10)
            if cost_ls < best_cost:
                best_p = p_ls.copy()
                best_cost = cost_ls
        except Exception:
            pass
    
    # 4. Random restarts with local search
    rng = np.random.RandomState(42)
    while time.time() < deadline - 5:
        p_rand = rng.permutation(n)
        p_ls, cost_ls = local_search(p_rand.copy(), deadline - 5)
        if cost_ls < best_cost:
            best_p = p_ls.copy()
            best_cost = cost_ls
    
    # Fallback
    if best_p is None:
        best_p = np.arange(n)
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
