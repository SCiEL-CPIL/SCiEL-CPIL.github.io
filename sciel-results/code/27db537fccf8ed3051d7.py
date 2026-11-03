"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # Ensure float for gradient computation, but keep original for final cost
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    
    # 1. Initialization: Use a rank-based heuristic to get a decent starting point,
    #    then convert to a "soft" matrix or just start with identity + noise.
    #    Actually, a good start for PGD is often a random doubly stochastic matrix 
    #    or the one derived from the best discrete solution found quickly.
    #    Let's start with the rank matching solution to give PGD a head start.
    a_sum = a_float.sum(axis=1)
    b_sum = b_float.sum(axis=1)
    heavy_order = np.argsort(-a_sum, kind="stable")
    central_order = np.argsort(b_sum, kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy_order] = central_order
    
    # Convert p_init to a permutation matrix X
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        X[i, p_init[i]] = 1.0
        
    # 2. Projected Gradient Descent on Birkhoff Polytope
    # Objective: f(X) = sum_{i,j} a[i,j] * sum_{k,l} X[i,k] b[k,l] X[j,l]
    #            = trace(X^T A X B) if we define the QAP properly.
    #            Standard QAP: min sum_{i,j} a_{ij} d_{\pi(i)\pi(j)}
    #            Matrix form: min trace(X^T A X D) where X is perm matrix.
    #            Gradient: 2 (A X D + A^T X D^T) ? 
    #            Let's derive: f(X) = sum_{i,j,k,l} A_{ij} X_{ik} D_{kl} X_{jl}
    #            df/dX_{pq} = sum_{j,l} A_{pj} D_{ql} X_{jl} + sum_{i,k} A_{ip} X_{iq} D_{kl} X_{jl} ?? 
    #            Simpler: f(X) = <X, A X D> is not quite right due to indices.
            # f(X) = sum_{i,j} A_{ij} (X D X^T)_{ij} ? No.
            # (X D X^T)_{ij} = sum_{k,l} X_{ik} D_{kl} X_{jl}.
            # Yes, cost = sum_{i,j} A_{ij} (X D X^T)_{ij} = trace(A^T X D X^T) = trace(A X D X^T).
            # Gradient of trace(A X D X^T) w.r.t X:
            # d/dX trace(A X D X^T) = A^T X D^T + A X D.
            # If A and D are symmetric: 2 A X D.
            
    A = a_float
    D = b_float
    # Check symmetry. QAP usually has symmetric distance, non-symmetric flow.
    # But often both are treated generally.
    # Gradient G = A^T X D^T + A X D
    
    lr = 1.0 / (2.0 * np.linalg.norm(A, ord=np.inf) * np.linalg.norm(D, ord=np.inf))
    # Heuristic learning rate scaling
    lr *= 0.5
    
    def project_birkhoff(X):
        """Project onto Birkhoff polytope using Sinkhorn-Knopp iteration."""
        # Normalize rows and columns
        X = np.abs(X)
        # Add small epsilon to avoid zero division
        X += 1e-9
        for _ in range(100):  # Fixed iterations for speed, or until convergence
            row_sums = X.sum(axis=1, keepdims=True)
            X /= row_sums
            col_sums = X.sum(axis=1, keepdims=True) # Wait, axis=1 is cols? No, axis=1 is columns sum for each row?
            # np.sum(axis=1) sums along columns, returning row sums.
            # np.sum(axis=0) sums along rows, returning col sums.
            col_sums = X.sum(axis=0, keepdims=True)
            X /= col_sums
        # Ensure exact permutation-like structure if needed, but Sinkhorn gives doubly stochastic
        return X

    # We will do a few rounds of PGD, then recover integer solution, then maybe local search?
    # The prompt asks for a DIFFERENT strategy. PGD + Hungarian is a strong candidate.
    
    best_cost = np.inf
    best_p = p_init.copy()
    
    # Evaluate initial
    def calc_cost_int(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost_int(best_p)
    best_cost = current_cost
    
    # PGD Loop
    # To make this effective, we need to run PGD for a while, then convert to integer.
    # Since n=150, Sinkhorn is O(n^2 * iter). 150^2 = 22500. 100 iters = 2.25M ops.
    # Gradient calc: Matrix mults. 150^3 = 3.3M.
    # Total per step ~ 5-10M ops. In Python/NumPy, this is fast.
    # We can do thousands of steps in 115s.
    
    steps_per_phase = 500
    phases = 10
    
    for phase in range(phases):
        if time.time() > deadline - 10:
            break
            
        for step in range(steps_per_phase):
            if time.time() > deadline - 10:
                break
            
            # Calculate Gradient
            # G = A^T X D^T + A X D
            # To save time, assume symmetry if possible? 
            # Let's just compute it generally.
            # AXD = A @ X @ D
            # AXTDT = A.T @ X @ D.T
            
            # Optimization: Precompute A and D properties?
            # For n=150, direct matmul is fine.
            
            AX = A @ X
            AXD = AX @ D
            
            ATX = A.T @ X
            ATXDT = ATX @ D.T
            
            G = AXD + ATXDT
            
            # Gradient Step
            X -= lr * G
            
            # Project
            X = project_birkhoff(X)
            
            # Periodic evaluation and integer recovery
            if step % 50 == 0:
                # Convert X to permutation
                # Use Hungarian algorithm on cost matrix C where C[i,j] = ?
                # Actually, recovering from X is tricky. 
                # Better: Use the "nearest permutation" via Hungarian on a derived cost matrix?
                # Or simply: The current X is a mixture. 
                # A common trick: Use the rows of X as probabilities. 
                # But for QAP, the interaction term makes simple row-wise argmax suboptimal.
                
                # Alternative: Use the "Sinkhorn" output X directly in the objective to get a lower bound,
                # but we need an integer solution.
                
                # Let's use a "rounding" heuristic:
                # 1. Compute the "marginal" assignment cost for each i->j based on current X?
                # 2. Better: Use the Hungarian algorithm on the matrix M where M[i,j] = (A X D + A^T X D^T)[i,j]? No.
                
                # Standard approach: 
                # The objective is quadratic. 
                # We can approximate the linearization at current X:
                # L(X) = <G, X> + const.
                # So we can solve the linear assignment problem min <G, Y> s.t. Y in Birkhoff.
                # This is exactly the Hungarian algorithm on matrix G (or -G depending on sign).
                
                # Since we want to minimize f, and gradient descent minimizes, 
                # the linear approximation of f at X is f(X) + <G, X-Y>.
                # Minimizing this over Y in Birkhoff is equivalent to minimizing <G, Y>.
                # So we run Hungarian on G.
                
                # Note: G might not be the exact cost matrix for the linearization if we used a specific gradient form, 
                # but for trace(A X D X^T), the gradient is the correct linearization term.
                
                row_ind, col_ind = linear_sum_assignment(G)
                p_candidate = col_ind # p[i] = j means i assigned to j
                
                cost_candidate = calc_cost_int(p_candidate)
                
                if cost_candidate < best_cost:
                    best_cost = cost_candidate
                    best_p = p_candidate.copy()
                    
                    # Update X to be closer to this new integer solution?
                    # Or just let PGD continue from current X.
                    # Often, snapping X to the best found integer solution helps escape plateaus.
                    # Let's snap X to the best integer solution found so far.
                    X = np.zeros((n, n), dtype=np.float64)
                    for i in range(n):
                        X[i, best_p[i]] = 1.0
                        
    # Final check
    final_cost = calc_cost_int(best_p)
    if final_cost < best_cost:
        best_cost = final_cost
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
