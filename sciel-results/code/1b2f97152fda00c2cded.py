# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    """
    # Deadline guard
    t_start = time.time()
    deadline = 600.0  # safety margin before 630s kill

    # Problem definition
    n_nodes = 10
    n_edges = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute edge incidence for each demand in each direction
    # x_d = 0 means clockwise, x_d = 1 means counter-clockwise
    # For demand d with source s, target t, weight w:
    #   Clockwise uses edges s, s+1, ..., t-1 (mod n)
    #   Counter-clockwise uses all other edges
    
    # Build A matrix: A[e, d] = weight if demand d uses edge e in CCW direction
    #                    0 otherwise
    # If x_d = 1, load contribution is A[e,d]
    # If x_d = 0, load contribution is (total_weight_on_path_ccw - A[e,d])? 
    # No, let's think differently.
    
    # Let w_d be the weight of demand d.
    # Let C_d be the set of edges used if clockwise.
    # Let K_d be the set of edges used if counter-clockwise.
    # Note that C_d and K_d partition the set of all edges.
    
    # Load on edge e = sum_{d: e in C_d} w_d * (1 - x_d) + sum_{d: e in K_d} w_d * x_d
    #               = sum_{d: e in C_d} w_d - sum_{d: e in C_d} w_d * x_d + sum_{d: e in K_d} w_d * x_d
    #               = base_load_e + sum_d (A_ccw[e,d] - A_cw[e,d]) * x_d
    # where A_ccw[e,d] = w_d if e in K_d else 0
    #       A_cw[e,d]   = w_d if e in C_d else 0
    
    # Let's precompute:
    # base_load[e] = sum of w_d for all d such that e is in C_d (clockwise path)
    # coeff[e, d] = (A_ccw[e,d] - A_cw[e,d])
    
    base_load = np.zeros(n_edges)
    coeff = np.zeros((n_edges, D))
    
    for d_idx, (s, t, w) in enumerate(demands):
        # Determine clockwise edges
        cw_edges = set()
        e = s
        while e != t:
            cw_edges.add(e)
            e = (e + 1) % n_nodes
        
        # Determine counter-clockwise edges
        ccw_edges = set(range(n_edges)) - cw_edges
        
        for e in cw_edges:
            base_load[e] += w
            coeff[e, d_idx] = -w  # A_ccw is 0, A_cw is w, so diff is -w
        
        for e in ccw_edges:
            # base_load doesn't include this
            coeff[e, d_idx] = w   # A_ccw is w, A_cw is 0, so diff is w

    def objective(x):
        """
        x is a vector of length D in [0, 1].
        Returns a smooth approximation of max edge load.
        """
        # Load vector L = base_load + coeff @ x
        L = base_load + coeff @ x
        # Log-sum-exp approximation of max(L)
        # max(L) ≈ (1/alpha) * log(sum(exp(alpha * L)))
        alpha = 10.0
        # Numerical stability
        L_max = np.max(L)
        if L_max > 500:
            # Avoid overflow
            return L_max
        return L_max + np.log(np.sum(np.exp(alpha * (L - L_max)))) / alpha

    def gradient(x):
        """
        Gradient of the log-sum-exp objective.
        d/dx_i [ (1/alpha) log sum_j exp(alpha L_j) ]
        = sum_j exp(alpha L_j) / sum_k exp(alpha L_k) * alpha * dL_j/dx_i / alpha
        = sum_j softmax(alpha L)_j * dL_j/dx_i
        where softmax(alpha L)_j = exp(alpha L_j) / sum_k exp(alpha L_k)
        """
        L = base_load + coeff @ x
        alpha = 10.0
        L_max = np.max(L)
        exp_terms = np.exp(alpha * (L - L_max))
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return np.zeros(D)
        probs = exp_terms / sum_exp
        # grad = coeff.T @ probs
        return coeff.T @ probs

    # Multi-start optimization
    best_routing = None
    best_score = np.inf

    # Initial guesses
    x0_list = []
    x0_list.append(np.zeros(D))
    x0_list.append(np.ones(D))
    x0_list.append(np.full(D, 0.5))
    
    # Random starts
    np.random.seed(42)
    for _ in range(5):
        x0_list.append(np.random.rand(D))

    for x0 in x0_list:
        if time.time() > deadline - 60:
            break
        
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                jac=gradient,
                bounds=[(0, 1)] * D,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                x_opt = res.x
                # Round to nearest binary
                x_bin = np.round(x_opt)
                # Ensure values are 0 or 1
                x_bin[x_bin < 0.5] = 0
                x_bin[x_bin >= 0.5] = 1
                
                # Evaluate exact max load
                L = base_load + coeff @ x_bin
                max_load = np.max(L)
                
                if max_load < best_score:
                    best_score = max_load
                    best_routing = x_bin.copy()
                    
        except Exception:
            pass

    # If optimization failed or no improvement, use a greedy heuristic as fallback
    if best_routing is None or best_score > 100:
        # Greedy: for each demand, choose direction that minimizes current max load
        # This is a simple local search
        x = np.zeros(D)
        
        # Compute initial load
        L = base_load + coeff @ x
        current_max = np.max(L)
        
        improved = True
        while improved:
            improved = False
            for d in range(D):
                # Try flipping demand d
                x[d] = 1 - x[d]
                L_new = base_load + coeff @ x
                new_max = np.max(L_new)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    x[d] = 1 - x[d]  # revert
            
            if time.time() > deadline - 30:
                break

        if current_max < best_score:
            best_score = current_max
            best_routing = x.copy()

    # Final local search polish: try flipping each demand and combinations of 2
    if best_routing is not None:
        x = best_routing.copy()
        L = base_load + coeff @ x
        current_max = np.max(L)
        
        # Single flip
        for d in range(D):
            if time.time() > deadline - 10:
                break
            x[d] = 1 - x[d]
            L_new = base_load + coeff @ x
            new_max = np.max(L_new)
            if new_max < current_max:
                current_max = new_max
                best_routing = x.copy()
            else:
                x[d] = 1 - x[d]  # revert
        
        # Pair flips
        for d1 in range(D):
            for d2 in range(d1 + 1, D):
                if time.time() > deadline - 5:
                    break
                x[d1] = 1 - x[d1]
                x[d2] = 1 - x[d2]
                L_new = base_load + coeff @ x
                new_max = np.max(L_new)
                if new_max < current_max:
                    current_max = new_max
                    best_routing = x.copy()
                else:
                    x[d1] = 1 - x[d1]
                    x[d2] = 1 - x[d2]  # revert
            if time.time() > deadline - 5:
                break

    # Ensure best_routing is not None
    if best_routing is None:
        best_routing = np.zeros(D)

    return best_routing

# EVOLVE-BLOCK-END
