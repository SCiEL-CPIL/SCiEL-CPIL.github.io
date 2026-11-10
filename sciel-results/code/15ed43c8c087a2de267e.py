# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    The baseline is the uniform distribution over the grid.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    # Initialize with uniform distribution
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Precompute indices for slopes
    x_vals = np.arange(n)
    y_vals = np.arange(n)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Slopes: -1, 0, 1, 2
    # Values: x - y, x, x + y, x + 2y
    V_m1 = X - Y
    V_0 = X
    V_1 = X + Y
    V_2 = X + 2 * Y
    
    # Flatten for optimization
    # We optimize over log P to ensure positivity
    # P = exp(u) / sum(exp(u))
    
    def get_P(u):
        u_max = np.max(u)
        exp_u = np.exp(u - u_max)
        P = exp_u / np.sum(exp_u)
        return P
    
    def compute_entropies(P):
        # P is 10x10
        # H_{-1}
        def entropy_from_values(P, V):
            # V is 10x10
            min_val = np.min(V)
            max_val = np.max(V)
            num_vals = int(max_val - min_val + 1)
            probs = np.zeros(num_vals)
            for i in range(num_vals):
                val = min_val + i
                mask = (V == val)
                probs[i] = np.sum(P[mask])
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        
        H_m1 = entropy_from_values(P, V_m1)
        H_0 = entropy_from_values(P, V_0)
        H_1 = entropy_from_values(P, V_1)
        H_2 = entropy_from_values(P, V_2)
        
        return H_m1, H_0, H_1, H_2

    def objective(u):
        P = get_P(u)
        H_m1, H_0, H_1, H_2 = compute_entropies(P)
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return -1e9
        return - (H_m1 / M)

    # Start from uniform
    u0 = np.zeros((n, n))
    
    # Use L-BFGS-B or SLSQP? 
    # L-BFGS-B is good for large problems, but we have 100 vars.
    # We don't have bounds on u, but P is bounded.
    # Let's try a few random restarts and local optimizations.
    
    best_P = P
    best_score = compute_entropies(P)[0] / max(compute_entropies(P)[1], compute_entropies(P)[2], compute_entropies(P)[3])
    
    start_time = time.time()
    deadline = 590.0
    
    # Define a gradient ascent step for the ratio
    # Since objective is non-smooth at max switch, we use a subgradient approach or just try to maximize H_m1 - lambda * M
    # Let's use a simple iterative improvement:
    # 1. Maximize H_m1 subject to M <= current_M? No, we want to increase ratio.
    
    # Let's use scipy.optimize.minimize on the negative ratio, but handle the non-differentiability.
    # We can use Nelder-Mead for small subspace? No.
    # Let's use a stochastic gradient ascent on u.
    
    def grad_objective(u):
        P = get_P(u)
        H_m1, H_0, H_1, H_2 = compute_entropies(P)
        M = max(H_0, H_1, H_2)
        # Gradient of H_m1 wrt u
        # dH/dP = -log P - 1
        # dP/du = P (1 - P) for softmax? 
        # If P = softmax(u), then dP_i/du_j = P_i (delta_ij - P_j)
        # dH/du = sum_k (dH/dP_k) (dP_k/du_j)
        # = sum_k (-log P_k - 1) P_k (delta_kj - P_j)
        # = (-log P_j - 1) P_j - P_j sum_k (-log P_k - 1) P_k
        # = P_j (-log P_j - 1) - P_j (-H + 1)  (since H = -sum P log P)
        # = -P_j log P_j - P_j + P_j H - P_j
        # = -P_j log P_j + P_j (H - 2)
        
        def grad_entropy(u, V):
            # Compute gradient of entropy of projection V
            P = get_P(u)
            # We need dH_V/du
            # H_V = - sum_v p_v log p_v
            # p_v = sum_{(x,y): V(x,y)=v} P_{x,y}
            # dp_v/du_j = sum_{(x,y): V(x,y)=v} dP_{x,y}/du_j
            # dP_{x,y}/du_j = P_{x,y} (delta_{(x,y),j} - P_j)
            
            # This is complex to vectorize directly for arbitrary V.
            # Let's approximate or use finite differences for gradient?
            # 100 dims, finite difference is 200 evaluations per step.
            # Too slow for many steps.
            
            # Let's use the analytic gradient formula for H wrt P, then chain rule.
            # dH/du_j = -log P_j - 1 + C_j
            # where C_j = sum_k (dH/dP_k) dP_k/du_j? No.
            
            # Let's just use a simple heuristic:
            # Increase P where it helps H_m1 and decreases M.
            
            # For now, let's just do a random search / hill climbing.
            return None

    # Given time constraints, let's try a few random perturbations and local polish.
    # Or use SLSQP on a reduced set of variables?
    
    # Let's try to maximize H_m1 - 0.1 * M using gradient ascent with finite differences for the gradient of M.
    
    def compute_M(P):
        _, H_0, H_1, H_2 = compute_entropies(P)
        return max(H_0, H_1, H_2)

    def compute_H_m1(P):
        H_m1, _, _, _ = compute_entropies(P)
        return H_m1

    # Simple Hill Climbing
    u = u0.copy()
    step_size = 0.1
    for _ in range(1000):
        if time.time() - start_time > deadline:
            break
        
        P = get_P(u)
        H_m1 = compute_H_m1(P)
        M = compute_M(P)
        score = H_m1 / M if M > 1e-9 else 0
        
        if score > best_score:
            best_score = score
            best_P = P.copy()
            
        # Try a random direction
        du = np.random.normal(0, step_size, (n, n))
        u_new = u + du
        P_new = get_P(u_new)
        H_m1_new = compute_H_m1(P_new)
        M_new = compute_M(P_new)
        score_new = H_m1_new / M_new if M_new > 1e-9 else 0
        
        if score_new > score:
            u = u_new
            step_size = min(step_size * 1.1, 1.0)
        else:
            step_size = max(step_size * 0.9, 0.01)
            
    return best_P

# EVOLVE-BLOCK-END
