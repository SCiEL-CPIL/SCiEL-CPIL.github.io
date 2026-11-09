# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        value < 0.5 -> route the demand clockwise
        value >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    # Problem parameters
    n_nodes = 10
    n_edges = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge usage matrices
    # For each demand d, and for each direction (CW=0, CCW=1), which edges are used
    # A[d, dir, e] = 1 if demand d uses edge e when routed in direction dir
    A = np.zeros((D, 2, n_edges))
    
    for d, (s, t, w) in enumerate(demands):
        # Clockwise: edges s, s+1, ..., t-1 (mod n_nodes)
        # If s <= t: edges s, s+1, ..., t-1
        # If s > t: edges s, s+1, ..., n-1, 0, ..., t-1
        cw_edges = []
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            A[d, 0, e] = 1.0
        
        # Counter-clockwise: all other edges
        for e in range(n_edges):
            if A[d, 0, e] == 0.0:
                A[d, 1, e] = 1.0
    
    weights = np.array([w for (_, _, w) in demands])
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580.0  # 580s out of 630s budget
    
    def compute_loads(routing_binary):
        """Compute edge loads given binary routing decisions."""
        # routing_binary: array of shape (D,) with 0 or 1
        # Load on edge e = sum over d of weight[d] * (A[d, 0, e] if routing_binary[d]==0 else A[d, 1, e])
        # = sum_d weight[d] * (A[d, 0, e] * (1 - routing_binary[d]) + A[d, 1, e] * routing_binary[d])
        # = sum_d weight[d] * A[d, 0, e] - sum_d weight[d] * A[d, 0, e] * routing_binary[d] + sum_d weight[d] * A[d, 1, e] * routing_binary[d]
        # = base_load_e + sum_d weight[d] * (A[d, 1, e] - A[d, 0, e]) * routing_binary[d]
        
        # Precompute base loads (all CW)
        base_loads = np.zeros(n_edges)
        for d in range(D):
            base_loads += weights[d] * A[d, 0, :]
        
        # Difference matrix: diff[d, e] = weight[d] * (A[d, 1, e] - A[d, 0, e])
        diff = np.zeros((D, n_edges))
        for d in range(D):
            diff[d, :] = weights[d] * (A[d, 1, :] - A[d, 0, :])
        
        # Loads = base_loads + diff @ routing_binary
        loads = base_loads + diff @ routing_binary
        return loads
    
    def smooth_max_load(x, alpha):
        """
        Smooth approximation of max edge load using log-sum-exp.
        x: continuous routing variables in [0, 1]
        alpha: sharpness parameter
        """
        # Compute loads for continuous x
        # x_d is the fraction routed CCW
        # Load on edge e = sum_d weight[d] * (A[d, 0, e] * (1 - x_d) + A[d, 1, e] * x_d)
        loads = np.zeros(n_edges)
        for e in range(n_edges):
            for d in range(D):
                loads[e] += weights[d] * (A[d, 0, e] * (1 - x[d]) + A[d, 1, e] * x[d])
        
        # Smooth max: (1/alpha) * log(sum(exp(alpha * loads)))
        # Use stable log-sum-exp
        max_load = np.max(loads)
        if alpha == 0:
            return max_load
        shifted = alpha * (loads - max_load)
        log_sum_exp = max_load * alpha + np.log(np.sum(np.exp(shifted)))
        return log_sum_exp / alpha
    
    def smooth_max_load_grad(x, alpha):
        """Gradient of smooth max load."""
        loads = np.zeros(n_edges)
        for e in range(n_edges):
            for d in range(D):
                loads[e] += weights[d] * (A[d, 0, e] * (1 - x[d]) + A[d, 1, e] * x[d])
        
        max_load = np.max(loads)
        shifted = alpha * (loads - max_load)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        probs = exp_shifted / sum_exp  # Softmax weights
        
        # d/dx_d of smooth_max = sum_e probs[e] * d(loads[e])/dx_d
        # d(loads[e])/dx_d = weight[d] * (A[d, 1, e] - A[d, 0, e])
        grad = np.zeros(D)
        for d in range(D):
            for e in range(n_edges):
                grad[d] += probs[e] * weights[d] * (A[d, 1, e] - A[d, 0, e])
        return grad
    
    # Try multiple alpha values and initializations
    best_load = np.inf
    best_routing = np.zeros(D)
    
    # Initial solutions to try
    initial_solutions = [
        np.zeros(D),  # all CW
        np.ones(D),   # all CCW
        np.random.rand(D),  # random
    ]
    
    # Add greedy initial solution
    greedy = np.zeros(D)
    for d in range(D):
        # Try both directions, pick the one that gives lower max load
        loads_cw = compute_loads(np.concatenate([greedy, [0]]))
        loads_ccw = compute_loads(np.concatenate([greedy, [1]]))
        if np.max(loads_cw) <= np.max(loads_ccw):
            greedy[d] = 0
        else:
            greedy[d] = 1
    initial_solutions.append(greedy)
    
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for alpha in alphas:
        for init in initial_solutions:
            if time.time() > deadline:
                break
            
            # Bounds for L-BFGS-B
            bounds = [(0.0, 1.0)] * D
            
            # Objective function
            def objective(x):
                return smooth_max_load(x, alpha)
            
            # Gradient
            def gradient(x):
                return smooth_max_load_grad(x, alpha)
            
            try:
                result = minimize(
                    objective,
                    init,
                    method='L-BFGS-B',
                    jac=gradient,
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            if result.success or result.fun < best_load:
                x_opt = result.x
                # Round to binary
                routing_binary = (x_opt >= 0.5).astype(float)
                loads = compute_loads(routing_binary)
                max_load = np.max(loads)
                
                if max_load < best_load:
                    best_load = max_load
                    best_routing = routing_binary.copy()
        
        if time.time() > deadline:
            break
    
    # Local search polish: try flipping each demand
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for d in range(D):
            old_val = best_routing[d]
            best_routing[d] = 1 - old_val
            loads = compute_loads(best_routing)
            max_load = np.max(loads)
            if max_load < best_load:
                best_load = max_load
                improved = True
            else:
                best_routing[d] = old_val
    
    # Ensure output is in [0, 1] with correct semantics
    # 0 = CW, 1 = CCW
    return best_routing

# EVOLVE-BLOCK-END
