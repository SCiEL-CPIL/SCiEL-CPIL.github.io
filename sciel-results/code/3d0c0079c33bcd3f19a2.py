# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem: 10 nodes, 15 demands, minimize max edge load.
    Returns array of 15 values: >=0.5 means counter-clockwise, <0.5 means clockwise.
    """
    # Problem setup
    n = 10  # number of nodes/edges
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute which edges each demand uses in each direction
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
    # Counter-clockwise: all other edges
    
    # For each demand and each direction, store the set of edges used
    demand_edges = []  # demand_edges[i][0] = clockwise edges, demand_edges[i][1] = ccw edges
    for s, t, w in demands:
        # Clockwise path edges
        cw_edges = []
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n)) + list(range(0, t))
        
        # Counter-clockwise path edges (all edges not in cw path)
        all_edges = set(range(n))
        ccw_edges = list(all_edges - set(cw_edges))
        
        demand_edges.append((cw_edges, ccw_edges))
    
    def compute_loads(routing):
        """Compute edge loads given a routing decision array (0=cw, 1=ccw)."""
        loads = np.zeros(n)
        for i, (s, t, w) in enumerate(demands):
            if routing[i] < 0.5:  # clockwise
                for e in demand_edges[i][0]:
                    loads[e] += w
            else:  # counter-clockwise
                for e in demand_edges[i][1]:
                    loads[e] += w
        return loads
    
    def max_load(routing):
        """Return the maximum edge load."""
        loads = compute_loads(routing)
        return np.max(loads)
    
    # Phase 1: Greedy initialization
    # Start with all clockwise, then try flipping each demand if it reduces max load
    routing = np.zeros(D)
    
    # Try all-cw and all-ccw as baselines
    all_cw = np.zeros(D)
    all_ccw = np.ones(D)
    
    load_cw = max_load(all_cw)
    load_ccw = max_load(all_ccw)
    
    if load_ccw < load_cw:
        routing = all_ccw.copy()
    
    # Greedy improvement: try flipping each demand
    improved = True
    while improved:
        improved = False
        for i in range(D):
            trial = routing.copy()
            trial[i] = 1.0 - trial[i]  # flip
            if max_load(trial) < max_load(routing):
                routing = trial
                improved = True
    
    best_routing = routing.copy()
    best_load = max_load(best_routing)
    
    # Phase 2: Simulated annealing for broader search
    deadline = time.time() + 500  # 500 seconds deadline (safety margin from 630s)
    
    # Multiple restarts with different initializations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Use current best
            current = best_routing.copy()
        elif restart == 1:
            # All clockwise
            current = np.zeros(D)
        elif restart == 2:
            # All counter-clockwise
            current = np.ones(D)
        else:
            # Random initialization
            current = np.random.rand(D)
        
        current_load = max_load(current)
        
        # Simulated annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        stuck_count = 0
        max_stuck = 5000
        
        while T > T_min:
            if time.time() > deadline:
                break
            
            # Pick a random demand to flip
            i = np.random.randint(D)
            trial = current.copy()
            trial[i] = 1.0 - trial[i]
            trial_load = max_load(trial)
            
            delta = trial_load - current_load
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current = trial
                current_load = trial_load
                stuck_count = 0
                
                if current_load < best_load:
                    best_load = current_load
                    best_routing = current.copy()
            else:
                stuck_count += 1
            
            T *= cooling
            
            if stuck_count > max_stuck:
                break
        
        # Restart with perturbation of best if not improving
        if current_load >= best_load and restart > 2:
            # Perturb best solution
            current = best_routing.copy()
            for _ in range(3):
                i = np.random.randint(D)
                current[i] = 1.0 - current[i]
    
    # Phase 3: Local search polish - try all single flips and double flips
    # Single flips
    improved = True
    while improved:
        improved = False
        for i in range(D):
            trial = best_routing.copy()
            trial[i] = 1.0 - trial[i]
            if max_load(trial) < best_load:
                best_routing = trial
                best_load = max_load(best_routing)
                improved = True
    
    # Double flips
    improved = True
    while improved:
        improved = False
        for i in range(D):
            for j in range(i + 1, D):
                trial = best_routing.copy()
                trial[i] = 1.0 - trial[i]
                trial[j] = 1.0 - trial[j]
                load = max_load(trial)
                if load < best_load:
                    best_routing = trial
                    best_load = load
                    improved = True
    
    # Phase 4: Try a smooth optimization approach using soft-max
    # Use scipy to optimize a smooth approximation
    def soft_max_load(x, beta=100):
        """Smooth approximation of max load using log-sum-exp."""
        loads = compute_loads(x)
        # Soft max via log-sum-exp
        m = np.max(loads)
        if beta > 0:
            return m + np.log(np.sum(np.exp(beta * (loads - m)))) / beta
        return m
    
    # Convert to continuous variables in [0, 1]
    # Use the best discrete solution as initial point
    x0 = best_routing.copy()
    
    def objective(x):
        return soft_max_load(x, beta=50)
    
    # Bounds: each variable in [0, 1]
    bounds = [(0, 1)] * D
    
    # Multiple optimization attempts
    for opt_restart in range(10):
        if time.time() > deadline:
            break
        
        if opt_restart == 0:
            x_init = best_routing.copy()
        else:
            # Perturb best solution
            x_init = best_routing.copy()
            for _ in range(3):
                i = np.random.randint(D)
                x_init[i] = 1.0 - x_init[i]
        
        try:
            result = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-10})
            
            # Round to nearest discrete value
            x_opt = result.x
            x_discrete = np.where(x_opt >= 0.5, 1.0, 0.0)
            
            load_opt = max_load(x_discrete)
            if load_opt < best_load:
                best_load = load_opt
                best_routing = x_discrete
        except Exception:
            pass
    
    # Final local search polish
    improved = True
    while improved:
        improved = False
        for i in range(D):
            trial = best_routing.copy()
            trial[i] = 1.0 - trial[i]
            if max_load(trial) < best_load:
                best_routing = trial
                best_load = max_load(best_routing)
                improved = True
    
    # Ensure output is in the correct format: >=0.5 for ccw, <0.5 for cw
    # Our convention: 0 = clockwise, 1 = counter-clockwise
    # So we return values where 0 -> cw (0 < 0.5) and 1 -> ccw (1 >= 0.5)
    return best_routing

# EVOLVE-BLOCK-END
