# EVOLVE-BLOCK-START
import numpy as np
import time
from math import log, exp, floor


def _build_routes(n=10):
    """Precompute which edges each demand uses for CW and CCW routing."""
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        
        cw_set = set(cw)
        ccw = [e for e in range(n) if e not in cw_set]
        ccw_edges.append(ccw)
    
    return cw_edges, ccw_edges, demands


def _compute_loads(routing, cw_edges, ccw_edges, demands):
    """Compute edge loads given a routing vector."""
    n = 10
    D = len(demands)
    loads = np.zeros(n)
    
    for d in range(D):
        s, t, w = demands[d]
        if routing[d] >= 0.5:
            for e in ccw_edges[d]:
                loads[e] += w
        else:
            for e in cw_edges[d]:
                loads[e] += w
    
    return loads


def _compute_max_load(routing, cw_edges, ccw_edges, demands):
    """Compute just the maximum load for a routing."""
    loads = _compute_loads(routing, cw_edges, ccw_edges, demands)
    return np.max(loads)


def _local_search(routing, cw_edges, ccw_edges, demands, deadline, max_iter=100000):
    """Local search with 1-flip moves using incremental load updates."""
    D = len(demands)
    n = 10
    best_routing = routing.copy()
    loads = _compute_loads(best_routing, cw_edges, ccw_edges, demands)
    best_max = np.max(loads)
    
    improved = True
    iters = 0
    while improved and (time.time() < deadline) and (iters < max_iter):
        improved = False
        iters += 1
        for d in range(D):
            s, t, w = demands[d]
            # Compute delta loads for flipping demand d
            if best_routing[d] >= 0.5:
                # Currently CCW, would flip to CW
                # Remove CCW contribution, add CW contribution
                for e in ccw_edges[d]:
                    loads[e] -= w
                for e in cw_edges[d]:
                    loads[e] += w
            else:
                # Currently CW, would flip to CCW
                for e in cw_edges[d]:
                    loads[e] -= w
                for e in ccw_edges[d]:
                    loads[e] += w
            
            trial_max = np.max(loads)
            
            if trial_max < best_max - 1e-9:
                best_routing[d] = 1.0 - best_routing[d]
                best_max = trial_max
                improved = True
            else:
                # Revert
                if best_routing[d] >= 0.5:
                    # We just flipped to CCW in the trial, so revert back to CW
                    for e in ccw_edges[d]:
                        loads[e] -= w
                    for e in cw_edges[d]:
                        loads[e] += w
                else:
                    # We just flipped to CW in the trial, so revert back to CCW
                    for e in cw_edges[d]:
                        loads[e] -= w
                    for e in ccw_edges[d]:
                        loads[e] += w
    
    return best_routing, best_max


def _pair_local_search(routing, cw_edges, ccw_edges, demands, deadline, max_iter=50000):
    """Local search with 2-flip moves using incremental load updates."""
    D = len(demands)
    n = 10
    best_routing = routing.copy()
    loads = _compute_loads(best_routing, cw_edges, ccw_edges, demands)
    best_max = np.max(loads)
    
    improved = True
    iters = 0
    while improved and (time.time() < deadline) and (iters < max_iter):
        improved = False
        iters += 1
        for d1 in range(D):
            if time.time() > deadline:
                break
            for d2 in range(d1 + 1, D):
                s1, t1, w1 = demands[d1]
                s2, t2, w2 = demands[d2]
                
                # Apply both flips
                if best_routing[d1] >= 0.5:
                    for e in ccw_edges[d1]:
                        loads[e] -= w1
                    for e in cw_edges[d1]:
                        loads[e] += w1
                else:
                    for e in cw_edges[d1]:
                        loads[e] -= w1
                    for e in ccw_edges[d1]:
                        loads[e] += w1
                
                if best_routing[d2] >= 0.5:
                    for e in ccw_edges[d2]:
                        loads[e] -= w2
                    for e in cw_edges[d2]:
                        loads[e] += w2
                else:
                    for e in cw_edges[d2]:
                        loads[e] -= w2
                    for e in ccw_edges[d2]:
                        loads[e] += w2
                
                trial_max = np.max(loads)
                
                if trial_max < best_max - 1e-9:
                    best_routing[d1] = 1.0 - best_routing[d1]
                    best_routing[d2] = 1.0 - best_routing[d2]
                    best_max = trial_max
                    improved = True
                    break
                else:
                    # Revert both
                    if best_routing[d1] >= 0.5:
                        for e in ccw_edges[d1]:
                            loads[e] -= w1
                        for e in cw_edges[d1]:
                            loads[e] += w1
                    else:
                        for e in cw_edges[d1]:
                            loads[e] -= w1
                        for e in ccw_edges[d1]:
                            loads[e] += w1
                    
                    if best_routing[d2] >= 0.5:
                        for e in ccw_edges[d2]:
                            loads[e] -= w2
                        for e in cw_edges[d2]:
                            loads[e] += w2
                    else:
                        for e in cw_edges[d2]:
                            loads[e] -= w2
                        for e in ccw_edges[d2]:
                            loads[e] += w2
    
    return best_routing, best_max


def _simulated_annealing(routing, cw_edges, ccw_edges, demands, deadline, T0=1.0, cooling=0.9995, max_iter=200000):
    """Simulated annealing for escaping local optima with incremental updates."""
    D = len(demands)
    n = 10
    current_routing = routing.copy()
    loads = _compute_loads(current_routing, cw_edges, ccw_edges, demands)
    current_max = np.max(loads)
    best_routing = current_routing.copy()
    best_max = current_max
    
    T = T0
    np_random = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        d = np_random.integers(0, D)
        s, t, w = demands[d]
        
        # Apply flip
        if current_routing[d] >= 0.5:
            for e in ccw_edges[d]:
                loads[e] -= w
            for e in cw_edges[d]:
                loads[e] += w
        else:
            for e in cw_edges[d]:
                loads[e] -= w
            for e in ccw_edges[d]:
                loads[e] += w
        
        trial_max = np.max(loads)
        delta = trial_max - current_max
        
        if delta < 0 or np_random.random() < np.exp(-delta / max(T, 1e-10)):
            current_routing[d] = 1.0 - current_routing[d]
            current_max = trial_max
            if current_max < best_max - 1e-9:
                best_routing = current_routing.copy()
                best_max = current_max
        else:
            # Revert
            if current_routing[d] >= 0.5:
                for e in ccw_edges[d]:
                    loads[e] -= w
                for e in cw_edges[d]:
                    loads[e] += w
            else:
                for e in cw_edges[d]:
                    loads[e] -= w
                for e in ccw_edges[d]:
                    loads[e] += w
        
        T *= cooling
    
    return best_routing, best_max


def _logsumexp_smooth_max(routing, cw_edges, ccw_edges, demands, beta=10.0):
    """
    Compute a smooth approximation of the max load using log-sum-exp.
    This provides a differentiable surrogate for local optimization.
    """
    loads = _compute_loads(routing, cw_edges, ccw_edges, demands)
    # log-sum-exp: log(sum(exp(beta * loads))) / beta approximates max(loads)
    max_load = np.max(loads)
    if max_load == 0:
        return 0.0
    shifted = beta * (loads - max_load)
    return max_load + np.log(np.sum(np.exp(shifted))) / beta


def _smooth_gradient_descent(routing, cw_edges, ccw_edges, demands, deadline, beta=5.0, lr=0.01, max_iter=5000):
    """
    Use gradient descent on the smooth log-sum-exp surrogate to find
    a good continuous solution, then threshold to binary.
    """
    D = len(demands)
    n = 10
    x = routing.copy().astype(float)
    # Map to [0, 1] range for continuous optimization
    # x[d] = 0 means CW, x[d] = 1 means CCW
    # Load on edge e = sum over demands of w_d * (x_d if e in ccw_edges[d] else (1-x_d) if e in cw_edges[d] else 0)
    
    # Precompute for each edge, which demands affect it and how
    # For edge e: load_e = sum_{d: e in ccw_edges[d]} w_d * x_d + sum_{d: e in cw_edges[d]} w_d * (1 - x_d)
    
    # Build sparse representation
    edge_ccw_demands = [[] for _ in range(n)]  # (d, w) for each edge
    edge_cw_demands = [[] for _ in range(n)]
    
    for d in range(D):
        s, t, w = demands[d]
        for e in ccw_edges[d]:
            edge_ccw_demands[e].append((d, w))
        for e in cw_edges[d]:
            edge_cw_demands[e].append((d, w))
    
    best_x = x.copy()
    best_smooth = _logsumexp_smooth_max(x, cw_edges, ccw_edges, demands, beta)
    best_max = _compute_max_load(x, cw_edges, ccw_edges, demands)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute loads
        loads = np.zeros(n)
        for e in range(n):
            load = 0.0
            for d, w in edge_ccw_demands[e]:
                load += w * x[d]
            for d, w in edge_cw_demands[e]:
                load += w * (1.0 - x[d])
            loads[e] = load
        
        # Compute gradient of log-sum-exp
        max_load = np.max(loads)
        if max_load == 0:
            grad = np.zeros(D)
        else:
            shifted = beta * (loads - max_load)
            exp_shifted = np.exp(shifted)
            sum_exp = np.sum(exp_shifted)
            # d/dx_d of (1/beta) * log(sum_e exp(beta * load_e))
            # = (1/beta) * sum_e exp(beta*load_e)/sum_exp * beta * d(load_e)/dx_d
            # = sum_e exp(beta*load_e)/sum_exp * d(load_e)/dx_d
            grad = np.zeros(D)
            for e in range(n):
                weight = exp_shifted[e] / sum_exp
                for d, w in edge_ccw_demands[e]:
                    grad[d] += weight * w
                for d, w in edge_cw_demands[e]:
                    grad[d] -= weight * w
        
        # Gradient descent step
        x = x - lr * grad
        # Project to [0, 1]
        x = np.clip(x, 0.0, 1.0)
        
        # Evaluate
        smooth_val = _logsumexp_smooth_max(x, cw_edges, ccw_edges, demands, beta)
        max_val = _compute_max_load(x, cw_edges, ccw_edges, demands)
        
        if max_val < best_max - 1e-9:
            best_max = max_val
            best_x = x.copy()
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
    
    # Threshold to binary
    best_routing = (best_x >= 0.5).astype(float)
    return best_routing, best_max


def _genetic_algorithm(cw_edges, ccw_edges, demands, deadline, pop_size=50, max_gen=500):
    """Simple genetic algorithm."""
    D = len(demands)
    np_random = np.random.default_rng(123)
    
    # Initialize population
    pop = np.zeros((pop_size, D))
    for i in range(pop_size):
        pop[i] = np_random.random(D)
    
    def fitness(r):
        return -_compute_max_load(r, cw_edges, ccw_edges, demands)
    
    best_routing = None
    best_max = float('inf')
    
    for gen in range(max_gen):
        if time.time() > deadline:
            break
        
        # Evaluate fitness
        fits = np.array([fitness(r) for r in pop])
        idx = np.argmax(fits)
        if fits[idx] > -best_max + 1e-9:
            best_max = -fits[idx]
            best_routing = pop[idx].copy()
        
        # Selection (tournament)
        new_pop = np.zeros_like(pop)
        new_pop[0] = pop[idx]  # Elitism
        
        for i in range(1, pop_size):
            # Tournament selection
            contenders = np_random.choice(pop_size, 3, replace=False)
            winner = contenders[np.argmax(fits[contenders])]
            new_pop[i] = pop[winner].copy()
        
        # Crossover and mutation
        for i in range(1, pop_size // 2):
            p1_idx = i
            p2_idx = pop_size - 1 - i
            if np_random.random() < 0.7:  # Crossover probability
                point = np_random.integers(1, D)
                child1 = np.concatenate([pop[p1_idx][:point], pop[p2_idx][point:]])
                child2 = np.concatenate([pop[p2_idx][:point], pop[p1_idx][point:]])
            else:
                child1 = pop[p1_idx].copy()
                child2 = pop[p2_idx].copy()
            
            # Mutation
            for j in range(D):
                if np_random.random() < 0.1:
                    child1[j] = 1.0 - child1[j]
                if np_random.random() < 0.1:
                    child2[j] = 1.0 - child2[j]
            
            new_pop[p1_idx] = child1
            new_pop[p2_idx] = child2
        
        pop = new_pop
    
    if best_routing is None:
        best_routing = pop[0].copy()
        best_max = _compute_max_load(best_routing, cw_edges, ccw_edges, demands)
    
    return best_routing, best_max


def _milp_solve(cw_edges, ccw_edges, demands, deadline):
    """
    Solve the ring routing problem as a Mixed Integer Linear Program.
    Variables: x_d in {0,1} for each demand d (0=CW, 1=CCW), and t >= 0 (max load).
    Minimize t subject to: for each edge e, load_e <= t.
    load_e = sum_{d: e in ccw_edges[d]} w_d * x_d + sum_{d: e in cw_edges[d]} w_d * (1 - x_d)
           = sum_{d: e in ccw_edges[d]} w_d * x_d - sum_{d: e in cw_edges[d]} w_d * x_d + const_e
    where const_e = sum_{d: e in cw_edges[d]} w_d
    So: load_e - t <= 0 becomes:
      sum_{d: e in ccw_edges[d]} w_d * x_d - sum_{d: e in cw_edges[d]} w_d * x_d - t + const_e <= 0
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None, float('inf')
    
    D = len(demands)
    n = 10
    
    # Variables: [x_0, ..., x_{D-1}, t] -> total D+1 variables
    # x_d: binary (0 or 1)
    # t: continuous, >= 0
    
    # Objective: minimize t -> c = [0, 0, ..., 0, 1]
    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize t
    
    # Constraints: for each edge e (n constraints):
    # load_e <= t
    # load_e = const_e + sum_{d: e in ccw_edges[d]} w_d * x_d - sum_{d: e in cw_edges[d]} w_d * x_d
    # So: sum_{d: e in ccw_edges[d]} w_d * x_d - sum_{d: e in cw_edges[d]} w_d * x_d - t <= -const_e
    
    A = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    b_lb = np.full(n, -np.inf)
    
    for e in range(n):
        const_e = 0.0
        for d in range(D):
            s, t_node, w = demands[d]
            if e in cw_edges[d]:
                A[e, d] -= w
                const_e += w
            elif e in ccw_edges[d]:
                A[e, d] += w
        A[e, D] = -1.0  # coefficient for t
        b_ub[e] = -const_e  # A * x <= -const_e
    
    constraint = LinearConstraint(A, b_lb, b_ub)
    
    # Bounds: x_d in [0, 1] integer, t in [0, inf)
    integrality = np.ones(D + 1)
    integrality[D] = 0  # t is continuous
    
    bounds = Bounds(
        lb=np.zeros(D + 1),
        ub=np.concatenate([np.ones(D), [np.inf]])
    )
    
    result = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
    
    if result.success:
        routing = result.x[:D]
        max_load = result.x[D]
        # Verify
        actual_max = _compute_max_load(routing, cw_edges, ccw_edges, demands)
        return routing, actual_max
    else:
        return None, float('inf')


def _brute_force_small(n_demands=15):
    """
    For 15 demands, brute force all 2^15 = 32768 combinations.
    """
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    n = 10
    
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        cw_set = set(cw)
        ccw = [e for e in range(n) if e not in cw_set]
        ccw_edges.append(ccw)
    
    best_routing = None
    best_max = float('inf')
    
    # 2^15 = 32768, this is feasible
    for mask in range(1 << D):
        routing = np.zeros(D)
        for d in range(D):
            routing[d] = 1.0 if (mask >> d) & 1 else 0.0
        
        loads = np.zeros(n)
        for d in range(D):
            s, t, w = demands[d]
            if routing[d] >= 0.5:
                for e in ccw_edges[d]:
                    loads[e] += w
            else:
                for e in cw_edges[d]:
                    loads[e] += w
        
        max_load = np.max(loads)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    return best_routing, best_max


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using a combination of
    local search, simulated annealing, genetic algorithm, and
    smooth surrogate optimization.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    cw_edges, ccw_edges, demands = _build_routes()
    D = len(demands)
    n = 10
    
    best_routing = None
    best_max = float('inf')
    
    # Phase 0a: Try MILP for exact global optimum (fast for this small problem)
    if time.time() < deadline - 5:
        mr, mm = _milp_solve(cw_edges, ccw_edges, demands, deadline - 5)
        if mr is not None and mm < best_max:
            best_max = mm
            best_routing = mr.copy()
    
    # Phase 0b: Brute force all 2^15 = 32768 combinations (feasible!)
    # This guarantees the global optimum
    br, bm = _brute_force_small()
    if bm < best_max:
        best_max = bm
        best_routing = br.copy()
    
    # Phase 1: Multiple random restarts with local search (to verify/improve)
    np_random = np.random.default_rng(2024)
    
    for restart in range(30):
        if time.time() > deadline - 30:
            break
        
        # Random initial solution
        routing = np_random.random(D)
        
        # Local search
        lr, lm = _local_search(routing, cw_edges, ccw_edges, demands, deadline - 10, max_iter=50000)
        if lm < best_max - 1e-9:
            best_max = lm
            best_routing = lr.copy()
    
    # Phase 2: Pair local search from best
    if best_routing is not None and time.time() < deadline - 20:
        pr, pm = _pair_local_search(best_routing, cw_edges, ccw_edges, demands, deadline - 10, max_iter=20000)
        if pm < best_max - 1e-9:
            best_max = pm
            best_routing = pr.copy()
    
    # Phase 3: Simulated annealing from best
    if best_routing is not None and time.time() < deadline - 10:
        sr, sm = _simulated_annealing(best_routing, cw_edges, ccw_edges, demands, deadline - 5, T0=1.0, cooling=0.9998, max_iter=100000)
        if sm < best_max - 1e-9:
            best_max = sm
            best_routing = sr.copy()
    
    # Phase 4: Smooth gradient descent to find good continuous solution
    if time.time() < deadline - 5:
        # Try from best binary solution
        gr, gm = _smooth_gradient_descent(best_routing, cw_edges, ccw_edges, demands, deadline - 2, beta=5.0, lr=0.01, max_iter=2000)
        if gm < best_max - 1e-9:
            best_max = gm
            best_routing = gr.copy()
        
        # Try from random
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            rand_routing = np_random.random(D)
            gr, gm = _smooth_gradient_descent(rand_routing, cw_edges, ccw_edges, demands, deadline - 2, beta=5.0, lr=0.01, max_iter=1000)
            if gm < best_max - 1e-9:
                best_max = gm
                best_routing = gr.copy()
    
    # Phase 5: Genetic algorithm
    if time.time() < deadline - 5:
        ga_r, ga_m = _genetic_algorithm(cw_edges, ccw_edges, demands, deadline - 2, pop_size=40, max_gen=200)
        if ga_m < best_max - 1e-9:
            best_max = ga_m
            best_routing = ga_r.copy()
    
    # Final polish: local search + pair search on the best found
    if best_routing is not None:
        if time.time() < deadline - 2:
            fr, fm = _local_search(best_routing, cw_edges, ccw_edges, demands, deadline - 1, max_iter=100000)
            if fm < best_max - 1e-9:
                best_max = fm
                best_routing = fr.copy()
        
        if time.time() < deadline - 1:
            pr, pm = _pair_local_search(best_routing, cw_edges, ccw_edges, demands, deadline, max_iter=10000)
            if pm < best_max - 1e-9:
                best_max = pm
                best_routing = pr.copy()
    
    # Ensure we have a valid routing
    if best_routing is None:
        best_routing = np.zeros(D)
    
    return best_routing


# EVOLVE-BLOCK-END
