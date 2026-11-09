# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem: 10-node ring, 15 demands,
    each routed CW or CCW to minimize max edge load.
    Uses MILP formulation with scipy.optimize.milp for provably optimal solution.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    # Precompute edge usage matrices
    A_cw = np.zeros((D, n), dtype=float)
    A_ccw = np.zeros((D, n), dtype=float)
    weights = np.zeros(D, dtype=float)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        e = s
        while e != t:
            A_cw[d, e % n] = 1
            e = (e + 1) % n
        A_ccw[d] = 1.0 - A_cw[d]

    # W_cw[d, e] = weight[d] if demand d uses edge e when routed CW, else 0
    # W_ccw[d, e] = weight[d] if demand d uses edge e when routed CCW, else 0
    W_cw = A_cw * weights[:, None]
    W_ccw = A_ccw * weights[:, None]

    start_time = time.time()
    deadline = start_time + 590.0

    def compute_loads(routing: np.ndarray) -> np.ndarray:
        """routing: 0/1 array per demand (0=CW, 1=CCW). Returns edge loads."""
        diff = W_ccw - W_cw
        return W_cw.sum(axis=0) + diff.T @ routing

    def max_load(routing: np.ndarray) -> float:
        return float(np.max(compute_loads(routing)))

    # MILP formulation:
    # Variables: x_0..x_{D-1} (binary, 0=CW, 1=CCW), t (continuous, max load)
    # Minimize: t
    # Constraints: for each edge e: sum_d (W_cw[d,e]*(1-x_d) + W_ccw[d,e]*x_d) <= t
    #   => sum_d (W_ccw[d,e]-W_cw[d,e])*x_d - t <= -sum_d W_cw[d,e]

    # c: objective coefficients (minimize t)
    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize t

    # A_ub: constraint matrix for inequalities A_ub @ vars <= b_ub
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    for e in range(n):
        A_ub[e, :D] = W_ccw[:, e] - W_cw[:, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -W_cw[:, e].sum()

    # Bounds: x_d in [0,1] for d=0..D-1, t >= 0
    bounds = Bounds(lb=np.concatenate([np.zeros(D), [0.0]]),
                    ub=np.concatenate([np.ones(D), [np.inf]]))

    # Integrality: x_d are binary (integrality=1), t is continuous (integrality=0)
    integrality = np.concatenate([np.ones(D, dtype=int), [0]])

    # Solve MILP
    best_routing = None
    best_milp_load = np.inf
    try:
        res = milp(c=c,
                   constraints=LinearConstraint(A_ub, ub=b_ub),
                   integrality=integrality,
                   bounds=bounds,
                   options={"time_limit": 580.0, "mip_rel_gap": 0.0, "mip_absgap": 1e-12})
        if res.status in [0, 2]:  # 0=optimal, 2=feasible (may be suboptimal)
            best_routing = np.where(res.x[:D] >= 0.5, 1.0, 0.0)
            best_milp_load = max_load(best_routing)
    except Exception:
        pass

    # Fallback: local search if MILP didn't find a solution
    if best_routing is None:
        routing = np.zeros(D, dtype=float)
        improved = True
        while improved:
            improved = False
            cur = max_load(routing)
            for d in range(D):
                routing[d] = 1.0 - routing[d]
                new = max_load(routing)
                if new < cur:
                    cur = new
                    improved = True
                else:
                    routing[d] = 1.0 - routing[d]
        best_routing = routing.copy()
        best_milp_load = max_load(best_routing)
    else:
        # Round to nearest 0 or 1 for safety
        best_routing = np.where(best_routing >= 0.5, 1.0, 0.0)

    # Local search polish from best found
    if time.time() < deadline - 5.0:
        routing = best_routing.copy()
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            cur = max_load(routing)
            for d in range(D):
                routing[d] = 1.0 - routing[d]
                new = max_load(routing)
                if new < cur - 1e-12:
                    cur = new
                    improved = True
                else:
                    routing[d] = 1.0 - routing[d]
        if max_load(routing) < best_milp_load:
            best_routing = routing.copy()
            best_milp_load = max_load(best_routing)

    # Try exhaustive search if time permits (2^15 = 32768 is small)
    if time.time() < deadline - 30.0:
        # Use vectorized exhaustive search
        # Generate all 2^15 combinations
        best_exhaustive = best_routing.copy()
        best_exhaustive_load = best_milp_load
        
        # For speed, use numpy to generate all combinations and compute loads
        # 2^15 = 32768 combinations, each with 10 edge loads
        # We can vectorize this
        
        # Create all binary vectors of length 15
        # This might use too much memory if done naively, but 32768 * 15 is small
        # Let's do it in batches or use a smarter approach
        
        # Actually, let's just do a simple loop since 32768 is small
        # But we need to be careful about time
        
        # Alternative: use the fact that we can precompute the contribution of each demand
        # and use Gray code or similar for efficient traversal
        
        # For now, let's try a direct approach with early termination
        # We'll iterate through all combinations but break early if we find the theoretical minimum
        
        # The theoretical minimum is at least max(sum of weights / 10, max single demand weight)
        # Sum of weights = 3+2+4+1+5+2+3+2+4+1+3+2+5+1+2 = 40
        # So min possible max load is at least 4
        # Max single demand weight is 5
        
        # Let's just do exhaustive search since 32768 is small
        # We'll use a recursive approach or itertools
        
        from itertools import product
        
        # This will take a bit but should be fast enough
        # 32768 * 10 operations = ~327k operations, very fast
        
        # But let's check time first
        if time.time() < deadline - 10.0:
            best_found = best_milp_load
            best_comb = best_routing.copy()
            
            # Use numpy to generate all combinations efficiently
            # 2^15 = 32768
            # We can use np.arange(2**15) and convert to binary
            
            for combo in range(2 ** D):
                if time.time() > deadline - 5.0:
                    break
                
                # Convert to binary vector
                routing_vec = np.array([(combo >> d) & 1 for d in range(D)], dtype=float)
                
                loads = compute_loads(routing_vec)
                ml = np.max(loads)
                
                if ml < best_found - 1e-12:
                    best_found = ml
                    best_comb = routing_vec.copy()
                    if best_found <= 4.0 + 1e-9:  # Theoretical lower bound
                        break
            
            if best_found < best_milp_load:
                best_routing = best_comb
                best_milp_load = best_found

    return best_routing

# EVOLVE-BLOCK-END
