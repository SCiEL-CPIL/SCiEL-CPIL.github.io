# EVOLVE-BLOCK-START
import numpy as np
import time


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
        
        all_edges = set(range(n))
        cw_set = set(cw)
        ccw = [e for e in range(n) if e not in cw_set]
        ccw_edges.append(ccw)
    
    return cw_edges, ccw_edges, demands


def _build_matrices(cw_edges, ccw_edges, demands, n=10):
    """Build incidence matrices for fast load computation."""
    D = len(demands)
    cw_mat = np.zeros((D, n), dtype=np.float64)
    ccw_mat = np.zeros((D, n), dtype=np.float64)
    weights = np.array([w for _, _, w in demands], dtype=np.float64)
    
    for d in range(D):
        for e in cw_edges[d]:
            cw_mat[d, e] = 1.0
        for e in ccw_edges[d]:
            ccw_mat[d, e] = 1.0
    
    return cw_mat, ccw_mat, weights


def _compute_loads_fast(routing, cw_mat, ccw_mat, weights):
    """Compute edge loads using matrix operations. routing[d]=1 means CCW."""
    ccw_mask = (routing >= 0.5).astype(np.float64)
    cw_mask = 1.0 - ccw_mask
    loads = np.sum(weights[:, None] * (ccw_mask[:, None] * ccw_mat + cw_mask[:, None] * cw_mat), axis=0)
    return loads


def _local_search_single_flip(routing, cw_mat, ccw_mat, weights, deadline, max_rounds=100):
    """Iterative single-flip local search."""
    D = len(routing)
    best_routing = routing.copy()
    best_loads = _compute_loads_fast(best_routing, cw_mat, ccw_mat, weights)
    best_max_load = np.max(best_loads)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for d in range(D):
            if time.time() > deadline:
                break
            trial_routing = best_routing.copy()
            trial_routing[d] = 1.0 - trial_routing[d]
            trial_loads = _compute_loads_fast(trial_routing, cw_mat, ccw_mat, weights)
            trial_max_load = np.max(trial_loads)
            
            if trial_max_load < best_max_load - 1e-12:
                best_routing = trial_routing
                best_loads = trial_loads
                best_max_load = trial_max_load
                improved = True
        if not improved:
            break
    
    return best_routing, best_max_load


def _local_search_pair_flip(routing, cw_mat, ccw_mat, weights, deadline, max_rounds=50):
    """Iterative pair-flip local search."""
    D = len(routing)
    best_routing = routing.copy()
    best_loads = _compute_loads_fast(best_routing, cw_mat, ccw_mat, weights)
    best_max_load = np.max(best_loads)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for d1 in range(D):
            if time.time() > deadline:
                break
            for d2 in range(d1 + 1, D):
                trial_routing = best_routing.copy()
                trial_routing[d1] = 1.0 - trial_routing[d1]
                trial_routing[d2] = 1.0 - trial_routing[d2]
                trial_loads = _compute_loads_fast(trial_routing, cw_mat, ccw_mat, weights)
                trial_max_load = np.max(trial_loads)
                
                if trial_max_load < best_max_load - 1e-12:
                    best_routing = trial_routing
                    best_loads = trial_loads
                    best_max_load = trial_max_load
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_routing, best_max_load


def _local_search_triple_flip(routing, cw_mat, ccw_mat, weights, deadline, max_rounds=20):
    """Iterative triple-flip local search for escaping local optima."""
    D = len(routing)
    best_routing = routing.copy()
    best_loads = _compute_loads_fast(best_routing, cw_mat, ccw_mat, weights)
    best_max_load = np.max(best_loads)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for d1 in range(D):
            if time.time() > deadline:
                break
            for d2 in range(d1 + 1, D):
                if time.time() > deadline:
                    break
                for d3 in range(d2 + 1, D):
                    trial_routing = best_routing.copy()
                    trial_routing[d1] = 1.0 - trial_routing[d1]
                    trial_routing[d2] = 1.0 - trial_routing[d2]
                    trial_routing[d3] = 1.0 - trial_routing[d3]
                    trial_loads = _compute_loads_fast(trial_routing, cw_mat, ccw_mat, weights)
                    trial_max_load = np.max(trial_loads)
                    
                    if trial_max_load < best_max_load - 1e-12:
                        best_routing = trial_routing
                        best_loads = trial_loads
                        best_max_load = trial_max_load
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_routing, best_max_load


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using multi-restart local search.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill
    
    cw_edges, ccw_edges, demands = _build_routes()
    D = len(demands)
    n = 10
    
    cw_mat, ccw_mat, weights = _build_matrices(cw_edges, ccw_edges, demands, n)
    
    # Generate multiple initial solutions
    initial_solutions = []
    
    # All clockwise
    initial_solutions.append(np.zeros(D))
    # All counter-clockwise
    initial_solutions.append(np.ones(D))
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        initial_solutions.append(rng.random(D))
    
    best_overall_routing = None
    best_overall_max_load = float('inf')
    
    for init_sol in initial_solutions:
        if time.time() > deadline:
            break
        
        # Phase 1: Single flip local search
        routing, max_load = _local_search_single_flip(init_sol, cw_mat, ccw_mat, weights, deadline)
        
        # Phase 2: Pair flip local search
        routing, max_load = _local_search_pair_flip(routing, cw_mat, ccw_mat, weights, deadline)
        
        # Phase 3: Triple flip local search (escape local optima)
        routing, max_load = _local_search_triple_flip(routing, cw_mat, ccw_mat, weights, deadline)
        
        # Phase 4: Final polish - single flip again
        routing, max_load = _local_search_single_flip(routing, cw_mat, ccw_mat, weights, deadline)
        
        if max_load < best_overall_max_load:
            best_overall_max_load = max_load
            best_overall_routing = routing.copy()
    
    # Final verification and cleanup
    if best_overall_routing is None:
        best_overall_routing = np.zeros(D)
    
    # Ensure clean output (exactly 0 or 1)
    best_overall_routing = np.where(best_overall_routing >= 0.5, 1.0, 0.0)
    
    return best_overall_routing


# EVOLVE-BLOCK-END
