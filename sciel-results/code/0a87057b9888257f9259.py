import time
import numpy as np
from autoresearch_suite.science_tcs_expansion import check_lift

def _build_signed_adjacency(graph, signs):
    """
    Builds the signed adjacency matrix for the base graph.
    graph: dict with 'n' and 'edges'
    signs: array of -1 or +1 for each edge
    """
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        val = float(signs[i])
        A[u, v] = val
        A[v, u] = val
    return A

def _spectral_radius(A):
    """
    Computes the spectral radius of a symmetric matrix A.
    """
    try:
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    except:
        return float('inf')

def _evaluate(graph, signs):
    """
    Evaluates the sign vector. Returns spectral radius if valid, inf if invalid.
    Uses check_lift to ensure validity and get the radius.
    """
    try:
        radius, details = check_lift(graph, signs)
        return radius
    except Exception:
        return float('inf')

def _local_search(graph, initial_signs, time_limit):
    """
    Performs local search starting from initial_signs.
    Returns the best signs and radius found.
    """
    start_time = time.time()
    current_signs = initial_signs.copy()
    current_radius = _evaluate(graph, current_signs)
    
    if np.isinf(current_radius):
        return initial_signs, float('inf')
    
    best_signs = current_signs.copy()
    best_radius = current_radius
    
    n_edges = len(graph['edges'])
    
    # Simple hill climbing with restarts within the time limit if stuck
    # We will just do a single pass of hill climbing but with multiple random restarts in the main loop
    # Or we can do simulated annealing here.
    # Let's stick to robust hill climbing with a "restart" logic if we want.
    # Actually, multiple initializations in the caller is better.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Iterate through edges multiple times if time permits
        # Randomize edge order to avoid bias
        edge_order = np.random.permutation(n_edges)
        
        for i in edge_order:
            if time.time() - start_time > time_limit:
                break
                
            new_signs = current_signs.copy()
            new_signs[i] *= -1
            new_radius = _evaluate(graph, new_signs)
            
            if new_radius < current_radius - 1e-9:
                current_signs = new_signs
                current_radius = new_radius
                improved = True
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = current_signs.copy()
        
        if not improved:
            break
            
    return best_signs, best_radius

def sign_edges(graph, budget):
    start_time = time.time()
    total_time_limit = 55.0 # Leave safety margin for 60s limit
    
    # We call this function 4 times, once for each m.
    # We don't know which m it is inside, but we can infer from graph['n'].
    # We should allocate time based on remaining time.
    # However, the function is called independently for each graph.
    # We should assume we have a fair share of time, or just use a fixed reasonable time.
    # Since the evaluator calls it 4 times, and the total limit is 90s, 
    # we have roughly 20-22s per call if we want to be safe.
    # Let's use 15s per call to be very safe.
    time_per_call = 15.0
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    best_signs = None
    best_radius = float('inf')
    
    # Generate initial candidates
    # 1. All +1
    # 2. All -1
    # 3. Alternating on cycles
    # 4. Random
    
    initial_candidates = []
    
    # All +1
    initial_candidates.append(np.ones(n_edges, dtype=int))
    # All -1
    initial_candidates.append(-np.ones(n_edges, dtype=int))
    
    # Alternating: 
    # Edges are typically ordered: cycle1 edges, cycle2 edges, matching edges?
    # The problem statement says: "two m-cycles, on vertices 0..m-1 and m..2m-1, joined by the matching (i, i + m)."
    # We don't know the exact edge order. Let's just use random for the rest.
    
    # Add some random initializations
    rng = np.random.default_rng(3)
    for _ in range(20):
        initial_candidates.append(rng.choice([-1, 1], size=n_edges))
    
    # Run local search on each candidate
    # Allocate time roughly equally among candidates
    time_per_candidate = time_per_call / len(initial_candidates)
    
    for i, init in enumerate(initial_candidates):
        if time.time() - start_time > time_per_call:
            break
            
        remaining = time_per_call - (time.time() - start_time)
        # Reserve a little time for the last few or just use the slice
        t_limit = min(time_per_candidate, remaining * 0.9)
        
        if t_limit < 0.1:
            continue
            
        signs, radius = _local_search(graph, init, t_limit)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
        # Early stop if we found a very good solution?
        # Theoretically, for these small graphs, the optimal might be low.
        # If best_radius is very small, we might stop.
        
    if best_signs is None:
        # Fallback to the first valid one found or just return all +1 if valid?
        # We need to ensure validity.
        # If no valid sign was found (unlikely), we return all +1 and hope it's valid?
        # Or better, find any valid sign.
        # Random sampling might find one.
        for _ in range(100):
            s = rng.choice([-1, 1], size=n_edges)
            r = _evaluate(graph, s)
            if not np.isinf(r):
                return s
        # If still none, return all +1 (might be invalid, but better than exception?)
        # The task says "or the whole result is invalid". So we must return valid.
        # Let's try a simple constructive valid sign.
        # All +1 is a valid lift? 
        # If all +1, the lift is two disjoint copies of the base graph. 
        # The base graph (prism) is connected. So the lift is TWO connected components.
        # The task requires the lift to be CONNECTED.
        # So all +1 is INVALID for connected base graph?
        # "The lift must be connected".
        # If all edges are +1, the lift has vertices (v,0) and (v,1).
        # Edge (u,v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1).
        # So (v,0) is only connected to other (w,0). (v,1) only to (w,1).
        # The lift is disconnected (two components).
        # So all +1 is invalid.
        # All -1?
        # Edge (u,v) sign -1 connects (u,0)-(v,1) and (u,1)-(v,0).
        # This effectively "twists" the lift.
        # For a bipartite graph, all -1 might make it connected?
        # Prism graphs are not bipartite if m is odd (contains odd cycles).
        # If m is even, prism is bipartite? No, m-cycle is bipartite if m even.
        # Prism graph C_m x K_2.
        # If m is odd, C_m has odd cycles.
        # Let's just rely on the local search finding a valid one.
        # Random signs are likely to produce a connected lift.
        
    return best_signs
