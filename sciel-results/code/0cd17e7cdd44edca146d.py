"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best_signs = None
    best_radius = float('inf')
    
    # Pre-compute edge types for structured generation
    # Edges are: top cycle (m), bottom cycle (m), matching (m)
    top_edges = list(range(m))
    bottom_edges = list(range(m, 2*m))
    match_edges = list(range(2*m, 3*m))
    
    def evaluate(signs):
        nonlocal best_signs, best_radius
        try:
            radius, details = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
            return radius
        except Exception:
            return None
    
    def make_structured_signs(pattern_top, pattern_bottom, pattern_match):
        signs = np.ones(num_edges, dtype=int)
        for i in range(m):
            signs[top_edges[i]] = pattern_top[i % len(pattern_top)]
        for i in range(m):
            signs[bottom_edges[i]] = pattern_bottom[i % len(pattern_bottom)]
        for i in range(m):
            signs[match_edges[i]] = pattern_match[i % len(pattern_match)]
        return signs
    
    def local_search(current, max_iters=500):
        nonlocal best_signs, best_radius
        cur = current.copy()
        cur_radius = evaluate(cur)
        if cur_radius is None:
            return
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Try single flips
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 2:
                    break
                trial = cur.copy()
                trial[i] = -trial[i]
                trial_radius = evaluate(trial)
                if trial_radius is not None and trial_radius < cur_radius - 1e-12:
                    cur = trial
                    cur_radius = trial_radius
                    improved = True
                    break
            
            # Try double flips
            if not improved:
                for i in range(num_edges):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i+1, num_edges):
                        trial = cur.copy()
                        trial[i] = -trial[i]
                        trial[j] = -trial[j]
                        trial_radius = evaluate(trial)
                        if trial_radius is not None and trial_radius < cur_radius - 1e-12:
                            cur = trial
                            cur_radius = trial_radius
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        # Update global best
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = cur.copy()
    
    def local_search_3flip(current, max_iters=100):
        nonlocal best_signs, best_radius
        cur = current.copy()
        cur_radius = evaluate(cur)
        if cur_radius is None:
            return
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 2:
                    break
                for j in range(i+1, num_edges):
                    if time.time() > deadline - 2:
                        break
                    for k in range(j+1, num_edges):
                        trial = cur.copy()
                        trial[i] = -trial[i]
                        trial[j] = -trial[j]
                        trial[k] = -trial[k]
                        trial_radius = evaluate(trial)
                        if trial_radius is not None and trial_radius < cur_radius - 1e-12:
                            cur = trial
                            cur_radius = trial_radius
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = cur.copy()
    
    # Generate structured initial guesses
    # Key insight: alternating patterns on cycles can create bipartite-like structure
    # which tends to reduce spectral radius
    patterns = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
        [1, 1, 1, -1],
        [1, -1, 1, 1],
        [1, -1, -1, -1],
        [-1, 1, 1, 1],
    ]
    
    # Try all combinations of patterns
    for pt in patterns:
        for pb in patterns:
            for pm in patterns:
                if time.time() > deadline - 10:
                    break
                signs = make_structured_signs(pt, pb, pm)
                evaluate(signs)
    
    # Random sampling
    rng = np.random.default_rng(42)
    
    num_random = 3000 if m <= 5 else 1000
    for _ in range(num_random):
        if time.time() > deadline - 15:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)
    
    # Local search from best
    if best_signs is not None:
        local_search(best_signs, max_iters=300)
    
    # More random restarts with local search
    num_restarts = 50 if m <= 5 else 20
    for _ in range(num_restarts):
        if time.time() > deadline - 5:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)
        if best_signs is not None:
            local_search(best_signs, max_iters=100)
    
    # Final polish: try 3-flip moves from the best solution
    if best_signs is not None and time.time() < deadline - 3:
        local_search_3flip(best_signs, max_iters=50)
    
    if best_signs is None:
        return np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
