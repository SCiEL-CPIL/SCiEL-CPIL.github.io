"""Guided local search penalty for TSP; direct GLS with edge-reversal targeting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0
_start_time = None


def _two_opt(D, tour, max_passes=10, tol=-1e-10):
    """2-opt on distance matrix D starting from tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return tour
    tour = list(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < tol:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _closed_len(D, tour):
    """Compute the length of the closed tour using distance matrix D."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]
    return total


def _compute_edge_improvements(D, tour, n):
    """
    For each edge in the tour, compute the best possible 2-opt improvement
    that would remove that edge. Returns dict mapping (u,v) -> best_improvement.
    """
    # For each edge (tour[i], tour[i+1]), find the best 2-opt move that removes it
    # A 2-opt move removes edges (a,b) and (c,d) and adds (a,c) and (b,d)
    # The improvement is D[a,b] + D[c,d] - D[a,c] - D[b,d]
    # We want the max improvement for each edge.
    
    # Build a list of edges in the tour
    tour_edges = []
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        tour_edges.append((u, v))
    
    # For each edge, find the best 2-opt improvement involving that edge
    # This is O(n^3) but n=100, so 100^3 = 1e6, which is fine
    edge_improvements = {}
    
    for i in range(n):
        u, v = tour_edges[i]
        # Edge (u,v) is between tour[i-1] and tour[i] (or tour[n-1] and tour[0])
        # Let's say a = tour[i-1], b = tour[i]
        a = tour[(i - 1) % n]
        b = tour[i]
        
        best_imp = 0.0
        # Try all other edges (c,d) where c = tour[j], d = tour[j+1]
        # The 2-opt move would remove (a,b) and (c,d), add (a,c) and (b,d)
        # or remove (a,b) and (c,d), add (a,d) and (b,c)
        # We need to ensure the edges are not adjacent
        
        for j in range(n):
            if j == i or j == (i - 1) % n:
                continue
            c = tour[j]
            d = tour[(j + 1) % n]
            
            # Option 1: add (a,c) and (b,d)
            imp1 = D[a, b] + D[c, d] - D[a, c] - D[b, d]
            if imp1 > best_imp:
                best_imp = imp1
            
            # Option 2: add (a,d) and (b,c)
            imp2 = D[a, b] + D[c, d] - D[a, d] - D[b, c]
            if imp2 > best_imp:
                best_imp = imp2
    
    edge_improvements[(u, v)] = best_imp
    
    return edge_improvements


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = edge_distance.shape[0]
    tour = list(local_opt_tour)
    
    # Check if we're close to the deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Return a simple penalty to be safe
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
    
    # Get the edges in the current local optimum tour
    tour_edges = set()
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Compute edge improvements for targeted penalties
    edge_improvements = _compute_edge_improvements(edge_distance, tour, n)
    
    # Get usage counts for tour edges
    tour_usage = []
    for (u, v) in tour_edges:
        tour_usage.append(edge_n_used[u, v])
    
    if tour_usage:
        max_tour_usage = max(tour_usage)
        min_tour_usage = min(tour_usage)
    else:
        max_tour_usage = 1.0
        min_tour_usage = 0.0
    
    if max_tour_usage < 1e-10:
        max_tour_usage = 1.0
    
    # Try multiple penalty strategies
    best_D_pen = None
    best_len = float('inf')
    best_tour = tour
    
    # Strategy parameters: (base_penalty, exponent, scope, use_improvement)
    # scope: 0 = penalize only current tour edges, 1 = penalize all edges by usage
    # use_improvement: if True, scale penalty by edge improvement
    strategies = [
        (0.1, 1.0, 0, False),
        (0.2, 1.0, 0, False),
        (0.3, 1.0, 0, False),
        (0.1, 2.0, 0, False),
        (0.2, 2.0, 0, False),
        (0.05, 1.5, 0, False),
        (0.15, 1.5, 0, False),
        (0.05, 1.0, 1, False),
        (0.1, 1.0, 1, False),
        (0.2, 1.0, 1, False),
        # Edge-reversal targeted strategies
        (0.1, 1.0, 0, True),
        (0.2, 1.0, 0, True),
        (0.3, 1.0, 0, True),
        (0.1, 2.0, 0, True),
        (0.2, 2.0, 0, True),
    ]
    
    for base_pen, exponent, scope, use_improvement in strategies:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        D_pen = edge_distance.copy()
        
        if scope == 0:
            # Penalize only edges in the current tour
            for (u, v) in tour_edges:
                usage = edge_n_used[u, v]
                if usage > 0:
                    normalized_usage = usage / max_tour_usage
                    smooth_penalty = base_pen * (normalized_usage ** exponent + 0.02 * normalized_usage)
                    
                    if use_improvement:
                        # Scale penalty by the best improvement available for this edge
                        imp = edge_improvements.get((u, v), 0.0)
                        if imp > 0:
                            # Use a smooth scaling: penalty proportional to improvement
                            # Normalize improvement to [0,1] range
                            max_imp = max(edge_improvements.values()) if edge_improvements else 1.0
                            if max_imp < 1e-10:
                                max_imp = 1.0
                            imp_scale = imp / max_imp
                            smooth_penalty *= (1.0 + imp_scale)
                    
                    D_pen[u, v] += smooth_penalty
                    D_pen[v, u] += smooth_penalty
        else:
            # Penalize all edges based on their usage
            max_used = edge_n_used.max()
            if max_used < 1e-10:
                max_used = 1.0
            usage_matrix = edge_n_used / max_used
            penalty_matrix = base_pen * (usage_matrix ** exponent + 0.02 * usage_matrix)
            D_pen = D_pen + penalty_matrix
        
        # Ensure symmetry
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        
        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        
        # Final polish: run 2-opt on true distances with tighter tolerance
        if time.time() - _start_time < _DEADLINE - 5:
            tour_polished = _two_opt(edge_distance, tour_pen, max_passes=10, tol=-1e-12)
            true_len = _closed_len(edge_distance, tour_polished)
            if true_len < best_len:
                best_len = true_len
                best_tour = tour_polished
                best_D_pen = D_pen
        else:
            true_len = _closed_len(edge_distance, tour_pen)
            if true_len < best_len:
                best_len = true_len
                best_tour = tour_pen
                best_D_pen = D_pen
    
    # Additional refinement: try a few more penalty variants if we have time
    if best_D_pen is not None and time.time() - _start_time < _DEADLINE - 20:
        for mult in [0.8, 1.2]:
            D_alt = best_D_pen * mult
            D_alt = (D_alt + D_alt.T) / 2.0
            np.fill_diagonal(D_alt, 0.0)
            
            tour_alt = _two_opt(D_alt, tour, max_passes=10)
            tour_alt = _two_opt(edge_distance, tour_alt, max_passes=10, tol=-1e-12)
            alt_len = _closed_len(edge_distance, tour_alt)
            
            if alt_len < best_len:
                best_len = alt_len
                best_tour = tour_alt
                best_D_pen = D_alt
            
            if time.time() - _start_time > _DEADLINE - 10:
                break
    
    if best_D_pen is not None:
        return best_D_pen
    else:
        # Fallback: return edge_distance with mild penalty
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
# EVOLVE-BLOCK-END
