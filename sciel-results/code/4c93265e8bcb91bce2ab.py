"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Identify edge types
    # Top cycle: edges 0 to m-1 (vertex i to i+1 mod m)
    # Bottom cycle: edges m to 2m-1 (vertex m+i to m+i+1 mod m)
    # Matching: edges 2m to 3m-1 (vertex i to m+i)
    
    top_edges = list(range(m))
    bottom_edges = list(range(m, 2*m))
    match_edges = list(range(2*m, 3*m))
    
    def build_signed_adjacency(signs):
        """Build the signed adjacency matrix of the base graph."""
        A = np.zeros((n, n), dtype=float)
        for idx, (u, v) in enumerate(edges):
            s = float(signs[idx])
            A[u, v] += s
            A[v, u] += s
        return A
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = build_signed_adjacency(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the 2-lift is connected and 3-regular."""
        try:
            radius, details = check_lift(graph, signs)
            return True
        except Exception:
            return False
    
    def evaluate(signs):
        """Evaluate a sign assignment, returning radius if valid, else None."""
        if not is_valid_lift(signs):
            return None
        return compute_radius(signs)
    
    best_signs = None
    best_radius = float('inf')
    
    def update_best(signs, radius):
        nonlocal best_signs, best_radius
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # For the prism graph, a spanning tree consists of:
    # - Top path: m-1 edges (0-1, 1-2, ..., m-2-m-1)
    # - Bottom path: m-1 edges (m-(m+1), ..., 2m-2-(2m-1))
    # - One matching edge: 0-m
    # Total: 2m-1 edges
    # Non-tree edges: 
    # - Top cycle closing edge: (m-1, 0) -> index m-1
    # - Bottom cycle closing edge: (2m-1, m) -> index 2m-1
    # - Remaining matching edges: (1, m+1), ..., (m-1, 2m-1) -> indices 2m+1 to 3m-1
    # Total non-tree edges: 2 + (m-1) = m+1
    
    # Gauge fix: set all spanning tree edges to +1
    # This determines the signs of all edges up to gauge
    # The free variables are the m+1 non-tree edges
    
    def enumerate_gauge_fixed():
        """Exhaustively enumerate all 2^(m+1) sign assignments for non-tree edges."""
        nonlocal best_signs, best_radius
        
        # Indices of non-tree edges
        non_tree_indices = [m-1, 2*m-1] + list(range(2*m+1, 3*m))
        
        # For each assignment of signs to non-tree edges
        for mask in range(2 ** (m + 1)):
            if time.time() > deadline - 3:
                return
            
            # Build signs array with all +1 initially
            signs = np.ones(num_edges, dtype=int)
            
            # Set non-tree edge signs based on mask
            for i, idx in enumerate(non_tree_indices):
                if (mask >> i) & 1:
                    signs[idx] = -1
            
            # Evaluate
            radius = evaluate(signs)
            if radius is not None:
                update_best(signs, radius)
    
    # Start with exhaustive enumeration (fast for m <= 9)
    enumerate_gauge_fixed()
    
    # If we have a good solution, try to improve with local search
    if best_signs is not None:
        # Local search: try flipping individual non-tree edges
        non_tree_indices = [m-1, 2*m-1] + list(range(2*m+1, 3*m))
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in non_tree_indices:
                if time.time() > deadline - 5:
                    break
                
                trial = best_signs.copy()
                trial[idx] *= -1
                
                radius = evaluate(trial)
                if radius is not None and radius < best_radius - 1e-12:
                    best_signs = trial
                    best_radius = radius
                    improved = True
        
        # Try flipping pairs of non-tree edges
        for i in range(len(non_tree_indices)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(non_tree_indices)):
                if time.time() > deadline - 5:
                    break
                
                trial = best_signs.copy()
                trial[non_tree_indices[i]] *= -1
                trial[non_tree_indices[j]] *= -1
                
                radius = evaluate(trial)
                if radius is not None and radius < best_radius - 1e-12:
                    best_signs = trial
                    best_radius = radius
    
    # If we still have time, try random restarts
    rng = np.random.default_rng(42)
    remaining_time = deadline - time.time()
    
    if remaining_time > 5:
        # Random restarts with local search
        num_restarts = min(100, int(remaining_time / 0.5))
        for _ in range(num_restarts):
            if time.time() > deadline - 3:
                break
            
            signs = rng.choice([-1, 1], size=num_edges)
            radius = evaluate(signs)
            if radius is not None:
                update_best(signs, radius)
            
            # Quick local search from this point
            if best_signs is not None and time.time() < deadline - 3:
                non_tree_indices = [m-1, 2*m-1] + list(range(2*m+1, 3*m))
                for idx in non_tree_indices:
                    if time.time() > deadline - 3:
                        break
                    trial = best_signs.copy()
                    trial[idx] *= -1
                    radius = evaluate(trial)
                    if radius is not None and radius < best_radius - 1e-12:
                        best_signs = trial
                        best_radius = radius
    
    if best_signs is None:
        # Fallback: all +1
        return np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
