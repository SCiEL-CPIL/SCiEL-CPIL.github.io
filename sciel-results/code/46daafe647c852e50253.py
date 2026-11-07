"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _build_signed_adjacency(graph, signs):
    """Build the signed adjacency matrix from graph and signs."""
    n = graph['n']
    A = np.zeros((n, n))
    for edge, s in zip(graph['edges'], signs):
        u, v = edge
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(A):
    """Compute spectral radius of symmetric matrix."""
    eigvals = np.linalg.eigvalsh(A)
    return np.max(np.abs(eigvals))

def _is_valid_lift(graph, signs):
    """Check if the lift is connected and 3-regular using check_lift."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, radius
    except Exception:
        return False, None

def _exhaustive_search(graph, m):
    """Exhaustive search over all 2^(3m) sign assignments."""
    n_edges = 3 * m
    best_radius = float('inf')
    best_signs = None
    
    # For m=4: 2^12 = 4096, m=5: 2^15 = 32768
    # For m=7: 2^21 = 2M (too slow), m=9: 2^27 (too slow)
    if m > 6:
        return None, None
    
    edges = graph['edges']
    n = graph['n']
    
    for mask in range(1 << n_edges):
        signs = np.zeros(n_edges, dtype=np.int8)
        for i in range(n_edges):
            signs[i] = 1 if (mask >> i) & 1 else -1
        
        valid, radius = _is_valid_lift(graph, signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    return best_signs, best_radius

def _simulated_annealing(graph, m, max_iters=500000, start_temp=1.0, end_temp=0.01, 
                          cooling_rate=0.99995, max_time=55.0):
    """Simulated annealing to find good sign assignments."""
    n_edges = 3 * m
    edges = graph['edges']
    n = graph['n']
    
    # Start with random valid assignment
    best_signs = None
    best_radius = float('inf')
    
    # Try several random starts
    rng = np.random.default_rng(42)
    
    for start_attempt in range(10):
        signs = rng.choice([-1, 1], size=n_edges).astype(np.int8)
        
        valid, radius = _is_valid_lift(graph, signs)
        if not valid:
            continue
        
        current_signs = signs.copy()
        current_radius = radius
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()
        
        temp = start_temp
        accepted = 0
        total = 0
        
        for it in range(max_iters):
            if time.time() - _sa_start_time > max_time:
                break
            
            # Pick a random edge to flip
            edge_idx = rng.integers(0, n_edges)
            
            # Flip the sign
            old_sign = current_signs[edge_idx]
            current_signs[edge_idx] = -old_sign
            
            new_valid, new_radius = _is_valid_lift(graph, current_signs)
            total += 1
            
            if new_valid:
                delta = new_radius - current_radius
                if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                    # Accept
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = current_signs.copy()
                    current_radius = new_radius
                    accepted += 1
                else:
                    # Reject
                    current_signs[edge_idx] = old_sign
            else:
                # Invalid - reject
                current_signs[edge_idx] = old_sign
            
            temp *= cooling_rate
        
        if time.time() - _sa_start_time > max_time:
            break
    
    return best_signs, best_radius

def _local_search(graph, signs, max_iters=100000, max_time=5.0):
    """Local search to refine a sign assignment."""
    n_edges = 3 * m
    edges = graph['edges']
    n = graph['n']
    
    current_signs = signs.copy()
    current_radius = None
    
    valid, current_radius = _is_valid_lift(graph, current_signs)
    if not valid:
        return signs, float('inf')
    
    best_radius = current_radius
    best_signs = current_signs.copy()
    
    start_time = time.time()
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for edge_idx in rng.permutation(n_edges):
            old_sign = current_signs[edge_idx]
            current_signs[edge_idx] = -old_sign
            
            new_valid, new_radius = _is_valid_lift(graph, current_signs)
            
            if new_valid and new_radius < current_radius:
                current_radius = new_radius
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[edge_idx] = old_sign
        
        if not improved:
            break
    
    return best_signs, best_radius

# Global time tracking
_sa_start_time = None

def sign_edges(graph, budget):
    global _sa_start_time
    
    m = graph['n'] // 2
    n_edges = 3 * m
    edges = graph['edges']
    n = graph['n']
    
    _sa_start_time = time.time()
    deadline = 55.0  # Safety margin
    
    # For small m, try exhaustive search
    if m <= 6:
        best_signs, best_radius = _exhaustive_search(graph, m)
        if best_signs is not None:
            return best_signs
    
    # Try simulated annealing
    sa_signs, sa_radius = _simulated_annealing(graph, m, max_time=deadline - (time.time() - _sa_start_time))
    
    if sa_signs is not None:
        # Refine with local search
        refined_signs, refined_radius = _local_search(graph, sa_signs, max_time=3.0)
        return refined_signs
    
    # Fallback: seed program
    from autoresearch_suite.science_tcs_expansion import lift_baseline
    return lift_baseline(graph, budget)
# EVOLVE-BLOCK-END
