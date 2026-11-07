"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
    return float(np.max(np.abs(eigvals)))

def _is_valid_lift(graph, signs):
    """Check if the lift is connected and 3-regular using check_lift."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, radius
    except Exception:
        return False, None

def _phase_to_signs(phases, graph):
    """Convert continuous phases to discrete signs."""
    n_edges = len(graph['edges'])
    signs = np.ones(n_edges, dtype=np.int8)
    for i, phase in enumerate(phases):
        if phase < 0:
            signs[i] = -1
    return signs

def _continuous_objective(phases, graph):
    """
    Compute a smooth surrogate for the spectral radius using the phase representation.
    For each edge with phase theta_i, the contribution to the signed adjacency is
    cos(theta_i) (real part of e^{i*theta_i}). We build a real symmetric matrix
    with these entries and compute its spectral radius.
    """
    n = graph['n']
    A = np.zeros((n, n))
    for i, (edge, phase) in enumerate(zip(graph['edges'], phases)):
        u, v = edge
        val = np.cos(phase)
        A[u, v] = val
        A[v, u] = val
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))

def _gradient_descent(phases, graph, lr=0.1, max_iter=500, tol=1e-8):
    """
    Perform gradient descent on the continuous phase space to minimize spectral radius.
    Uses numerical gradient (finite differences) since analytic gradient is complex.
    """
    n_phases = len(phases)
    best_phases = phases.copy()
    best_obj = _continuous_objective(phases, graph)
    
    for it in range(max_iter):
        # Compute numerical gradient
        grad = np.zeros(n_phases)
        h = 1e-5
        for i in range(n_phases):
            phases_plus = phases.copy()
            phases_plus[i] += h
            obj_plus = _continuous_objective(phases_plus, graph)
            
            phases_minus = phases.copy()
            phases_minus[i] -= h
            obj_minus = _continuous_objective(phases_minus, graph)
            
            grad[i] = (obj_plus - obj_minus) / (2 * h)
        
        # Gradient step
        phases_new = phases - lr * grad
        # Wrap phases to [-pi, pi]
        phases_new = (phases_new + np.pi) % (2 * np.pi) - np.pi
        
        obj_new = _continuous_objective(phases_new, graph)
        
        if obj_new < best_obj - tol:
            best_phases = phases_new.copy()
            best_obj = obj_new
        
        # Adaptive learning rate
        if obj_new > best_obj:
            lr *= 0.8
        else:
            lr = min(lr * 1.05, 0.5)
        
        phases = phases_new
    
    return best_phases, best_obj

def _local_search(graph, signs, max_iter=2000):
    """
    Local search: flip one sign at a time to minimize spectral radius,
    ensuring validity is maintained.
    """
    n_edges = len(graph['edges'])
    best_signs = signs.copy()
    best_radius = None
    try:
        valid, radius = _is_valid_lift(graph, best_signs)
        if not valid:
            return None
        best_radius = radius
    except Exception:
        return None
    
    for it in range(max_iter):
        improved = False
        for i in range(n_edges):
            trial = best_signs.copy()
            trial[i] = -trial[i]
            try:
                valid, radius = _is_valid_lift(graph, trial)
                if valid and radius < best_radius - 1e-10:
                    best_signs = trial
                    best_radius = radius
                    improved = True
                    break
            except Exception:
                pass
        if not improved:
            break
    
    return best_signs

def sign_edges(graph, budget):
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Initial random sampling with fixed seed
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = float('inf')
    
    # Phase 1: Random sampling
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=n_edges).astype(np.int8)
        try:
            valid, radius = _is_valid_lift(graph, signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        except Exception:
            pass
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(n_edges, dtype=np.int8)
        try:
            valid, radius = _is_valid_lift(graph, best_signs)
            if valid:
                best_radius = radius
        except Exception:
            pass
    
    # Phase 2: Continuous optimization via gradient descent on phases
    if time.time() < deadline:
        # Initialize phases: for +1, phase=0; for -1, phase=pi
        phases = np.zeros(n_edges)
        for i in range(n_edges):
            if best_signs[i] == -1:
                phases[i] = np.pi
            else:
                phases[i] = 0.0
        
        # Run gradient descent with multiple restarts
        for restart in range(3):
            if time.time() >= deadline:
                break
            if restart == 0:
                init_phases = phases.copy()
            else:
                init_phases = phases + rng.normal(0, 0.5, size=n_edges)
                init_phases = (init_phases + np.pi) % (2 * np.pi) - np.pi
            
            opt_phases, opt_obj = _gradient_descent(init_phases, graph, lr=0.1, max_iter=200)
            
            # Convert to signs
            trial_signs = _phase_to_signs(opt_phases, graph)
            try:
                valid, radius = _is_valid_lift(graph, trial_signs)
                if valid and radius < best_radius:
                    best_radius = radius
                    best_signs = trial_signs.copy()
            except Exception:
                pass
    
    # Phase 3: Local search to polish
    if time.time() < deadline and best_signs is not None:
        refined = _local_search(graph, best_signs, max_iter=1000)
        if refined is not None:
            try:
                valid, radius = _is_valid_lift(graph, refined)
                if valid and radius < best_radius:
                    best_radius = radius
                    best_signs = refined.copy()
            except Exception:
                pass
    
    # Final check
    try:
        valid, radius = _is_valid_lift(graph, best_signs)
        if not valid:
            # Fallback to all +1
            best_signs = np.ones(n_edges, dtype=np.int8)
    except Exception:
        best_signs = np.ones(n_edges, dtype=np.int8)
    
    return best_signs
# EVOLVE-BLOCK-END
