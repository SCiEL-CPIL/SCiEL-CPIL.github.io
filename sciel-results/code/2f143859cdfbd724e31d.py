"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def _build_cut_matrix(n, edges):
    """Build the cut matrix C where C[i,j] = 1 if (i,j) in edges."""
    C = np.zeros((n, n), dtype=float)
    for (u, v) in edges:
        C[u, v] = 1.0
        C[v, u] = 1.0
    return C


def _expected_cut_value(n, edges, gammas, betas):
    """
    Compute the expected cut value for a given QAOA circuit.
    Uses exact state simulation for small n.
    """
    num_states = 1 << n
    # Start in |+>^n = uniform superposition
    psi = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    # Build the diagonal of the cost Hamiltonian C
    # C = sum_{(i,j) in edges} (I - Z_i Z_j) / 2
    # For each basis state, the eigenvalue of C is the number of cut edges
    C_diag = np.zeros(num_states, dtype=float)
    for state in range(num_states):
        bits = [(state >> i) & 1 for i in range(n)]
        cut_val = 0
        for (u, v) in edges:
            if bits[u] != bits[v]:
                cut_val += 1
        C_diag[state] = cut_val
    
    # Build the diagonal of the mixer Hamiltonian M = sum X_i
    # This is not diagonal in Z basis, so we need to apply it as a unitary
    # Actually, let's use a different approach: apply each layer as a unitary
    
    # For efficiency, let's precompute the action of exp(-i*gamma*C) and exp(-i*beta*M)
    # exp(-i*gamma*C) is diagonal: multiply each component by exp(-i*gamma*C_diag[state])
    # exp(-i*beta*M) where M = sum X_i
    # X_i flips bit i, so exp(-i*beta*M) = product over i of exp(-i*beta*X_i)
    # Each exp(-i*beta*X_i) = cos(beta)*I - i*sin(beta)*X_i
    
    # Apply layers
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i*gamma*C): diagonal
        phase = np.exp(-1j * gamma * C_diag)
        psi = psi * phase
        
        # Apply exp(-i*beta*M) = product_i (cos(beta)*I - i*sin(beta)*X_i)
        # This is equivalent to applying each single-qubit rotation
        # For each qubit i, apply the 2x2 matrix [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
        # We can do this by iterating over states and applying the transformation
        # More efficiently: for each bit position, we can pair states that differ only in that bit
        
        c = np.cos(beta)
        s = np.sin(beta)
        
        for i in range(n):
            # For states where bit i is 0, pair with state where bit i is 1
            # Apply [[c, -is], [-is, c]] to the pair
            mask = 1 << i
            for state in range(num_states):
                if (state & mask) == 0:
                    paired = state | mask
                    psi0 = psi[state]
                    psi1 = psi[paired]
                    psi[state] = c * psi0 - 1j * s * psi1
                    psi[paired] = -1j * s * psi0 + c * psi1
    
    # Compute expected cut: sum over states of |psi[state]|^2 * C_diag[state]
    probs = np.abs(psi) ** 2
    expected_cut = np.sum(probs * C_diag)
    
    return expected_cut


def _exhaustive_maxcut(n, edges):
    """Compute the exact maximum cut by exhaustive search."""
    best = 0
    for state in range(1 << n):
        bits = [(state >> i) & 1 for i in range(n)]
        cut_val = 0
        for (u, v) in edges:
            if bits[u] != bits[v]:
                cut_val += 1
        if cut_val > best:
            best = cut_val
    return best


def _optimize_angles_for_graph(n, edges, p, time_budget=50.0):
    """Optimize QAOA angles for a specific graph using Nelder-Mead."""
    start_time = time.time()
    
    # Initial guess
    best_angles = np.array([0.35, 0.25, 0.35, 0.25])
    best_value = -1
    
    # Compute optimal cut value
    opt_cut = _exhaustive_maxcut(n, edges)
    if opt_cut == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    def objective(params):
        nonlocal best_angles, best_value
        gammas = params[:p]
        betas = params[p:]
        try:
            val = _expected_cut_value(n, edges, gammas, betas)
            ratio = val / opt_cut
            if ratio > best_value:
                best_value = ratio
                best_angles = params.copy()
            return -ratio  # Minimize negative for maximization
        except Exception:
            return 1e10
    
    # Nelder-Mead implementation
    def nelder_mead(func, x0, max_iter=200, tol=1e-8, time_limit=None):
        n_params = len(x0)
        # Create initial simplex
        simplex = [x0.copy()]
        for i in range(n_params):
            xi = x0.copy()
            xi[i] += 0.1 if xi[i] == 0 else 0.1 * xi[i]
            simplex.append(xi)
        
        # Evaluate initial simplex
        fvals = [func(x) for x in simplex]
        
        iter_count = 0
        while iter_count < max_iter:
            if time_limit is not None and (time.time() - time_limit) > 0:
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (xr - centroid)
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                # Contraction
                rho = 0.5
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n_params + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
            
            iter_count += 1
        
        return simplex[0], fvals[0]
    
    # Run optimization with time guard
    time_limit = start_time + time_budget
    try:
        x_opt, f_opt = nelder_mead(objective, best_angles, max_iter=300, time_limit=time_limit)
        if f_opt < -1:  # Valid optimization
            best_angles = x_opt
    except Exception:
        pass
    
    # Ensure angles are in reasonable range
    best_angles = np.clip(best_angles, -np.pi, np.pi)
    
    # Format output
    result = np.zeros((p, 2))
    result[:, 0] = best_angles[:p]  # gammas
    result[:, 1] = best_angles[p:]  # betas
    
    return result


def angles(n, edges, p):
    # Time guard: must finish within 120s total
    start_time = time.time()
    time_budget = 100.0  # Leave 20s margin
    
    try:
        result = _optimize_angles_for_graph(n, edges, p, time_budget=time_budget)
    except Exception:
        # Fallback to reasonable default
        result = np.tile([0.35, 0.25], (p, 1))
    
    # Final time check
    elapsed = time.time() - start_time
    if elapsed > 115.0:
        # Should not happen, but return what we have
        pass
    
    return result
# EVOLVE-BLOCK-END
