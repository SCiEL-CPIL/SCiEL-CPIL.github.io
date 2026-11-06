import time
import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre, roots_chebyshev
from scipy.optimize import linprog

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0."""
    Omega_de = 1.0 - Omega_m
    if z == 0:
        return 1.0
    term_m = Omega_m * (1 + z)**3
    term_de = Omega_de * (1 + z)**(3 * (1 + w0 + wa * z))
    return np.sqrt(term_m + term_de)

def _true_integral(z, w0, wa, Omega_m=0.3):
    """Compute \int_0^1 dt / E(z*t) using quad."""
    if z == 0:
        return 1.0
    def integrand(t):
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    val, _ = quad(integrand, 0, 1, limit=100)
    return val

def _approx_integral(z, w0, wa, nodes, weights, Omega_m=0.3):
    """Compute \sum w_i / E(z*t_i)."""
    if z == 0:
        return np.sum(weights)
    z_nodes = z * nodes
    E_vals = _E_of_z(z_nodes, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E_vals)

def construct():
    # Deadline guard
    if time.time() > _DEADLINE - 5:
        n = 8
        nodes, weights = roots_legendre(n)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}

    # Define a dense test set of cosmologies and redshifts
    # w0 in [-1.5, -0.5], wa in [-0.5, 0.5], z in [0.01, 5.0]
    w0_vals = np.linspace(-1.5, -0.5, 10)
    wa_vals = np.linspace(-0.5, 0.5, 10)
    z_vals = np.logspace(-2, np.log10(5.0), 20)
    
    test_set = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                test_set.append((z, w0, wa))
                
    # Precompute true integrals and integrand values for candidate nodes
    # We will test a few node sets: GL, Chebyshev, and a few shifted/scaled variants
    
    candidate_nodes_list = []
    
    # 1. Gauss-Legendre on [0,1]
    gl_nodes, gl_weights = roots_legendre(8)
    candidate_nodes_list.append((gl_nodes + 1) / 2)
    
    # 2. Gauss-Chebyshev on [0,1]
    ch_nodes, ch_weights = roots_chebyshev(8)
    candidate_nodes_list.append((ch_nodes + 1) / 2)
    
    # 3. Equally spaced nodes
    candidate_nodes_list.append(np.linspace(0.05, 0.95, 8))
    
    # 4. Slightly shifted GL
    candidate_nodes_list.append((gl_nodes + 1) / 2 + 0.01 * np.sin(np.linspace(0, np.pi, 8)))
    
    best_nodes = None
    best_weights = None
    best_max_err = np.inf
    
    num_tests = len(test_set)
    n = 8
    
    for cand_nodes in candidate_nodes_list:
        if time.time() > _DEADLINE - 10:
            break
            
        # Ensure nodes are strictly increasing and in [0,1]
        cand_nodes = np.sort(cand_nodes)
        if np.any(cand_nodes <= 0) or np.any(cand_nodes >= 1) or np.any(np.diff(cand_nodes) <= 0):
            continue
            
        # Build matrix A and vector b
        # A[i, j] = 1/E(z_i * t_j)
        # b[i] = true_integral(z_i)
        A = np.zeros((num_tests, n))
        b = np.zeros(num_tests)
        
        for idx, (z, w0, wa) in enumerate(test_set):
            b[idx] = _true_integral(z, w0, wa)
            for j in range(n):
                A[idx, j] = 1.0 / _E_of_z(z * cand_nodes[j], w0, wa)
                
        # We want to minimize max_i | A[i] @ w - b[i] | / b[i]
        # Let e_i = A[i] @ w - b[i]
        # We want |e_i| / b[i] <= eps
        # => -eps * b[i] <= A[i] @ w - b[i] <= eps * b[i]
        # => b[i] - eps * b[i] <= A[i] @ w <= b[i] + eps * b[i]
        # => (1 - eps) b[i] <= A[i] @ w <= (1 + eps) b[i]
        
        # This is a linear programming problem with variable eps.
        # Minimize eps
        # Subject to:
        # A[i] @ w - (1 + eps) b[i] <= 0
        # (1 - eps) b[i] - A[i] @ w <= 0
        # sum(w) = 1
        # w >= 0
        
        # Variables: w[0..7], eps
        # c = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        
        c = np.zeros(n + 1)
        c[-1] = 1.0
        
        # Inequality constraints: A_ub @ x <= b_ub
        # 1. A[i] @ w - (1 + eps) b[i] <= 0  =>  A[i] @ w - b[i] * eps <= b[i]
        # 2. (1 - eps) b[i] - A[i] @ w <= 0  =>  -A[i] @ w - b[i] * eps <= -b[i]
        
        A_ub = np.zeros((2 * num_tests, n + 1))
        b_ub = np.zeros(2 * num_tests)
        
        for i in range(num_tests):
            # Row 2i
            A_ub[2*i, :n] = A[i]
            A_ub[2*i, n] = -b[i]
            b_ub[2*i] = b[i]
            
            # Row 2i+1
            A_ub[2*i+1, :n] = -A[i]
            A_ub[2*i+1, n] = -b[i]
            b_ub[2*i+1] = -b[i]
            
        # Equality constraints: sum(w) = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w >= 0, eps >= 0
        bounds = [(0, None)] * n + [(0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                eps = res.x[-1]
                w = res.x[:n]
                # Check if this is better
                if eps < best_max_err:
                    best_max_err = eps
                    best_nodes = cand_nodes
                    best_weights = w
        except Exception:
            pass
            
    if best_nodes is None:
        # Fallback to GL
        gl_nodes, gl_weights = roots_legendre(8)
        best_nodes = (gl_nodes + 1) / 2
        best_weights = gl_weights / 2
        
    # Final validation and normalization
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
