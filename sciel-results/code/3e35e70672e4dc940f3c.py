"""Product-formula design for Heisenberg-chain time evolution."""
import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

_DEADLINE = 150.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START) - 5.0  # 5s safety margin

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def site_op(op, site):
        """Apply op on `site`, identity elsewhere."""
        result = np.array([[1.0]], dtype=complex)
        for i in range(N):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, identity)
        return result
    
    def tensor_product(ops_list):
        result = np.array([[1.0]], dtype=complex)
        for op in ops_list:
            result = np.kron(result, op)
        return result
    
    # XXZ interaction: (1/2)(sigma_x sigma_x + sigma_y sigma_y + anisotropy * sigma_z sigma_z)
    # = sigma_x sigma_x / 2 + sigma_y sigma_y / 2 + anisotropy/2 * sigma_z sigma_z
    # Actually standard Heisenberg: J * (sx*sx + sy*sy + anisotropy*sz*sz)
    # Let's use: H_bond = sx*sx + sy*sy + anisotropy*sz*sz (times some coupling)
    # For a unit-coupling Heisenberg chain with open boundaries:
    # H = sum_{i=0}^{N-2} (sx_i sx_{i+1} + sy_i sy_{i+1} + anisotropy * sz_i sz_{i+1})
    # Plus uniform field: h * sum_i sz_i
    
    # Odd bonds: (0,1), (2,3), (4,5) -> indices 0,2,4
    # Even bonds: (1,2), (3,4) -> indices 1,3
    # Field split: half to A, half to B
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Build bond operators
    for i in range(N - 1):
        bond = np.zeros((2**N, 2**N), dtype=complex)
        # sx_i sx_{i+1}
        ops = [identity] * N
        ops[i] = sx
        ops[i+1] = sx
        bond += tensor_product(ops)
        
        # sy_i sy_{i+1}
        ops = [identity] * N
        ops[i] = sy
        ops[i+1] = sy
        bond += tensor_product(ops)
        
        # anisotropy * sz_i sz_{i+1}
        ops = [identity] * N
        ops[i] = sz
        ops[i+1] = sz
        bond += anisotropy * tensor_product(ops)
        
        if i % 2 == 0:  # even-indexed bonds: (0,1), (2,3), (4,5)
            A += bond
        else:  # odd-indexed bonds: (1,2), (3,4)
            B += bond
    
    # Uniform field: h * sum_i sz_i, split half to A and half to B
    for i in range(N):
        ops = [identity] * N
        ops[i] = sz
        sz_i = tensor_product(ops)
        A += 0.5 * field * sz_i
        B += 0.5 * field * sz_i
    
    return A, B

def _compute_error(a, b, A, B, t_values):
    """Compute mean spectral norm error."""
    errors = []
    for t in t_values:
        U = np.eye(2**6, dtype=complex)
        for k in range(len(a)):
            U = expm(-1j * b[k] * t * B) @ U
            U = expm(-1j * a[k] * t * A) @ U
        U_exact = expm(-1j * t * (A + B))
        diff = U - U_exact
        # Spectral norm = largest singular value
        sv = np.linalg.svd(diff, compute_uv=False)
        errors.append(sv[0])
    return np.mean(errors)

def _objective(params, A, B, t_values):
    """Objective: mean spectral norm error with symmetric parameterization."""
    # params: 10 free parameters
    # Symmetric: a = [p0, p1, p2, p3, p4, p5, p5, p4, p3, p2, p1, p0]
    #            b = [q0, q1, q2, q3, q4, q5, q5, q4, q3, q2, q1, q0]
    # Constraints: sum(a) = 1, sum(b) = 1
    # sum(a) = 2*(p0+p1+p2+p3+p4+p5) = 1 => p0+...+p5 = 0.5
    # sum(b) = 2*(q0+q1+q2+q3+q4+q5) = 1 => q0+...+q5 = 0.5
    
    # Use parameterization: p5 = 0.5 - (p0+...+p4), q5 = 0.5 - (q0+...+q4)
    p = params[:5]
    q = params[5:10]
    p5 = 0.5 - np.sum(p)
    q5 = 0.5 - np.sum(q)
    
    a = np.array([p[0], p[1], p[2], p[3], p[4], p5, p5, p[4], p[3], p[2], p[1], p[0]])
    b = np.array([q[0], q[1], q[2], q[3], q[4], q5, q5, q[4], q[3], q[2], q[1], q[0]])
    
    # Check bounds
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return 1e10
    
    return _compute_error(a, b, A, B, t_values)

def coefficients(stages):
    """Optimized 12-stage symmetric product formula for Heisenberg chain."""
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 150.0
    
    t_values = [0.25, 0.5, 1.0]
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Initial guess: uniform symmetric
    # For 12 stages symmetric: 6 unique a values, 6 unique b values
    # Start with simple splitting
    init_a = np.array([1/12.0] * 6)  # will be doubled by symmetry
    init_b = np.array([1/12.0] * 6)
    # Adjust: sum of 6 values should be 0.5
    init_a = np.array([0.5/6.0] * 6)
    init_b = np.array([0.5/6.0] * 6)
    init_params = np.concatenate([init_a[:5], init_b[:5]])
    
    best_params = init_params.copy()
    best_score = _objective(init_params, A, B, t_values)
    
    # Try multiple starting points and local optimization
    starts = [init_params]
    
    # Perturbed starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        if _remaining() < 20:
            break
        pert = rng.normal(0, 0.05, size=10)
        starts.append(init_params + pert)
    
    # Also try known good symmetric patterns
    # Strang-like: concentrate weight in middle
    strang_a = np.array([0.02, 0.05, 0.10, 0.15, 0.18, 0.5 - sum([0.02,0.05,0.10,0.15,0.18])])
    strang_b = np.array([0.02, 0.05, 0.10, 0.15, 0.18, 0.5 - sum([0.02,0.05,0.10,0.15,0.18])])
    starts.append(np.concatenate([strang_a[:5], strang_b[:5]]))
    
    # Optimizer start: slightly non-uniform
    opt_a = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.5 - sum([0.04,0.06,0.08,0.10,0.12])])
    opt_b = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.5 - sum([0.04,0.06,0.08,0.10,0.12])])
    starts.append(np.concatenate([opt_a[:5], opt_b[:5]]))
    
    for start in starts:
        if _remaining() < 15:
            break
        
        # Evaluate initial
        score = _objective(start, A, B, t_values)
        if score < best_score:
            best_score = score
            best_params = start.copy()
        
        # L-BFGS-B optimization
        if _remaining() < 15:
            continue
        
        try:
            result = minimize(
                _objective,
                start,
                args=(A, B, t_values),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Polish: tighter optimization from best point
    if _remaining() > 10:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(A, B, t_values),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract final a, b
    p = best_params[:5]
    q = best_params[5:10]
    p5 = 0.5 - np.sum(p)
    q5 = 0.5 - np.sum(q)
    
    a = np.array([p[0], p[1], p[2], p[3], p[4], p5, p5, p[4], p[3], p[2], p[1], p[0]])
    b = np.array([q[0], q[1], q[2], q[3], q[4], q5, q5, q[4], q[3], q[2], q[1], q[0]])
    
    # Clamp to [-50, 50]
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Re-normalize to ensure exact sum = 1
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    if abs(sb) > 1e-15:
        b = b / sb
    
    # Final check
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Convert to lists
    a_list = [float(x) for x in a]
    b_list = [float(x) for x in b]
    
    return a_list, b_list
