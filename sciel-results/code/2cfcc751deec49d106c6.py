"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain."""
import numpy as np
import time
import sys

_DEADLINE = 130.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    
    # Build full operators using kron
    def kron_all(ops):
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    I2 = np.eye(2, dtype=complex)
    
    # XXZ interaction on bond (i, j): delta*(Sx_i*Sx_j + Sy_i*Sy_j) + Sz_i*Sz_j
    # = 0.5*delta*(Sx_i*Sx_j + Sy_i*Sy_j) + Sz_i*Sz_j  (since Sx*Sx + Sy*Sy = 0.5*(S+*S- + S-*S+))
    # Actually: Sx_i*Sx_j + Sy_i*Sy_j = 0.5*(S+_i*S-_j + S-_i*S+_j)
    # XXZ: delta*(Sx_i*Sx_j + Sy_i*Sy_j) + Sz_i*Sz_j
    
    def bond_op(i, j):
        ops = [I2] * N
        # Sx_i * Sx_j
        ops_x = [I2] * N
        ops_x[i] = sx
        ops_x[j] = sx
        term_xx = kron_all(ops_x)
        
        # Sy_i * Sy_j
        ops_y = [I2] * N
        ops_y[i] = sy
        ops_y[j] = sy
        term_yy = kron_all(ops_y)
        
        # Sz_i * Sz_j
        ops_z = [I2] * N
        ops_z[i] = sz
        ops_z[j] = sz
        term_zz = kron_all(ops_z)
        
        return delta * (term_xx + term_yy) + term_zz
    
    # Odd bonds: (1,2) and (3,4)
    A = bond_op(1, 2) + bond_op(3, 4)
    # Even bonds: (0,1), (2,3), (4,5)
    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    
    # Half field on each: -h/2 * sum_i Sz_i
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for site in range(N):
        ops = [I2] * N
        ops[site] = sz
        field_op += kron_all(ops)
    
    A += -field / 2.0 * field_op
    B += -field / 2.0 * field_op
    
    return A, B

def _compute_error(a, b, A, B, t_values):
    """Compute mean spectral norm error over t_values."""
    N = A.shape[0]
    total_err = 0.0
    for t in t_values:
        # Build U(t) = prod_k exp(-i*b_k*t*B) * exp(-i*a_k*t*A)
        U = np.eye(N, dtype=complex)
        for k in range(len(a)):
            # exp(-i * b_k * t * B)
            eB = np.exp(-1j * b[k] * t * B)
            # exp(-i * a_k * t * A)
            eA = np.exp(-1j * a[k] * t * A)
            U = eB @ eA
        
        # Exact: exp(-i * t * (A + B))
        H = A + B
        U_exact = np.exp(-1j * t * H)
        
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    return total_err / len(t_values)

def _gradient_descent(A, B, t_values, max_iter=200, lr=0.01):
    """Optimize coefficients using projected gradient descent."""
    stages = 12
    # Initialize with symmetric pattern
    a = [1.0/12] * stages
    b = [1.0/12] * stages
    
    best_a = a[:]
    best_b = b[:]
    best_err = _compute_error(a, b, A, B, t_values)
    
    for it in range(max_iter):
        if _remaining() < 10.0:
            break
        
        # Compute gradient numerically
        grad_a = np.zeros(stages)
        grad_b = np.zeros(stages)
        eps = 1e-5
        
        for i in range(stages):
            a_plus = a[:]
            a_plus[i] += eps
            # Project: adjust to maintain sum=1
            a_plus = np.array(a_plus)
            a_plus = a_plus - (a_plus.sum() - 1.0) / stages
            err_plus = _compute_error(a_plus.tolist(), b, A, B, t_values)
            grad_a[i] = (err_plus - best_err) / eps
            
            b_plus = b[:]
            b_plus[i] += eps
            b_plus = np.array(b_plus)
            b_plus = b_plus - (b_plus.sum() - 1.0) / stages
            err_plus = _compute_error(a, b_plus.tolist(), A, B, t_values)
            grad_b[i] = (err_plus - best_err) / eps
        
        # Gradient step
        a_new = np.array(a) - lr * grad_a
        b_new = np.array(b) - lr * grad_b
        
        # Project to sum=1
        a_new = a_new - (a_new.sum() - 1.0) / stages
        b_new = b_new - (b_new.sum() - 1.0) / stages
        
        # Clip to [-50, 50]
        a_new = np.clip(a_new, -50, 50)
        b_new = np.clip(b_new, -50, 50)
        
        # Re-project after clipping
        a_new = a_new - (a_new.sum() - 1.0) / stages
        b_new = b_new - (b_new.sum() - 1.0) / stages
        
        a = a_new.tolist()
        b = b_new.tolist()
        
        err = _compute_error(a, b, A, B, t_values)
        if err < best_err:
            best_err = err
            best_a = a[:]
            best_b = b[:]
        
        # Adaptive learning rate
        if it % 20 == 0 and it > 0:
            lr *= 0.8
    
    return best_a, best_b, best_err

def _try_symmetric_patterns(A, B, t_values):
    """Try various symmetric coefficient patterns."""
    stages = 12
    results = []
    
    # Pattern 1: uniform
    a = [1.0/stages] * stages
    b = [1.0/stages] * stages
    results.append((a, b, _compute_error(a, b, A, B, t_values)))
    
    # Pattern 2: Strang-like (a at edges, b in middle)
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        a = [0.0] * stages
        b = [0.0] * stages
        # Put weight on a at positions 0, 11 and b at positions 1-10
        a[0] = w
        a[11] = w
        remaining = 1.0 - 2*w
        if remaining > 0:
            for i in range(1, 11):
                a[i] = remaining / 10
            for i in range(12):
                b[i] = 1.0/12
        else:
            for i in range(12):
                a[i] = 1.0/12
                b[i] = 1.0/12
        results.append((a, b, _compute_error(a, b, A, B, t_values)))
    
    # Pattern 3: alternating emphasis
    for alpha in [0.0, 0.1, 0.2, 0.3]:
        a = [0.0] * stages
        b = [0.0] * stages
        for i in range(stages):
            if i % 2 == 0:
                a[i] = (1.0 + alpha) / stages
                b[i] = (1.0 - alpha) / stages
            else:
                a[i] = (1.0 - alpha) / stages
                b[i] = (1.0 + alpha) / stages
        # Normalize
        a_sum = sum(a)
        b_sum = sum(b)
        a = [x / a_sum for x in a]
        b = [x / b_sum for x in b]
        results.append((a, b, _compute_error(a, b, A, B, t_values)))
    
    # Pattern 4: quadratic weight
    for c in [0.0, 0.5, 1.0, 1.5, 2.0]:
        a = [0.0] * stages
        b = [0.0] * stages
        for i in range(stages):
            x = (i - 5.5) / 5.5  # centered, range ~[-1, 1]
            w = 1.0 + c * x * x
            a[i] = w
            b[i] = w
        a_sum = sum(a)
        b_sum = sum(b)
        a = [x / a_sum for x in a]
        b = [x / b_sum for x in b]
        results.append((a, b, _compute_error(a, b, A, B, t_values)))
    
    # Pattern 5: linear ramp
    for slope in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        a = [0.0] * stages
        b = [0.0] * stages
        for i in range(stages):
            x = (i - 5.5) / 5.5
            w = 1.0 + slope * x
            if w < 0.01:
                w = 0.01
            a[i] = w
            b[i] = w
        a_sum = sum(a)
        b_sum = sum(b)
        a = [x / a_sum for x in a]
        b = [x / b_sum for x in b]
        results.append((a, b, _compute_error(a, b, A, B, t_values)))
    
    # Sort by error
    results.sort(key=lambda x: x[2])
    return results

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    if stages != 12:
        # Fallback: uniform
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    A, B = _build_hamiltonians()
    t_values = [0.25, 0.5, 1.0]
    
    # First try symmetric patterns
    patterns = _try_symmetric_patterns(A, B, t_values)
    best_a, best_b, best_err = patterns[0]
    
    # Check time
    if _remaining() < 30.0:
        return best_a, best_b
    
    # Try gradient descent from the best pattern
    try:
        gd_a, gd_b, gd_err = _gradient_descent(A, B, t_values, max_iter=100, lr=0.05)
        if gd_err < best_err:
            best_a, best_b, best_err = gd_a, gd_b, gd_err
    except Exception:
        pass
    
    # Try a few more gradient descents from different starting points
    if _remaining() > 20.0:
        for start_idx in [1, 2, 3]:
            if start_idx < len(patterns) and _remaining() > 15.0:
                try:
                    start_a = patterns[start_idx][0][:]
                    start_b = patterns[start_idx][1][:]
                    # Modify gradient descent to start from different point
                    stages_count = 12
                    a = start_a[:]
                    b = start_b[:]
                    current_err = patterns[start_idx][2]
                    
                    lr = 0.05
                    for it in range(50):
                        if _remaining() < 10.0:
                            break
                        grad_a = np.zeros(stages_count)
                        grad_b = np.zeros(stages_count)
                        eps = 1e-5
                        
                        for i in range(stages_count):
                            a_plus = a[:]
                            a_plus[i] += eps
                            a_plus = np.array(a_plus)
                            a_plus = a_plus - (a_plus.sum() - 1.0) / stages_count
                            err_plus = _compute_error(a_plus.tolist(), b, A, B, t_values)
                            grad_a[i] = (err_plus - current_err) / eps
                            
                            b_plus = b[:]
                            b_plus[i] += eps
                            b_plus = np.array(b_plus)
                            b_plus = b_plus - (b_plus.sum() - 1.0) / stages_count
                            err_plus = _compute_error(a, b_plus.tolist(), A, B, t_values)
                            grad_b[i] = (err_plus - current_err) / eps
                        
                        a_new = np.array(a) - lr * grad_a
                        b_new = np.array(b) - lr * grad_b
                        
                        a_new = a_new - (a_new.sum() - 1.0) / stages_count
                        b_new = b_new - (b_new.sum() - 1.0) / stages_count
                        
                        a_new = np.clip(a_new, -50, 50)
                        b_new = np.clip(b_new, -50, 50)
                        
                        a_new = a_new - (a_new.sum() - 1.0) / stages_count
                        b_new = b_new - (b_new.sum() - 1.0) / stages_count
                        
                        a = a_new.tolist()
                        b = b_new.tolist()
                        
                        err = _compute_error(a, b, A, B, t_values)
                        if err < current_err:
                            current_err = err
                            if current_err < best_err:
                                best_a, best_b, best_err = a[:], b[:], current_err
                        else:
                            lr *= 0.7
                    
                except Exception:
                    pass
    
    # Final validation: ensure sums are exactly 1 and values in range
    best_a = [max(-50, min(50, x)) for x in best_a]
    best_b = [max(-50, min(50, x)) for x in best_b]
    
    # Re-normalize to ensure exact sum
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x / sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x / sb for x in best_b]
    
    return best_a, best_b
