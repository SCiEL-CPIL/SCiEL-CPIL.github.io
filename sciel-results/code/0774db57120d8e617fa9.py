import numpy as np
import time
import scipy.linalg
from scipy.optimize import minimize

def _build_hamiltonians():
    N = 6
    lam = 0.8
    h = 0.3
    
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    I = np.eye(2, dtype=complex)
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    field_A = 0.0
    field_B = 0.0
    for i in range(N):
        ops_f = [I] * N
        ops_f[i] = sz
        field_op = kron_all(ops_f)
        field_A += 0.5 * h * field_op
        field_B += 0.5 * h * field_op
        
    A += field_A
    B += field_B
    
    for i in [0, 2, 4]:
        ops_xx = [I] * N
        ops_xx[i] = sx
        ops_xx[i+1] = sx
        A += kron_all(ops_xx)
        
        ops_yy = [I] * N
        ops_yy[i] = sy
        ops_yy[i+1] = sy
        A += kron_all(ops_yy)
        
        ops_zz = [I] * N
        ops_zz[i] = sz
        ops_zz[i+1] = sz
        A += lam * kron_all(ops_zz)
        
    for i in [1, 3]:
        ops_xx = [I] * N
        ops_xx[i] = sx
        ops_xx[i+1] = sx
        B += kron_all(ops_xx)
        
        ops_yy = [I] * N
        ops_yy[i] = sy
        ops_yy[i+1] = sy
        B += kron_all(ops_yy)
        
        ops_zz = [I] * N
        ops_zz[i] = sz
        ops_zz[i+1] = sz
        B += lam * kron_all(ops_zz)
        
    return A, B

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 110.0
    
    A, B = _build_hamiltonians()
    
    wA, VA = np.linalg.eigh(A)
    wB, VB = np.linalg.eigh(B)
    
    t_vals = [0.25, 0.5, 1.0]
    wAB, VAB = np.linalg.eigh(A + B)
    exact_propagators = []
    for t in t_vals:
        exact_propagators.append((VAB * np.exp(-1j * t * wAB)) @ VAB.conj().T)
        
    def expm_A(c, t):
        return (VA * np.exp(-1j * c * t * wA)) @ VA.conj().T
        
    def expm_B(c, t):
        return (VB * np.exp(-1j * c * t * wB)) @ VB.conj().T
        
    def objective(params):
        a = params[:stages]
        b = params[stages:]
        
        penalty = 100.0 * (abs(np.sum(a) - 1.0) + abs(np.sum(b) - 1.0))
        
        err = 0.0
        for idx, t in enumerate(t_vals):
            U = np.eye(2**6, dtype=complex)
            for k in range(stages):
                U = expm_B(b[k], t) @ U
                U = expm_A(a[k], t) @ U
            diff = U - exact_propagators[idx]
            err += np.linalg.norm(diff, 2)
            
        return err / len(t_vals) + penalty

    best_params = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    for i in range(10):
        if time.time() > deadline - 20:
            break
            
        x0 = np.random.uniform(-0.5, 0.5, size=2*stages)
        
        bounds = [(-50, 50)] * (2*stages)
        
        res = minimize(objective, x0, method='Powell', bounds=bounds, 
                       options={'maxiter': 200, 'maxfev': 2000})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    if best_params is None:
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
        
    a = list(best_params[:stages])
    b = list(best_params[stages:])
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a != 0:
        a = [x / sum_a for x in a]
    if sum_b != 0:
        b = [x / sum_b for x in b]
        
    a = [max(-50, min(50, x)) for x in a]
    b = [max(-50, min(50, x)) for x in b]
    
    return a, b
