"""Optimized product formula design for Heisenberg chain evolution."""
import numpy as np
import scipy.linalg as la
import time
from scipy.optimize import minimize

def _build_hamiltonians():
    """
    Build A and B for 6-site open XXZ Heisenberg chain.
    A: odd bonds + half field
    B: even bonds + half field
    Anisotropy lambda = 0.8, field h = 0.3
    """
    N = 6
    lambda_aniso = 0.8
    h_field = 0.3
    
    # Spin operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def op_on_site(op, site, N):
        ops = [np.eye(2, dtype=complex) for _ in range(N)]
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result

    def bond_op(op1, op2, site1, site2, N):
        # Create operator op1 on site1 and op2 on site2
        ops = [np.eye(2, dtype=complex) for _ in range(N)]
        ops[site1] = op1
        ops[site2] = op2
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result

    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field term: h * sum sz_i
    # Split field: half in A, half in B
    field_term = 0.0
    for i in range(N):
        field_term += h_field * op_on_site(sz, i, N)
    
    A += 0.5 * field_term
    B += 0.5 * field_term
    
    # Bond terms
    # XXZ Hamiltonian: J (Sx Sx + Sy Sy + lambda Sz Sz)
    # Usually J=1.
    J = 1.0
    
    for i in range(N - 1):
        bond = J * (bond_op(sx, sx, i, i+1, N) + 
                    bond_op(sy, sy, i, i+1, N) + 
                    lambda_aniso * bond_op(sz, sz, i, i+1, N))
        if i % 2 == 0:
            # Even bond (0-1, 2-3, 4-5) -> B
            B += bond
        else:
            # Odd bond (1-2, 3-4) -> A
            A += bond
            
    return A, B

def _precompute_eigs(A, B):
    """Compute eigen-decompositions for A, B, and A+B."""
    # A is Hermitian
    wA, vA = la.eigh(A)
    # B is Hermitian
    wB, vB = la.eigh(B)
    # A+B is Hermitian
    H = A + B
    wH, vH = la.eigh(H)
    
    # Pre-compute conjugate transposes
    vA_dag = vA.conj().T
    vB_dag = vB.conj().T
    vH_dag = vH.conj().T
    
    return wA, vA, vA_dag, wB, vB, vB_dag, wH, vH, vH_dag

def _compute_error(a, b, wA, vA, vA_dag, wB, vB, vB_dag, wH, vH, vH_dag, times):
    """
    Compute mean spectral norm error.
    a, b: lists of coefficients.
    """
    D = len(wA)
    I = np.eye(D, dtype=complex)
    
    total_err = 0.0
    
    for t in times:
        # Exact propagator
        # U_exact = vH exp(-i t wH) vH_dag
        exp_wH = np.exp(-1j * t * wH)
        temp_exact = vH_dag * exp_wH
        U_exact = vH @ temp_exact
        
        # Product formula
        U = I.copy()
        
        for k in range(len(a)):
            ak = a[k]
            bk = b[k]
            
            # Step 1: exp(-i bk t B)
            # U = vB @ (exp(-i bk t wB) * (vB_dag @ U))
            temp = vB_dag @ U
            exp_wB = np.exp(-1j * bk * t * wB)
            temp = temp * exp_wB
            U = vB @ temp
            
            # Step 2: exp(-i ak t A)
            # U = vA @ (exp(-i ak t wA) * (vA_dag @ U))
            temp = vA_dag @ U
            exp_wA = np.exp(-1j * ak * t * wA)
            temp = temp * exp_wA
            U = vA @ temp
            
        # Error: spectral norm of U - U_exact
        diff = U - U_exact
        # Use norm 2 (spectral norm)
        err = la.norm(diff, 2)
        total_err += err
        
    return total_err / len(times)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    wA, vA, vA_dag, wB, vB, vB_dag, wH, vH, vH_dag = _precompute_eigs(A, B)
    
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: Uniform
    a0 = [1.0/stages] * stages
    b0 = [1.0/stages] * stages
    
    # Parameterization: optimize a_0..a_10, b_0..b_10
    # a_11 = 1 - sum(a_0..a_10)
    # b_11 = 1 - sum(b_0..b_10)
    
    def objective(x):
        a = list(x[:stages-1])
        b = list(x[stages-1:])
        
        a_sum = sum(a)
        b_sum = sum(b)
        
        a.append(1.0 - a_sum)
        b.append(1.0 - b_sum)
        
        # Check bounds
        for val in a + b:
            if val < -50.0 or val > 50.0:
                return 1e10
        
        return _compute_error(a, b, wA, vA, vA_dag, wB, vB, vB_dag, wH, vH, vH_dag, times)
    
    x0 = np.array(a0[:stages-1] + b0[:stages-1])
    
    best_x = x0.copy()
    best_f = objective(x0)
    
    # Nelder-Mead
    xatol = 1e-5
    fatol = 1e-5
    
    # Perform multiple restarts if time permits
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            current_x = best_x
        else:
            # Random restart near best solution
            perturbation = np.random.normal(0, 0.5, best_x.shape)
            current_x = best_x + perturbation
            current_x = np.clip(current_x, -5, 5)
            
        res = minimize(objective, current_x, method='Nelder-Mead', 
                       options={'xatol': xatol, 'fatol': fatol, 'maxiter': 200, 'adaptive': True})
        
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
            
    # Reconstruct a, b
    a_opt = list(best_x[:stages-1])
    b_opt = list(best_x[stages-1:])
    
    a_opt.append(1.0 - sum(a_opt))
    b_opt.append(1.0 - sum(b_opt))
    
    # Final check and clipping
    a_final = []
    for val in a_opt:
        if val < -50: val = -50
        if val > 50: val = 50
        a_final.append(val)
        
    b_final = []
    for val in b_opt:
        if val < -50: val = -50
        if val > 50: val = 50
        b_final.append(val)
        
    # Ensure sums are exactly 1 (floating point correction)
    sum_a = sum(a_final)
    if abs(sum_a - 1.0) > 1e-12:
        a_final[-1] += 1.0 - sum_a
        
    sum_b = sum(b_final)
    if abs(sum_b - 1.0) > 1e-12:
        b_final[-1] += 1.0 - sum_b
        
    return a_final, b_final

