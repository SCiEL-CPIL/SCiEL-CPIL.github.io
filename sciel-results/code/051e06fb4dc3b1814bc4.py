import numpy as np
import scipy.optimize
import time

def _build_hamiltonians():
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy
    sm = sx - 1j * sy
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def bond_op(op1, op2, site1, site2):
        ops = [I2] * N
        ops[site1] = op1
        ops[site2] = op2
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) in 0-indexed
    odd_bonds = [(0,1), (2,3), (4,5)]
    # Even bonds: (1,2), (3,4) in 0-indexed
    even_bonds = [(1,2), (3,4)]
    
    for i, j in odd_bonds:
        A += gamma/2 * (bond_op(sp, sm, i, j) + bond_op(sm, sp, i, j)) + bond_op(sz, sz, i, j)
    for i, j in even_bonds:
        B += gamma/2 * (bond_op(sp, sm, i, j) + bond_op(sm, sp, i, j)) + bond_op(sz, sz, i, j)
        
    # Field term: -h sum S_j^z
    # Split half to A, half to B
    field_term = np.zeros((dim, dim), dtype=complex)
    for site in range(N):
        field_term += op_on_site(sz, site)
        
    A += -0.5 * h * field_term
    B += -0.5 * h * field_term
    
    return A, B

def _precompute_eigen(A, B):
    VA, LA = np.linalg.eigh(A)
    VB, LB = np.linalg.eigh(B)
    H = A + B
    VH, LH = np.linalg.eigh(H)
    return VA, LA, VB, LB, VH, LH

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin for 60s limit
    
    A, B = _build_hamiltonians()
    VA, LA, VB, LB, VH, LH = _precompute_eigen(A, B)
    
    # Precompute exact propagators
    t_values = [0.25, 0.5, 1.0]
    U_exact = []
    for t in t_values:
        U_exact.append(VH @ np.diag(np.exp(-1j * t * LH)) @ VH.conj().T)
        
    def cost(params):
        if time.time() > deadline:
            raise TimeoutError()
        a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 = params
        a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
        b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
        
        a_half = [a1, a2, a3, a4, a5, a6]
        b_half = [b1, b2, b3, b4, b5, b6]
        
        a_full = a_half + a_half[::-1]
        b_full = b_half + b_half[::-1]
        
        # Check bounds
        if np.any(np.abs(a_full) > 50) or np.any(np.abs(b_full) > 50):
            return 1e10
            
        total_err = 0.0
        for idx, t in enumerate(t_values):
            U = np.eye(64, dtype=complex)
            for k in range(12):
                # e^{-i b_k t B}
                exp_b = VB @ np.diag(np.exp(-1j * b_full[k] * t * LB)) @ VB.conj().T
                U = U @ exp_b
                # e^{-i a_k t A}
                exp_a = VA @ np.diag(np.exp(-1j * a_full[k] * t * LA)) @ VA.conj().T
                U = U @ exp_a
            
            err = np.linalg.norm(U - U_exact[idx], 2)
            total_err += err
        return total_err

    # Initial guess: Symmetric Strang-like
    # a = [0.5/12]*12? No, sum(a)=1.
    # Symmetric: a1..a6, a6..a1. Sum = 2*sum(a1..a6) = 1 => sum(a1..a6)=0.5.
    # Uniform guess: a_i = 0.5/6 = 1/12.
    init_a = [1/12] * 5
    init_b = [1/12] * 5
    init_params = init_a + init_b
    
    # Try multiple starting points
    best_result = None
    best_cost = np.inf
    
    # Starting point 1: Uniform
    x0 = np.array(init_params)
    res = scipy.optimize.minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
    if res.fun < best_cost:
        best_cost = res.fun
        best_result = res.x
        
    # Starting point 2: Random
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        x0_rand = np.random.uniform(-0.1, 0.1, 10)
        # Ensure sums don't lead to huge a6/b6 initially? 
        # The cost function handles bounds.
        try:
            res_rand = scipy.optimize.minimize(cost, x0_rand, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res_rand.fun < best_cost:
                best_cost = res_rand.fun
                best_result = res_rand.x
        except TimeoutError:
            break
        except Exception:
            continue
            
    if best_result is None:
        # Fallback to seed
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 = best_result
    a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
    b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
    
    a_half = [a1, a2, a3, a4, a5, a6]
    b_half = [b1, b2, b3, b4, b5, b6]
    
    a_full = a_half + a_half[::-1]
    b_full = b_half + b_half[::-1]
    
    return a_full, b_full
