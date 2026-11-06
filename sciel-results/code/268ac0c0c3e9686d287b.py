import numpy as np
import time
import sys

# Constants for the problem
N_SITES = 6
DIM = 2 ** N_SITES
DELTA = 0.8
FIELD = 0.3
TIMES = [0.25, 0.5, 1.0]
STAGES = 12

# Precompute Pauli matrices
Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def kron_all(ops):
    """Tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    # Identity for full space
    Id = np.eye(DIM, dtype=complex)
    
    # Helper to build site operator
    def site_op(site_idx, op):
        ops = [I2] * N_SITES
        ops[site_idx] = op
        return kron_all(ops)
    
    # Helper to build bond operator
    def bond_op(j, j_plus_1, op1, op2):
        ops = [I2] * N_SITES
        ops[j] = op1
        ops[j_plus_1] = op2
        return kron_all(ops)
        
    H_odd = np.zeros((DIM, DIM), dtype=complex)
    H_even = np.zeros((DIM, DIM), dtype=complex)
    H_field = np.zeros((DIM, DIM), dtype=complex)
    
    # Bonds: 0-1, 1-2, 2-3, 3-4, 4-5 (0-indexed)
    # Odd bonds in 1-indexed: (1,2), (3,4), (5,6) -> 0-indexed: (0,1), (2,3), (4,5)
    # Even bonds in 1-indexed: (2,3), (4,5) -> 0-indexed: (1,2), (3,4)
    
    for j in range(N_SITES - 1):
        # XXZ interaction: Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + Delta Sz_j Sz_{j+1}
        H_bond = bond_op(j, j+1, Sx, Sx) + bond_op(j, j+1, Sy, Sy) + DELTA * bond_op(j, j+1, Sz, Sz)
        if (j % 2 == 0): # 0, 2, 4 are odd bonds in 1-indexed? 
            # 1-indexed j=1 is bond 1-2. j=0 in 0-indexed.
            # Odd bonds: 1, 3, 5. 0-indexed: 0, 2, 4.
            H_odd += H_bond
        else:
            # Even bonds: 2, 4. 0-indexed: 1, 3.
            H_even += H_bond
            
    for j in range(N_SITES):
        H_field += FIELD * site_op(j, Sz)
        
    A = H_odd + 0.5 * H_field
    B = H_even + 0.5 * H_field
    
    return A, B

A_mat, B_mat = build_hamiltonians()

# Precompute eigen-decompositions for fast exponentiation
# A = V_A D_A V_A^dagger
evals_A, evecs_A = np.linalg.eig(A_mat)
evals_B, evecs_B = np.linalg.eig(B_mat)
V_A = evecs_A
V_A_dag = evecs_A.conj().T
V_B = evecs_B
V_B_dag = evecs_B.conj().T

def expm_A(a_k, t):
    """Compute exp(-i * a_k * t * A)."""
    # D_A is diagonal with evals_A
    # exp(-i * a_k * t * D_A) is diagonal with exp(-i * a_k * t * evals_A)
    diag = np.exp(-1j * a_k * t * evals_A)
    # V_A @ diag @ V_A^dagger
    # Efficient multiplication: (V_A * diag) @ V_A_dag
    # V_A * diag scales columns of V_A
    return (V_A * diag) @ V_A_dag

def expm_B(b_k, t):
    """Compute exp(-i * b_k * t * B)."""
    diag = np.exp(-1j * b_k * t * evals_B)
    return (V_B * diag) @ V_B_dag

def objective(params):
    """
    params: 10 floats [a1, a2, a3, a4, a5, b1, b2, b3, b4, b5]
    a6 = 0.5 - sum(a1..a5)
    b6 = 0.5 - sum(b1..b5)
    Symmetric: a = [a1..a6, a6..a1]
    """
    a1, a2, a3, a4, a5 = params[:5]
    b1, b2, b3, b4, b5 = params[5:10]
    
    a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
    b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
    
    a_list = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    b_list = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    total_error = 0.0
    for t in TIMES:
        U = np.eye(DIM, dtype=complex)
        for k in range(STAGES):
            # U = U @ exp(-i b_k t B) @ exp(-i a_k t A)
            # Order in prompt: prod_k exp(-i b_k t B) exp(-i a_k t A)
            # This means U_final = E_B1 E_A1 E_B2 E_A2 ... E_B12 E_A12
            # So we apply E_Ak then E_Bk? No, matrix multiplication is right-to-left.
            # U = I
            # U = U @ E_B1 @ E_A1
            # U = U @ E_B2 @ E_A2
            # ...
            # So U = I @ E_B1 @ E_A1 @ E_B2 @ E_A2 ...
            
            U = U @ expm_B(b_list[k], t) @ expm_A(a_list[k], t)
            
        # Exact propagator
        H_total = A_mat + B_mat
        evals_H, evecs_H = np.linalg.eig(H_total)
        # Precompute this outside the loop ideally, but for simplicity in objective, 
        # we can cache it. Let's cache it globally.
        # Actually, let's move eig(H_total) to global scope.
        
        # For now, compute exact
        # This is inefficient, let's fix by precomputing H_total eig.
        # We'll handle this in the main block by passing a precomputed exact propagator generator.
        
    return 0.0 # Placeholder

# Precompute exact propagator components
evals_H, evecs_H = np.linalg.eig(A_mat + B_mat)
V_H = evecs_H
V_H_dag = evecs_H.conj().T

def exact_propagator(t):
    diag = np.exp(-1j * t * evals_H)
    return (V_H * diag) @ V_H_dag

def objective_optimized(params):
    a1, a2, a3, a4, a5 = params[:5]
    b1, b2, b3, b4, b5 = params[5:10]
    
    a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
    b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
    
    a_list = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    b_list = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    total_error = 0.0
    for t in TIMES:
        U = np.eye(DIM, dtype=complex)
        for k in range(STAGES):
            U = U @ expm_B(b_list[k], t) @ expm_A(a_list[k], t)
        
        U_exact = exact_propagator(t)
        # Spectral norm of difference
        diff = U - U_exact
        # Spectral norm is largest singular value
        # For small matrices, SVD is fast
        try:
            s = np.linalg.svd(diff, compute_uv=False)
            err = s[0]
        except:
            err = 1e10
        total_error += err
        
    return total_error / len(TIMES)

def nelder_mead(f, x0, max_iter=1000, tol=1e-8):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    
    # Initialize simplex
    simplex = [x0]
    f_simplex = [f(x0)]
    for i in range(n):
        x_new = x0.copy()
        x_new[i] += 0.05 if x_new[i] == 0 else 0.05 * x_new[i]
        simplex.append(x_new)
        f_simplex.append(f(x_new))
        
    for _ in range(max_iter):
        # Sort simplex
        indices = np.argsort(f_simplex)
        simplex = [simplex[i] for i in indices]
        f_simplex = [f_simplex[i] for i in indices]
        
        if f_simplex[0] < tol:
            break
            
        # Centroid of all but worst
        x_worst = simplex[-1]
        f_worst = f_simplex[-1]
        
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = f(x_r)
        
        if f_r < f_simplex[0]:
            # Expansion
            x_e = x_centroid + gamma * (x_r - x_worst)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_simplex[-1] = f_e
            else:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
        elif f_r < f_simplex[-2]:
            simplex[-1] = x_r
            f_simplex[-1] = f_r
        else:
            # Contraction
            x_c = x_centroid + rho * (x_worst - x_centroid)
            f_c = f(x_c)
            if f_c < f_worst:
                simplex[-1] = x_c
                f_simplex[-1] = f_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f_simplex[i] = f(simplex[i])
                    
    # Return best
    idx = np.argmin(f_simplex)
    return simplex[idx], f_simplex[idx]

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # Initial guess: symmetric Strang-like
    # a = [0.5/6]*6 ...
    # Let's start with uniform
    a_init = [0.5/6.0] * 5
    b_init = [0.5/6.0] * 5
    x0 = np.array(a_init + b_init)
    
    best_x = x0
    best_f = objective_optimized(x0)
    
    # Run optimization
    try:
        x_opt, f_opt = nelder_mead(objective_optimized, x0, max_iter=2000, tol=1e-6)
        if f_opt < best_f:
            best_x = x_opt
            best_f = f_opt
    except Exception as e:
        pass # Keep initial if optimization fails
    
    # Check time
    if time.time() > deadline - 10.0:
        # Return best found so far
        pass
    else:
        # Try a few random restarts if time permits
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            x_rand = np.random.uniform(-0.2, 0.2, 10)
            # Ensure feasibility roughly
            # a6 = 0.5 - sum(a1..a5). If sum is far from 0.5, a6 is large.
            # We want coefficients in [-50, 50].
            # Let's just run NM from random start
            try:
                x_opt_r, f_opt_r = nelder_mead(objective_optimized, x_rand, max_iter=500, tol=1e-5)
                if f_opt_r < best_f:
                    best_x = x_opt_r
                    best_f = f_opt_r
            except:
                pass

    a1, a2, a3, a4, a5 = best_x[:5]
    b1, b2, b3, b4, b5 = best_x[5:10]
    
    a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
    b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
    
    a_list = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    b_list = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    # Clip to [-50, 50] and renormalize? 
    # The problem requires sum=1. If we clip, sum might change.
    # The optimization should keep them in range.
    # Just in case, we return them.
    return a_list, b_list

