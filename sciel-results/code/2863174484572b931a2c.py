import numpy as np
import scipy.linalg
import scipy.optimize
import time

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) 
    for a 6-site open XXZ chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    dim = 2 ** N
    anisotropy = 0.8
    field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    # Identify sites: 0, 1, 2, 3, 4, 5
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    
    # Term: sigma_x_i sigma_x_j + sigma_y_i sigma_y_j + anisotropy * sigma_z_i sigma_z_j
    # Field term: field * sigma_z_i
    
    def bond_term(i, j):
        # sigma_x_i sigma_x_j
        ops_xx = [I2]*N
        ops_xx[i] = sx
        ops_xx[j] = sx
        term_xx = kron_all(ops_xx)
        
        # sigma_y_i sigma_y_j
        ops_yy = [I2]*N
        ops_yy[i] = sy
        ops_yy[j] = sy
        term_yy = kron_all(ops_yy)
        
        # anisotropy * sigma_z_i sigma_z_j
        ops_zz = [I2]*N
        ops_zz[i] = sz
        ops_zz[j] = sz
        term_zz = anisotropy * kron_all(ops_zz)
        
        return term_xx + term_yy + term_zz

    def field_term(i):
        ops_z = [I2]*N
        ops_z[i] = sz
        return field * kron_all(ops_z)

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Add bonds to A and B
    # Odd bonds go to A
    A += bond_term(0, 1)
    A += bond_term(2, 3)
    A += bond_term(4, 5)
    
    # Even bonds go to B
    B += bond_term(1, 2)
    B += bond_term(3, 4)
    
    # Add field: split evenly between A and B
    for i in range(N):
        half_field = 0.5 * field_term(i)
        A += half_field
        B += half_field
        
    return A, B

def coefficients(stages):
    """
    Optimize the coefficients for a 12-stage product formula.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    
    # Pre-compute exact evolution operators
    times = [0.25, 0.5, 1.0]
    U_exacts = [scipy.linalg.expm(-1j * t * H) for t in times]
    
    def objective(x):
        # x has length 2 * (stages - 1)
        # First half is a_1..a_{stages-1}, second half is b_1..b_{stages-1}
        a_free = x[:stages-1]
        b_free = x[stages-1:]
        
        a = np.append(a_free, 1.0 - np.sum(a_free))
        b = np.append(b_free, 1.0 - np.sum(b_free))
        
        error = 0.0
        for i, t in enumerate(times):
            U = np.eye(dim, dtype=complex)
            for k in range(stages):
                # U = U @ exp(-i b_k t B) @ exp(-i a_k t A)
                # Note: The formula is prod_k exp(-i b_k t B) exp(-i a_k t A)
                # Order is k=1..12.
                # So U = (E_b1 E_a1) (E_b2 E_a2) ...
                # Matrix multiplication is associative.
                U = U @ scipy.linalg.expm(-1j * b[k] * t * B) @ scipy.linalg.expm(-1j * a[k] * t * A)
            
            # Spectral norm of difference
            diff = U - U_exacts[i]
            # Use SVD for spectral norm (largest singular value)
            # For 64x64, this is fast.
            error += np.linalg.norm(diff, 2)
        
        return error / len(times)

    # Initial guesses
    # 1. Uniform
    x0_uniform = np.zeros(2 * (stages - 1))
    # 2. Strang-like (approx)
    # For 12 stages, we can't just do Strang directly, but we can start with small values
    x0_strang = np.zeros(2 * (stages - 1))
    # 3. Random
    rng = np.random.default_rng(42)
    
    best_x = x0_uniform
    best_val = objective(best_x)
    
    # List of initializations to try
    inits = [x0_uniform]
    # Add a few random inits
    for _ in range(3):
        # Small random perturbations around uniform
        init = rng.normal(0, 0.05, size=2 * (stages - 1))
        inits.append(init)
    
    for x0 in inits:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = scipy.optimize.minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception as e:
            pass
            
    # Extract final coefficients
    a_free = best_x[:stages-1]
    b_free = best_x[stages-1:]
    
    a = np.append(a_free, 1.0 - np.sum(a_free))
    b = np.append(b_free, 1.0 - np.sum(b_free))
    
    # Clip to [-50, 50] just in case, though they should be small
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Ensure sums are exactly 1 by adjusting the last element slightly if needed 
    # (floating point errors)
    a[-1] = 1.0 - np.sum(a[:-1])
    b[-1] = 1.0 - np.sum(b[:-1])
    
    return list(map(float, a)), list(map(float, b))

# EVOLVE-BLOCK-END
