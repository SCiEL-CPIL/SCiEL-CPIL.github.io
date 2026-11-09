import numpy as np
import scipy.optimize as opt
import scipy.linalg as sla

def coefficients(stages):
    """
    Optimizes the coefficients of a 12-stage product formula to minimize
    the spectral norm error for the specific Heisenberg chain problem.
    """
    
    # 1. Construct the Hamiltonian parts A and B
    # 6-site chain, spin-1/2
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id2 = np.eye(2, dtype=complex)
    
    # Build tensor products for 6 sites
    # We need operators for each bond and the field
    # H = sum_{j=1}^5 (sx_j sx_{j+1} + sy_j sy_{j+1} + 0.8 sz_j sz_{j+1}) + 0.3 sum_{j=1}^6 sz_j
    
    # Helper to get operator on site j (1-indexed in physics, 0-indexed in code)
    def get_op(site, op):
        ops = [id2] * 6
        ops[site] = op
        result = ops[0]
        for i in range(1, 6):
            result = np.kron(result, ops[i])
        return result
    
    # Build A (odd bonds: 1-2, 3-4, 5-6 -> indices 0-1, 2-3, 4-5)
    # and B (even bonds: 2-3, 4-5 -> indices 1-2, 3-4)
    # Field is split: 0.5 * h * sum sz_j in both A and B
    
    A = np.zeros((64, 64), dtype=complex)
    B = np.zeros((64, 64), dtype=complex)
    
    h = 0.3
    gamma = 0.8
    
    for j in range(5):
        # Bond between j and j+1 (0-indexed)
        # sx_j sx_{j+1} + sy_j sy_{j+1} + gamma sz_j sz_{j+1}
        bond_op = np.zeros((64, 64), dtype=complex)
        
        # sx_j sx_{j+1}
        ops = [id2]*6
        ops[j] = sx; ops[j+1] = sx
        term = ops[0]
        for i in range(1,6): term = np.kron(term, ops[i])
        bond_op += term
        
        # sy_j sy_{j+1}
        ops = [id2]*6
        ops[j] = sy; ops[j+1] = sy
        term = ops[0]
        for i in range(1,6): term = np.kron(term, ops[i])
        bond_op += term
        
        # gamma sz_j sz_{j+1}
        ops = [id2]*6
        ops[j] = sz; ops[j+1] = sz
        term = ops[0]
        for i in range(1,6): term = np.kron(term, ops[i])
        bond_op += gamma * term
        
        if j % 2 == 0: # Odd bond (1-2 is j=0, 3-4 is j=2, 5-6 is j=4)
            A += bond_op
        else: # Even bond (2-3 is j=1, 4-5 is j=3)
            B += bond_op
            
    # Add half field to both
    field_op = np.zeros((64, 64), dtype=complex)
    for j in range(6):
        ops = [id2]*6
        ops[j] = sz
        term = ops[0]
        for i in range(1,6): term = np.kron(term, ops[i])
        field_op += term
        
    A += 0.5 * h * field_op
    B += 0.5 * h * field_op
    
    # 2. Define the cost function
    # We optimize 22 variables: a[0..10] and b[0..10]
    # a[11] and b[11] are determined by the sum constraint.
    
    def cost(params):
        a_vars = params[:11]
        b_vars = params[11:22]
        
        # Complete the vectors
        a = np.zeros(12)
        b = np.zeros(12)
        a[:11] = a_vars
        b[:11] = b_vars
        a[11] = 1.0 - np.sum(a_vars)
        b[11] = 1.0 - np.sum(b_vars)
        
        # Check bounds (soft penalty or hard clip? L-BFGS-B handles bounds)
        # We will use bounds in the optimizer
        
        # Compute U(t) for t = 0.25
        t = 0.25
        U = np.eye(64, dtype=complex)
        
        # Pre-compute exponentials? No, they depend on coefficients.
        # But A and B are constant.
        # exp(-i a_k t A) and exp(-i b_k t B)
        
        # To speed up, we can diagonalize A and B once
        # A is Hermitian, so A = V_A D_A V_A^dagger
        # exp(-i a t A) = V_A exp(-i a t D_A) V_A^dagger
        
        # Let's do this outside the cost function if possible, but cost function is called many times.
        # We can pass eigenvectors via closure.
        
        # For now, compute directly or use pre-computed eigendecomposition.
        # Let's assume we have eig_A, eigvals_A, eig_B, eigvals_B pre-computed.
        
        # This is inefficient if done inside. Let's restructure.
        
        # We will define the cost function inside a scope where eigendecompositions are available.
        
        # Placeholder for now, will be replaced in the actual code structure.
        
        # For the sake of the SEARCH/REPLACE, I will write the full optimized code.
        
        pass

    # Since I need to provide SEARCH/REPLACE, I will replace the entire function body with a version that performs the optimization.
    
    # Pre-compute eigendecompositions for A and B
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    # Exact propagator
    H_total = A + B
    eigvals_H, eigvecs_H = np.linalg.eigh(H_total)
    
    def cost_func(params):
        a_vars = params[:11]
        b_vars = params[11:22]
        
        a = np.zeros(12)
        b = np.zeros(12)
        a[:11] = a_vars
        b[:11] = b_vars
        a[11] = 1.0 - np.sum(a_vars)
        b[11] = 1.0 - np.sum(b_vars)
        
        # Clip to bounds to avoid huge values causing overflow in exp
        a = np.clip(a, -50.0, 50.0)
        b = np.clip(b, -50.0, 50.0)
        
        # Recalculate sum after clipping? If clipped, sum might not be 1.
        # It's better to rely on L-BFGS-B bounds to keep them in range and sum constraint handled by parameterization.
        # However, if we clip, the sum constraint is broken.
        # Better to not clip inside cost, but set bounds in optimizer.
        
        t = 0.25
        U = np.eye(64, dtype=complex)
        
        for k in range(12):
            # exp(-i b_k t B)
            if b[k] != 0:
                exp_B = eigvecs_B @ np.diag(np.exp(-1j * b[k] * t * eigvals_B)) @ eigvecs_B.conj().T
            else:
                exp_B = np.eye(64, dtype=complex)
            
            # exp(-i a_k t A)
            if a[k] != 0:
                exp_A = eigvecs_A @ np.diag(np.exp(-1j * a[k] * t * eigvals_A)) @ eigvecs_A.conj().T
            else:
                exp_A = np.eye(64, dtype=complex)
                
            U = exp_B @ exp_A @ U
            
        # Exact
        exact = eigvecs_H @ np.diag(np.exp(-1j * t * eigvals_H)) @ eigvecs_H.conj().T
        
        diff = U - exact
        # Spectral norm
        # For small matrices, svd is fine
        # Or use norm 2
        err = np.linalg.norm(diff, 2)
        return err

    # Initial guess: uniform
    a_init = np.ones(11) / 12.0
    b_init = np.ones(11) / 12.0
    x0 = np.concatenate([a_init, b_init])
    
    # Bounds
    bounds = [(-50.0, 50.0)] * 22
    
    # Optimize
    result = opt.minimize(cost_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
    
    a_opt = result.x[:11]
    b_opt = result.x[11:22]
    
    a = np.zeros(12)
    b = np.zeros(12)
    a[:11] = a_opt
    b[:11] = b_opt
    a[11] = 1.0 - np.sum(a_opt)
    b[11] = 1.0 - np.sum(b_opt)
    
    # Final safety check
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    
    # Normalize again just in case
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    if abs(sb) > 1e-15:
        b = b / sb
        
    return list(a), list(b)
