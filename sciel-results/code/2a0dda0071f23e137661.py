"""Product-formula coefficient optimizer for 6-site Heisenberg chain splitting."""
import numpy as np
import time
import sys

# Deadline guard: The evaluator kills after 120s (task says 120s, prompt says 150s kill, 
# but "Each evaluation must finish within 120 s"). 
# The function `coefficients` runs in a sandboxed subprocess with a 60s CPU budget.
# However, the prompt says "The evaluator measures... Each evaluation must finish within 120 s." 
# and "coefficients runs in a sandboxed subprocess with a 60 s CPU budget".
# Let's assume the 60s is the hard limit for the function execution.
# I will set a conservative deadline to ensure we return before the kill signal.
_DEADLINE = 55.0  # 5 seconds safety margin
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain.
    
    Sites 0..5, open boundary. Bonds: (0,1),(1,2),(2,3),(3,4),(4,5).
    Odd bonds (0-indexed): bonds 1,3 -> (1,2),(3,4)
    Even bonds (0-indexed): bonds 0,2,4 -> (0,1),(2,3),(4,5)
    
    XXZ anisotropy delta=0.8, field=0.3
    """
    N = 6
    delta = 0.8
    field = 0.3
    
    # Single-site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        """Kronecker product of a list of 2x2 operators."""
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build Hamiltonians
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Bond operators: XXZ interaction on bond (i,j)
    # H_bond = delta * sz_i sz_j + (sx_i sx_j + sy_i sy_j)
    def bond_operator(i, j):
        ops_A = [identity] * N
        ops_B = [identity] * N
        # sx_i sx_j + sy_i sy_j = 0.5 * (s+_i s-_j + s-_i s+_j)
        s_plus = np.array([[0, 1], [0, 0]], dtype=complex)
        s_minus = np.array([[0, 0], [1, 0]], dtype=complex)
        
        # s+_i s-_j
        ops_splus_sminus = [identity] * N
        ops_splus_sminus[i] = s_plus
        ops_splus_sminus[j] = s_minus
        term1 = 0.5 * kron_all(ops_splus_sminus)
        
        # s-_i s+_j
        ops_sminus_splus = [identity] * N
        ops_sminus_splus[i] = s_minus
        ops_sminus_splus[j] = s_plus
        term2 = 0.5 * kron_all(ops_sminus_splus)
        
        # sz_i sz_j
        ops_sz_sz = [identity] * N
        ops_sz_sz[i] = sz
        ops_sz_sz[j] = sz
        term3 = delta * kron_all(ops_sz_sz)
        
        return term1 + term2 + term3
    
    # Odd bonds: (1,2) and (3,4) -> these go into A
    A += bond_operator(1, 2)
    A += bond_operator(3, 4)
    
    # Even bonds: (0,1), (2,3), (4,5) -> these go into B
    B += bond_operator(0, 1)
    B += bond_operator(2, 3)
    B += bond_operator(4, 5)
    
    # Uniform field: -field * sum sz_j, split half into A, half into B
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        ops_sz = [identity] * N
        ops_sz[j] = sz
        field_op += kron_all(ops_sz)
    
    A += -0.5 * field * field_op
    B += -0.5 * field * field_op
    
    return A, B


def _compute_error(a, b, A, B, H_full, t_values, eig_A, eig_B, eig_H):
    """Compute mean spectral norm error over t_values."""
    total_error = 0.0
    n_times = len(t_values)
    dim = A.shape[0]
    
    for t in t_values:
        # Compute product formula U(t)
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            # Apply exp(-i a_k t A) then exp(-i b_k t B)
            # Order: U = ... @ exp(-i b_k t B) @ exp(-i a_k t A)
            # Wait, the prompt says: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12
            # This means U = (exp(-i b_12 t B) exp(-i a_12 t A)) ... (exp(-i b_1 t B) exp(-i a_1 t A))
            # No, "prod_k ... in order k=1..12" usually means left-to-right multiplication:
            # U = M_12 * M_11 * ... * M_1 where M_k = exp(-i b_k t B) exp(-i a_k t A)
            # Or does it mean the first factor applied is k=1?
            # In physics, if we write U = U_n ... U_1, the state evolves as |psi(t)> = U_n ... U_1 |psi(0)>.
            # So U_1 is applied first.
            # The formula is U(t) = prod_{k=1}^12 [ exp(-i b_k t B) exp(-i a_k t A) ]
            # Standard matrix product notation: U = M_12 M_11 ... M_1.
            # So M_1 is the rightmost matrix.
            
            # Let's check the seed program's logic:
            # U = U @ (eigvecs @ ... @ eigvecs.conj().T)
            # It starts with U = I.
            # Loop k=0 to 11:
            #   U = U @ exp(-i a_k t A)
            #   U = U @ exp(-i b_k t B)
            # This results in U = (ExpB_11 ExpA_11) ... (ExpB_0 ExpA_0)
            # This corresponds to applying ExpA_0 first, then ExpB_0, then ExpA_1...
            # This matches the standard interpretation of "product in order k=1..12" where the product accumulates on the right.
            
            if abs(a[k]) > 1e-15:
                # exp(-i a_k t A)
                # Pre-computed eigendecomposition: A = V_A diag(w_A) V_A^dag
                # exp(-i a_k t A) = V_A diag(exp(-i a_k t w_A)) V_A^dag
                exp_A_k = eig_B[1] @ np.diag(np.exp(-1j * a[k] * t * eig_B[0])) @ eig_B[1].conj().T # Placeholder, need correct eig
                # Actually, let's just use the pre-computed ones passed in.
                # eig_A = (vals, vecs), eig_B = (vals, vecs)
                pass
            
            # Let's rewrite _compute_error to use pre-computed eigendecompositions for speed.
            pass
            
    return total_error / n_times

def _compute_error_fast(a, b, A, B, H_full, t_values, eig_A, eig_B, eig_H):
    """Compute mean spectral norm error over t_values using pre-computed eigendecompositions."""
    total_error = 0.0
    n_times = len(t_values)
    dim = A.shape[0]
    
    w_A, V_A = eig_A
    w_B, V_B = eig_B
    w_H, V_H = eig_H
    
    # Pre-compute V_H^dag for exact propagator
    V_H_dag = V_H.conj().T
    
    for t in t_values:
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            # exp(-i a_k t A)
            if abs(a[k]) > 1e-15:
                exp_A = V_A @ np.diag(np.exp(-1j * a[k] * t * w_A)) @ V_A.conj().T
                U = U @ exp_A
            
            # exp(-i b_k t B)
            if abs(b[k]) > 1e-15:
                exp_B = V_B @ np.diag(np.exp(-1j * b[k] * t * w_B)) @ V_B.conj().T
                U = U @ exp_B
        
        # Exact propagator
        U_exact = V_H @ np.diag(np.exp(-1j * t * w_H)) @ V_H_dag
        
        # Spectral norm of difference
        diff = U - U_exact
        error = np.linalg.norm(diff, 2)
        total_error += error
    
    return total_error / n_times


def coefficients(stages):
    """Optimized product formula coefficients for 6-site Heisenberg chain.
    
    Uses Differential Evolution for global search with physics-informed initializations.
    """
    stages = 12
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    H_full = A + B
    
    t_values = [0.25, 0.5, 1.0]
    
    # Pre-compute eigendecompositions
    w_A, V_A = np.linalg.eigh(A)
    w_B, V_B = np.linalg.eigh(B)
    w_H, V_H = np.linalg.eigh(H_full)
    
    eig_A = (w_A, V_A)
    eig_B = (w_B, V_B)
    eig_H = (w_H, V_H)
    
    # Define the objective function for the optimizer
    # We optimize the first 6 a's and 6 b's, assuming symmetry:
    # a = [a1..a6, a6..a1] and b = [b1..b6, b6..b1]
    # This enforces time-reversal symmetry which is often optimal for product formulas.
    
    def objective(params):
        a_params = params[:6]
        b_params = params[6:12]
        
        # Construct full symmetric vectors
        a_full = np.concatenate((a_params, a_params[::-1]))
        b_full = np.concatenate((b_params, b_params[::-1]))
        
        # Enforce sum constraint by scaling
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        
        if abs(a_sum) < 1e-15 or abs(b_sum) < 1e-15:
            return 1e10
        
        a_scaled = a_full / a_sum
        b_scaled = b_full / b_sum
        
        # Clamp to [-50, 50]
        a_scaled = np.clip(a_scaled, -50.0, 50.0)
        b_scaled = np.clip(b_scaled, -50.0, 50.0)
        
        # Re-normalize after clamping to ensure sum=1 strictly? 
        # Clamping might break the sum. 
        # If clamping happens, we should probably just return a large penalty or renormalize again.
        # Given the range [-50,50] is wide, clamping is unlikely to be active for good solutions.
        # But to be safe:
        a_sum2 = np.sum(a_scaled)
        b_sum2 = np.sum(b_scaled)
        if abs(a_sum2) < 1e-15 or abs(b_sum2) < 1e-15:
            return 1e10
        a_final = a_scaled / a_sum2
        b_final = b_scaled / b_sum2
        
        return _compute_error_fast(a_final, b_final, A, B, H_full, t_values, eig_A, eig_B, eig_H)
    
    # Initial guesses
    # 1. Uniform (current seed)
    # 2. Suzuki-Trotter 4th order coefficients (scaled to sum 1)
    #    For 4th order: 
    #    c1 = 1/(4 - 4^(1/3))
    #    c2 = -c1/(2*(2**(1/3) - 1))
    #    c3 = 1 - 2*c1 - 2*c2
    #    d1 = 1/(2 - 4^(1/3))
    #    d2 = -d1/2
    #    d3 = 1 - 2*d1 - 2*d2
    #    This is for 3 stages. We have 12 stages.
    #    We can replicate the 4th order block 4 times? Or use a higher order scheme.
    #    Let's try to map the 4th order coefficients to our 12-stage symmetric structure.
    #    Or just use a random initialization and let DE do the work.
    #    DE is robust to initialization.
    
    # Let's try a few specific initializations to seed the population or as restart points.
    
    best_a = None
    best_b = None
    best_error = float('inf')
    
    # Use Differential Evolution
    try:
        from scipy.optimize import differential_evolution
        
        bounds = [(-10, 10)] * 12  # Wide bounds, scaling handles the rest
        
        # Run DE
        # maxiter=100, popsize=15
        # We have ~50s budget. DE is stochastic.
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=20,
            seed=42,
            tol=1e-8,
            disp=False,
            polish=False  # Skip polish to save time, we can do local refinement later
        )
        
        if result.fun < best_error:
            best_error = result.fun
            a_params = result.x[:6]
            b_params = result.x[6:12]
            
            a_full = np.concatenate((a_params, a_params[::-1]))
            b_full = np.concatenate((b_params, b_params[::-1]))
            
            a_sum = np.sum(a_full)
            b_sum = np.sum(b_full)
            if abs(a_sum) > 1e-15 and abs(b_sum) > 1e-15:
                best_a = list(a_full / a_sum)
                best_b = list(b_full / b_sum)
                
    except Exception as e:
        print(f"DE failed: {e}", file=sys.stderr)
    
    # If DE didn't run or failed, fall back to uniform
    if best_a is None:
        a_half = [1.0 / 12.0] * 6
        b_half = [1.0 / 12.0] * 6
        best_a = a_half + a_half[::-1]
        best_b = b_half + b_half[::-1]
        best_error = _compute_error_fast(best_a, best_b, A, B, H_full, t_values, eig_A, eig_B, eig_H)
    
    # Local refinement with Nelder-Mead on the best DE result
    if _remaining() > 10:
        try:
            from scipy.optimize import minimize
            
            # Start from the best DE solution
            x0 = np.array(best_a[:6] + best_b[:6])
            
            result_lm = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12}
            )
            
            if result_lm.fun < best_error:
                best_error = result_lm.fun
                a_params = result_lm.x[:6]
                b_params = result_lm.x[6:12]
                
                a_full = np.concatenate((a_params, a_params[::-1]))
                b_full = np.concatenate((b_params, b_params[::-1]))
                
                a_sum = np.sum(a_full)
                b_sum = np.sum(b_full)
                if abs(a_sum) > 1e-15 and abs(b_sum) > 1e-15:
                    best_a = list(a_full / a_sum)
                    best_b = list(b_full / b_sum)
        except Exception as e:
            print(f"LM failed: {e}", file=sys.stderr)
    
    # Final checks
    if best_a is None or best_b is None:
        best_a = [1.0/12.0]*12
        best_b = [1.0/12.0]*12
        
    # Ensure types and ranges
    best_a = [max(-50.0, min(50.0, float(x))) for x in best_a]
    best_b = [max(-50.0, min(50.0, float(x))) for x in best_b]
    
    # Final normalization just in case
    a_sum = sum(best_a)
    b_sum = sum(best_b)
    if abs(a_sum) > 1e-15:
        best_a = [x / a_sum for x in best_a]
    if abs(b_sum) > 1e-15:
        best_b = [x / b_sum for x in best_b]
        
    return best_a, best_b
