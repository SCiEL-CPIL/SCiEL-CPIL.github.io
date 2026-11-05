import time
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin before 150s kill

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    Each bond: gamma/2*(Sx*Sx + Sy*Sy) + 1/2*Sz*Sz + h/2*(Sz_i + Sz_j)
    Split: A = sum over odd bonds (1-2, 3-4, 5-6), B = sum over even bonds (2-3, 4-5)
    Each includes half the uniform field."""
    N = 6
    gamma = 0.8
    h = 0.3
    
    # Spin-1/2 operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def op_on_site(op, site):
        """Apply op on a specific site in N-site chain."""
        result = np.array([[1]], dtype=complex)
        for i in range(N):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2, dtype=complex))
        return result
    
    def bond_op(i, j):
        """Two-spin bond operator between sites i and j (0-indexed).
        gamma/2*(Sx_i*Sx_j + Sy_i*Sy_j) + 1/2*Sz_i*Sz_j + h/2*(Sz_i + Sz_j)"""
        Sxi = op_on_site(Sx, i)
        Sxj = op_on_site(Sx, j)
        Syi = op_on_site(Sy, i)
        Syj = op_on_site(Sy, j)
        Szi = op_on_site(Sz, i)
        Szj = op_on_site(Sz, j)
        
        return (gamma/2)*(Sxi @ Sxj + Syi @ Syj) + 0.5*(Szi @ Szj) + h/2*(Szi + Szj)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    A = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    B = bond_op(1, 2) + bond_op(3, 4)
    
    return A, B

class ProductFormulaOptimizer:
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.dim = A.shape[0]
        self.times = [0.25, 0.5, 1.0]
        
        # Precompute eigen-decompositions for fast exponentials
        # A and B are Hermitian, so eigenvectors are unitary
        self.eigvals_A, self.eigvecs_A = la.eigh(A)
        self.eigvals_B, self.eigvecs_B = la.eigh(B)
        
        # Precompute exact propagators
        self.exact_propagators = {}
        for t in self.times:
            self.exact_propagators[t] = la.expm(-1j * t * (A + B))
            
        # Cache for exponential evaluations
        # Key: (H_type, coeff, t) -> Matrix
        # Since coeff varies, we can't cache across optimization steps easily unless we quantize,
        # but we can reuse within a single error evaluation if coeffs repeat.
        # Better: just compute directly. Eigh is O(N^3) ~ 250k ops, very fast.
        
    def _expm_scaled(self, eigvals, eigvecs, c, t):
        """Compute exp(-i * c * t * H) using precomputed eigh."""
        # exp(-i * c * t * Lambda) is diagonal
        diag = np.exp(-1j * c * t * eigvals)
        # V @ diag @ V^H
        # eigvecs is V, so V^H is eigvecs.conj().T
        # Result = (eigvecs * diag) @ eigvecs.conj().T
        # (eigvecs * diag) scales columns of eigvecs
        scaled_vecs = eigvecs * diag
        return scaled_vecs @ eigvecs.conj().T

    def _compute_error(self, a, b):
        """Compute mean spectral-norm error for given coefficients a, b."""
        total_err = 0.0
        I = np.eye(self.dim, dtype=complex)
        
        for t in self.times:
            U = I.copy()
            for k in range(12):
                # exp(-i b_k t B)
                if abs(b[k]) > 1e-15:
                    U = self._expm_scaled(self.eigvals_B, self.eigvecs_B, b[k], t) @ U
                # exp(-i a_k t A)
                if abs(a[k]) > 1e-15:
                    U = self._expm_scaled(self.eigvals_A, self.eigvecs_A, a[k], t) @ U
            
            U_exact = self.exact_propagators[t]
            err = la.norm(U - U_exact, 2)
            total_err += err
        
        return total_err / 3.0

    def _objective(self, x):
        """Objective function for optimizer.
        x contains 22 variables: a_1..a_11, b_1..b_11.
        a_12 = 1 - sum(a_1..a_11)
        b_12 = 1 - sum(b_1..b_11)
        """
        a = np.array(x[:11])
        b = np.array(x[11:22])
        
        # Construct full arrays
        a_full = np.zeros(12)
        b_full = np.zeros(12)
        a_full[:11] = a
        a_full[11] = 1.0 - np.sum(a)
        
        b_full[:11] = b
        b_full[11] = 1.0 - np.sum(b)
        
        # Check bounds (loose check, optimizer handles bounds)
        # If bounds are violated significantly, return large penalty
        if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10
            
        return self._compute_error(a_full, b_full)

    def optimize(self, initial_guess_a, initial_guess_b):
        """Run optimization."""
        # Convert initial guess to 22-dim vector
        x0 = np.zeros(22)
        x0[:11] = initial_guess_a[:11]
        x0[11:22] = initial_guess_b[:11]
        
        # Bounds for the 22 free variables
        # a_i, b_i in [-50, 50]
        bounds = [(-50.0, 50.0)] * 22
        
        # We also need to ensure the 12th element stays in bounds.
        # sum(a_1..a_11) must be such that 1 - sum is in [-50, 50]
        # So sum(a_1..a_11) in [-49, 51]. This is hard to enforce with simple bounds on individual vars.
        # However, typical optimal coefficients are small (O(1/12)).
        # We'll rely on the optimizer staying in a reasonable region.
        
        start_time = time.time()
        best_x = x0.copy()
        best_f = self._objective(x0)
        
        # Use L-BFGS-B
        # maxiter is a soft limit, we check time
        while time.time() < _DEADLINE:
            try:
                res = opt.minimize(
                    self._objective, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                else:
                    # If it didn't improve, break
                    break
            except Exception as e:
                print(f"Optimization error: {e}")
                break
            
            # If we used a lot of time, stop
            if time.time() > _DEADLINE - 10:
                break
                
        # Extract final a, b
        a_final = np.zeros(12)
        b_final = np.zeros(12)
        a_final[:11] = best_x[:11]
        a_final[11] = 1.0 - np.sum(a_final[:11])
        
        b_final[:11] = best_x[11:22]
        b_final[11] = 1.0 - np.sum(b_final[:11])
        
        # Clip to valid range just in case
        a_final = np.clip(a_final, -50.0, 50.0)
        b_final = np.clip(b_final, -50.0, 50.0)
        
        # Renormalize to ensure sum is exactly 1 (floating point drift)
        if np.sum(a_final) != 0:
             a_final = a_final / np.sum(a_final)
        if np.sum(b_final) != 0:
             b_final = b_final / np.sum(b_final)
             
        return a_final, b_final, best_f

def coefficients(stages):
    """Return optimized (a, b) lists of length `stages` (12)."""
    if stages != 12:
        # Fallback: simple uniform splitting
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # 1. Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # 2. Initialize Optimizer
    optimizer = ProductFormulaOptimizer(A, B)
    
    # 3. Define Initial Guess
    # Symmetric 2nd order split:
    # Blocks: [A/2, B, A/2] repeated 4 times? No, that's 12 stages.
    # Stage 1: A/2, Stage 2: B, Stage 3: A/2 ...
    # Wait, the formula is prod exp(-i b_k t B) exp(-i a_k t A).
    # Order is B then A.
    # Standard Strang: exp(-i dt/2 A) exp(-i dt B) exp(-i dt/2 A).
    # Our order is B then A.
    # So: exp(-i b1 t B) exp(-i a1 t A) ...
    # Let's try:
    # k=1: b=0, a=1/8 (A)
    # k=2: b=1/4, a=0 (B)
    # k=3: b=0, a=1/8 (A)
    # k=4: b=0, a=1/8 (A)
    # k=5: b=1/4, a=0 (B)
    # k=6: b=0, a=1/8 (A)
    # k=7: b=0, a=1/8 (A)
    # k=8: b=1/4, a=0 (B)
    # k=9: b=0, a=1/8 (A)
    # k=10: b=0, a=1/8 (A)
    # k=11: b=1/4, a=0 (B)
    # k=12: b=0, a=1/8 (A)
    # Sum a = 8 * 1/8 = 1. Sum b = 4 * 1/4 = 1.
    
    init_a = np.array([1/8, 0, 1/8, 1/8, 0, 1/8, 1/8, 0, 1/8, 1/8, 0, 1/8])
    init_b = np.array([0, 1/4, 0, 0, 1/4, 0, 0, 1/4, 0, 0, 1/4, 0])
    
    # 4. Optimize
    a_opt, b_opt, err = optimizer.optimize(init_a, init_b)
    
    # 5. Return lists
    return list(a_opt), list(b_opt)
